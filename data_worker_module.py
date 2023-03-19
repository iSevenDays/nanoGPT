import os
import pickle
import re
import string
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import List

from langdetect import detect_langs
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

english_stop_words = set(stopwords.words('english'))
russian_stop_words = set(stopwords.words('russian'))
import msgpack
from datasets import IterableDataset
from tqdm import tqdm
from transformers import GPT2Tokenizer
from nltk.stem import WordNetLemmatizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
BUFFER_SIZE = 10000
block_size = 1024
MAX_LENGTH = block_size
import random


class TextRelevanceChecker(ABC):
    @abstractmethod
    def is_relevant(self, text: str) -> bool:
        pass


class IrrelevantChecker(TextRelevanceChecker):
    def __init__(self, is_irrelevant_function):
        self.is_irrelevant_function = is_irrelevant_function

    def is_relevant(self, text: str) -> bool:
        return not self.is_irrelevant_function(text)


def load_from_file(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return msgpack.load(f, use_list=True)
    except msgpack.exceptions.ExtraData:  # If MessagePack fails, try loading with pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)

class CachedDataLoader:
    def __init__(self, datasets: List[str], train_queue, validation_queue,
                 parallel: bool = True,
                 cache_dir='cache'):
        self.validation_dataset = None
        self.train_dataset = None
        self.parallel = parallel
        self.train_queue = train_queue
        self.validation_queue = validation_queue
        self.debug = False
        self.parallel = parallel
        self.cache_dir = cache_dir
        self.datasets = datasets

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Initialize datasets and create cache files for each dataset
        self.init_default_datasets()

    def init_default_datasets(self):
        for dataset_name in self.datasets:
            self.init_datasets(dataset_name)

    def init_datasets(self, dataset_name):
        # Create cache files for each dataset
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            raise Exception('cache not found')

        train_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_train_cache.pkl")
        validation_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_validation_cache.pkl")

        self.load_items_from_cache(self.train_queue, train_cache_file)
        self.load_items_from_cache(self.validation_queue, validation_cache_file)
        print(f"{dataset_name} Train queue length: {self.train_queue.qsize()}")
        print(f"{dataset_name} Validation queue length: {self.validation_queue.qsize()}")

    def get_random_item(self, queue: Queue):
        queue_list = list(queue.queue)
        random_item = random.choice(queue_list)
        queue.queue.remove(random_item)
        return random_item

    def get_train_item(self, timeout=None):
        return self.get_item(self.train_queue, timeout=timeout)

    def get_validation_item(self, timeout=None):
        return self.get_item(self.validation_queue, timeout=timeout)

    def get_item(self, queue, timeout=None, random_item=False):
        if queue.qsize() == 0:
            print('Will reload datasets, reason: queue is empty')
            self.init_default_datasets()
        if random_item:
            item = self.get_random_item(queue)
        else:
            item = queue.get(timeout=timeout)

        return item

    def load_items_from_cache(self, queue, cache_file_path):
        cached_data = load_from_file(cache_file_path)
        if cached_data:
            random.shuffle(cached_data)
            for chunk_ids in cached_data:
                queue.put(chunk_ids)


class DataWorker:
    def __init__(self, relevance_checker: TextRelevanceChecker,
                 datasets: dict[str, IterableDataset], train_queue, validation_queue,
                 block_size: int = 1024, buffer_size: int = 100_000, parallel: bool = True,
                 train_queue_max_size=500,
                 validation_queue_max_size=500,
                 prefetch_items_count=1000000,
                 cache_dir='cache',
                 load_cached_data=True):
        self.validation_dataset = None
        self.train_dataset = None
        self.relevance_checker = relevance_checker
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.parallel = parallel
        self.train_queue = train_queue
        self.validation_queue = validation_queue
        self.train_queue_max_size = train_queue_max_size
        self.validation_queue_max_size = validation_queue_max_size
        self.running = False
        self.debug = False
        self.load_cached_data = load_cached_data
        self.prefetch_items_count = prefetch_items_count
        self.validation_datasets = {}
        self.train_datasets = {}
        self.relevance_checker = relevance_checker
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.parallel = parallel
        self.cache_dir = cache_dir
        self.train_cache_files = {}
        self.validation_cache_files = {}
        self.datasets = datasets

        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Initialize datasets and create cache files for each dataset
        self.init_default_datasets()

    def init_default_datasets(self):
        for dataset_name, dataset in self.datasets.items():
            self.init_datasets(dataset_name, dataset)

    def init_datasets(self, dataset_name, dataset):
        SEED = 42
        TRAIN_RATIO = 0.8

        train_dataset = dataset.filter(
            lambda example: self.filter_data(example, SEED, TRAIN_RATIO)
        )
        validation_dataset = dataset.filter(
            lambda example: not self.filter_data(example, SEED, TRAIN_RATIO)
        )

        self.train_datasets[dataset_name] = train_dataset
        self.validation_datasets[dataset_name] = validation_dataset

        # Create cache files for each dataset
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        train_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_train_cache.pkl")
        validation_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_validation_cache.pkl")

        if self.load_cached_data:
            self.load_items_from_cache(self.train_queue, train_cache_file)
            self.load_items_from_cache(self.validation_queue, validation_cache_file)
            print(f"{dataset_name} Train queue length: {self.train_queue.qsize()}")
            print(f"{dataset_name} Validation queue length: {self.validation_queue.qsize()}")

        self.train_cache_files[dataset_name] = train_cache_file
        self.validation_cache_files[dataset_name] = validation_cache_file

    def filter_data(self, example, seed, train_ratio):
        random.seed(example['text'] + str(seed))
        return random.random() < train_ratio

    def chunk_text(self, text, block_size):
        # a more context-aware approach. For example, split the text based on complete sentences.
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        chunks = []

        chunk = []
        chunk_len = 0

        for sentence in sentences:
            sentence_len = len(sentence)
            if chunk_len + sentence_len <= block_size:
                chunk.append(sentence)
                chunk_len += sentence_len
            else:
                chunks.append(' '.join(chunk))
                chunk = [sentence]
                chunk_len = sentence_len

        if chunk:
            chunks.append(' '.join(chunk))

        return chunks

    def is_relevant_chunk(self, chunk):
        if not self.relevance_checker.is_relevant(chunk):
            return None
        ids = tokenizer.encode(chunk, truncation=True)
        ids.append(tokenizer.eos_token_id)
        return ids

    def process_texts(self, texts):
        all_chunks = [list(self.chunk_text(text, self.block_size)) for text in texts]
        all_chunks_flat = [chunk for chunks in all_chunks for chunk in chunks]
        with ThreadPoolExecutor() as executor:
            relevant_chunks_ids = list(filter(None, executor.map(self.is_relevant_chunk, all_chunks_flat)))

        return relevant_chunks_ids

    def has_cyrillic(self, text):
        return bool(re.search('[\u0400-\u04FF]', text))

    def clean_text(self, text):
        # Remove or replace irrelevant or noisy information, such as non-textual characters or HTML tags
        # Example:
        cleaned_text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
        cleaned_text = re.sub(r'http\S+', ' ', cleaned_text)  # Remove URLs
        cleaned_text = re.sub(r'@[^\s]+', ' ', cleaned_text)  # Remove mentions
        cleaned_text = re.sub(r'\n', ' ', cleaned_text)  # Remove newline characters
        cleaned_text = re.sub(r'[\r\u200b]', '', cleaned_text)  # Remove zero-width space and carriage return characters
        cleaned_text = re.sub(r'["“”]', '"', cleaned_text)  # Replace double quotes with standard double quotes
        cleaned_text = re.sub(r"[‘’´`]", "'", cleaned_text)  # Replace single quotes and apostrophes with standard single quotes
        cleaned_text = re.sub(r'RT[\s]+', ' ', cleaned_text)  # Remove retweet prefix
        cleaned_text = re.sub(r'#', '', cleaned_text)  # Remove hashtag symbol
        cleaned_text = re.sub(r'[^\w\s-]', ' ', cleaned_text)  # Remove punctuation and special characters except hyphens
        cleaned_text = re.sub(r'\d', ' ', cleaned_text)  # Remove digits
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)  # Remove extra whitespace
        cleaned_text = cleaned_text.strip()  # Remove leading and trailing whitespace

        # Remove additional types of non-textual content that might be present in your data
        cleaned_text = re.sub(r'\b\d{10}\b', ' ', cleaned_text)  # Remove phone numbers
        cleaned_text = re.sub(r'\b@[^\s]+\b', ' ', cleaned_text)  # Remove social media handles

        # Filter out sentences with overly complex vocabulary or rare words
        tokens = word_tokenize(cleaned_text.lower())
        cleaned_tokens = [token for token in tokens if token not in english_stop_words and token not in russian_stop_words]
        vocab_size = len(set(cleaned_tokens))
        if vocab_size / len(cleaned_tokens) < 0.4:
            return None

        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(cleaned_text.lower())
        cleaned_tokens = [lemmatizer.lemmatize(token) for token in tokens if
                          token not in english_stop_words and token not in russian_stop_words]
        cleaned_text = " ".join(cleaned_tokens)

        # Filter out texts with programming-related content
        if re.search(r'\b(?:if|else|while|for|def|class|import|from|return|try|except|finally|raise|assert|yield|with)\b',
                     cleaned_text, re.IGNORECASE):
            return None

        # Filter out short or incomplete sentences
        sentences = sent_tokenize(cleaned_text)
        cleaned_sentences = []
        for sentence in sentences:
            sentence_len = len(sentence)
            if sentence_len < 5 or sentence[-1] not in ['.', '!', '?']:
                continue
            cleaned_sentences.append(sentence)
        # Combine cleaned sentences into a single string
        cleaned_text = ' '.join(cleaned_sentences)

        main_language = 'en'
        try:
            detected_languages = detect_langs(text)
            main_language = max(detected_languages, key=lambda x: x.prob).lang
            if main_language != "en" and main_language != "ru":
                return None
        except Exception:
            return None

        stop_words = []
        if main_language == 'en':
            stop_words = set(stopwords.words('english'))
        elif main_language == 'ru':
            stop_words = set(stopwords.words('russian'))
        tokens = word_tokenize(cleaned_text)
        filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
        cleaned_text = ' '.join(filtered_tokens)

        # Remove incomplete sentences and sentence fragments
        if len(cleaned_text) < 20 or cleaned_text[-1] not in string.punctuation:
            return None

        # Remove incomplete words at the beginning or end of sentences
        cleaned_text = re.sub(r'\b\w{1,2}\b', '', cleaned_text)

        # Remove words containing non-alphabetic characters
        cleaned_text = re.sub(r'\b\w*\d\w*\b', '', cleaned_text)

        # Remove words containing only one character
        cleaned_text = re.sub(r'\b\w{1}\b', '', cleaned_text)

        # Remove lines with incomplete or missing target text
        if ':' in cleaned_text and len(cleaned_text.split(':')[-1]) < 10:
            return None

        # Remove lines with incomplete or missing prompt text
        if ':' in cleaned_text and len(cleaned_text.split(':')[-2]) < 10:
            return None

        if cleaned_text.count('.') <= 1:
            return None
        cleaned_text = cleaned_text.split('.')[:-1]
        cleaned_text = '.'.join(cleaned_text)

        cleaned_text = ' '.join(sentence for sentence in cleaned_text.split('.') if len(sentence.split()) > 1)

        return cleaned_text

    def process_data(self, data_queue, data_set, max_queue_size, label):
        processed_texts_count = 0
        batch_size = 300  # You can adjust this value as needed

        if self.debug:
            print(f'{label}: Start process data, qsize: {data_queue.qsize()}')

        texts = []
        i = 0
        for d in data_set:
            cleaned_text = self.clean_text(d['text'])
            if cleaned_text is None:
                continue
            texts.append(cleaned_text)
            if i % 50 == 0:
                if self.debug:
                    print(f'{label}: Processed {i} texts, qsize: {data_queue.qsize()}')

            if len(texts) >= batch_size:
                relevant_chunks_ids = self.process_texts(texts)
                if self.debug:
                    print(f'{label}: found {len(relevant_chunks_ids)} relevant chunks')
                for chunk_ids in relevant_chunks_ids:
                    if data_queue.qsize() >= max_queue_size:
                        print(f'{label}: Processed texts: {processed_texts_count}, qsize: {data_queue.qsize()}')
                        return
                    data_queue.put(chunk_ids)
                texts = []
                processed_texts_count += batch_size
            i += 1

    def process_data_wrapper(self, args):
        return self.process_data(*args)

    def run(self):
        self.running = True

        if self.parallel:
            with ThreadPoolExecutor(max_workers=2) as executor:
                train_args = (self.train_queue, self.train_dataset, self.train_queue_max_size, 'train')
                validation_args = (
                    self.validation_queue, self.validation_dataset, self.validation_queue_max_size, 'validation')
                executor.map(self.process_data_wrapper, [train_args, validation_args])
        else:
            self.process_data(self.train_queue, self.train_dataset, self.train_queue_max_size, 'train')
            self.process_data(self.validation_queue, self.validation_dataset, self.validation_queue_max_size, 'validation')

        if self.train_queue.qsize() == self.train_queue_max_size and self.validation_queue.qsize() == self.validation_queue_max_size:
            print("Train and validation queues are full.")

        self.running = False

    def get_random_item(self, queue: Queue):
        queue_list = list(queue.queue)
        random_item = random.choice(queue_list)
        queue.queue.remove(random_item)
        return random_item

    def get_train_item(self, timeout=None):
        return self.get_item(self.train_queue, timeout=timeout)

    def get_validation_item(self, timeout=None):
        return self.get_item(self.validation_queue, timeout=timeout)

    def get_item(self, queue, timeout=None, random_item=False):
        if queue.qsize() == 0:
            print('Will reload datasets, reason: queue is empty')
            self.init_default_datasets()
        if random_item:
            item = self.get_random_item(queue)
        else:
            item = queue.get(timeout=timeout)

        return item

    def load_items_from_cache(self, queue, cache_file_path):
        cached_data = load_from_file(cache_file_path)
        if cached_data:
            random.shuffle(cached_data)
            for chunk_ids in cached_data:
                queue.put(chunk_ids)

    def prefetch_train(self, dataset_name):
        cache_file_path = self.train_cache_files[dataset_name]
        dataset = self.train_datasets[dataset_name]
        self.prefetch_items(f'train_{dataset_name}', dataset, self.prefetch_items_count, self.train_queue, cache_file_path)

    def prefetch_validation(self, dataset_name):
        cache_file_path = self.validation_cache_files[dataset_name]
        dataset = self.validation_datasets[dataset_name]
        self.prefetch_items(f'validation_{dataset_name}', dataset, self.prefetch_items_count, self.validation_queue,
                            cache_file_path)

    def prefetch_items(self, label, dataset, num_items, queue, cache_file_path):
        print(f'Will prefetch {num_items} items for {label}, dataset: {dataset}, queue: {queue}')
        i = 0
        with tqdm(total=num_items, desc=f"Prefetching {label} items") as pbar:
            for d in dataset:
                if self.debug:
                    print(f'Processing next item for {label}')
                if queue.qsize() >= num_items:
                    print(
                        f'Stopping prefetching items for {label}, reason: queue.qsize() ({queue.qsize()}) >= num_items {num_items}')
                    break
                try:
                    text = d['text']
                except StopIteration:
                    print(f'StopIteration encountered for {label}')
                    break
                relevant_chunks_ids = self.process_texts([text])
                if not relevant_chunks_ids:
                    if self.debug:
                        print(f'No relevant chunks found for {label}')
                for chunk_ids in relevant_chunks_ids:
                    queue.put(chunk_ids)
                    if queue.qsize() >= num_items:
                        print(
                            f'Queue size reached the limit for {label}: queue.qsize() ({queue.qsize()}) >= num_items {num_items}')
                        break
                    i += 1
                    if i % 100 == 0:
                        if self.debug:
                            print(f'Saving intermediate results for {label} to cache file')
                        self.save_to_file(cache_file_path, list(queue.queue))
                # Update progress bar to the current queue.qsize()
                pbar.update(queue.qsize() - pbar.n)
            print(f'Finished processing all items for {label}')
            self.save_to_file(cache_file_path, list(queue.queue))

    def save_to_file(self, file_path, data):
        with open(file_path, 'wb') as f:
            msgpack.dump(data, f)

    def remove_items_from_cache(self, file_path, items_to_remove):
        cache = load_from_file(file_path)
        if cache:
            cache_set = set(tuple(item) for item in cache)
            items_to_remove_set = set(tuple(item) for item in items_to_remove)
            cache_set = cache_set - items_to_remove_set
            cache = [list(item) for item in cache_set]
            self.save_to_file(file_path, cache)
