import concurrent
import itertools
import multiprocessing
import os
import pickle
import random
import re
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager

import msgpack
from datasets import Dataset
from transformers import GPT2Tokenizer

from IrrelevantChecker import TextRelevanceChecker
from text_cleaner import TextCleaner

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
BUFFER_SIZE = 10000


def load_from_file(file_path):
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return msgpack.load(f, use_list=True)
    except msgpack.exceptions.ExtraData:  # If MessagePack fails, try loading with pickle
        with open(file_path, 'rb') as f:
            return pickle.load(f)


class DataWorker:
    def __init__(self, relevance_checker: TextRelevanceChecker,
                 datasets: dict[str, Dataset],
                 block_size: int = 1024, buffer_size: int = 100_000, parallel: bool = True,
                 prefetch_items_count=1000000,
                 cache_dir='cache',
                 load_cached_data=True):
        self.validation_dataset = None
        self.train_dataset = None
        self.relevance_checker = relevance_checker
        self.buffer_size = buffer_size
        self.parallel = parallel
        self.train_lists = {}
        self.validation_lists = {}
        self.train_lists_locks = {}
        self.validation_lists_locks = {}
        self.output_queue = multiprocessing.Manager().Queue()
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
        self.text_cleaner = TextCleaner()
        self.stop_event = Manager().Event()

        # self.seen_chunk_ids = {'train': {}, 'validation': {}}
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        # Initialize datasets and create cache files for each dataset
        self.init_default_datasets()

    def init_default_datasets(self):
        for dataset_name, dataset in self.datasets.items():
            self.init_datasets(dataset_name, dataset)

    def init_datasets(self, dataset_name, dataset):
        split = dataset.train_test_split(test_size=0.2, seed=42)

        train_dataset = split['train']
        validation_dataset = split['test']

        # self.seen_chunk_ids['train'][dataset_name] = set()
        # self.seen_chunk_ids['validation'][dataset_name] = set()

        self.train_lists[dataset_name] = Manager().list()
        self.validation_lists[dataset_name] = Manager().list()
        self.train_lists_locks[dataset_name] = Manager().Lock()
        self.validation_lists_locks[dataset_name] = Manager().Lock()

        self.train_datasets[dataset_name] = train_dataset
        self.validation_datasets[dataset_name] = validation_dataset

        # Create cache files for each dataset
        cache_dir = 'cache'
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        train_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_train_cache.pkl")
        validation_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_validation_cache.pkl")

        if self.load_cached_data:
            self.load_items_from_cache(self.train_lists[dataset_name], train_cache_file, 'train', dataset_name)
            self.load_items_from_cache(self.validation_lists[dataset_name], validation_cache_file, 'validation', dataset_name)
            self.print(f"{dataset_name} Train queue length after loading cached data: {len(self.train_lists[dataset_name])}")
            self.print(f"{dataset_name} Validation queue length after loading cached data: {len(self.validation_lists[dataset_name])}")

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

    # def process_texts(self, texts):
    #     with ProcessPoolExecutor(max_workers=4) as executor:
    #         relevant_chunks_ids = list(filter(None, executor.map(self.is_relevant_chunk, texts)))
    #
    #     return relevant_chunks_ids
    def process_texts(self, texts):
        relevant_chunks_ids = []

        for text in texts:
            chunk_ids = self.is_relevant_chunk(text)
            if chunk_ids is not None:
                relevant_chunks_ids.append(chunk_ids)

        return relevant_chunks_ids

    def has_cyrillic(self, text):
        return bool(re.search('[\u0400-\u04FF]', text))

    def load_items_from_cache(self, data_list, cache_file_path, label, dataset_name):
        self.print(f'Loading file from cache for {label} for dataset_name {dataset_name}')
        cached_data = load_from_file(cache_file_path)
        if cached_data and len(cached_data) > 0:
            self.print(f'Loading {len(cached_data)} items from cache for {label} for dataset_name {dataset_name}')
        else:
            self.print(f'Loaded no items from cache for {label} for dataset_name {dataset_name}')
        # seen_chunk_ids = self.seen_chunk_ids[label][dataset_name]  # Get the seen_chunk_ids set for the dataset
        if cached_data and len(cached_data) > 0:
            random.shuffle(cached_data)
            with Manager().Lock():
                data_list.extend(cached_data)

    def prefetch_train(self, dataset_name):
        cache_file_path = self.train_cache_files[dataset_name]
        dataset = self.train_datasets[dataset_name]

        self.prefetch_items(label=f'train_{dataset_name}',
                            dataset=dataset,
                            dataset_name=dataset_name,
                            num_items=self.prefetch_items_count,
                            data_list=self.train_lists[dataset_name],
                            cache_file_path=cache_file_path)

    def prefetch_validation(self, dataset_name):
        cache_file_path = self.validation_cache_files[dataset_name]
        dataset = self.validation_datasets[dataset_name]
        self.prefetch_items(label=f'validation_{dataset_name}',
                            dataset=dataset,
                            dataset_name=dataset_name,
                            num_items=self.prefetch_items_count,
                            data_list=self.validation_lists[dataset_name],
                            cache_file_path=cache_file_path)

    def process_dataset_indices(self, args):
        dataset, start_idx, end_idx, buffer_size = args
        texts = [dataset[i]['text'] for i in range(start_idx, end_idx)]
        text_buffer = []
        relevant_chunks_ids_list = []

        for text in texts:
            if self.stop_event.is_set():
                break
            text_buffer.append(text)
            if len(text_buffer) >= buffer_size:
                relevant_chunks_ids_list.extend(self.process_texts(text_buffer))
                text_buffer = []

        if text_buffer:
            relevant_chunks_ids_list.extend(self.process_texts(text_buffer))

        return relevant_chunks_ids_list

    def prefetch_items(self, label, dataset, dataset_name, num_items, data_list, cache_file_path):
        self.print(f'Will prefetch {num_items} items for {label}, dataset: {dataset}, list: {data_list}')

        num_workers = 4
        buffer_size = 128
        total_dataset_size = len(dataset)
        subset_size = 10000  # Adjust this to the desired subset size
        num_subsets = total_dataset_size // subset_size

        for subset_idx in range(num_subsets + 1):
            lock_file = f'cache/{dataset_name}_{label}_{subset_idx}.lock'
            if os.path.exists(lock_file):
                self.print(f'Skipping prefetch for {label} subset {subset_idx}, dataset: {dataset_name} - lock file exists')
                continue
            start_subset = subset_idx * subset_size
            end_subset = min((subset_idx + 1) * subset_size, total_dataset_size)
            subset = dataset.select(range(start_subset, end_subset))

            chunk_size = max(len(subset) // num_workers, 1)  # Calculate chunk size for each worker
            index_ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_workers)]
            index_ranges[-1] = (index_ranges[-1][0], len(subset))  # Make sure the last range includes the last element

            # Process dataset in parallel
            args_list = [(subset, start_idx, end_idx, buffer_size) for start_idx, end_idx in index_ranges]
            self.print(f'Processing subset {subset_idx + 1} of {num_subsets + 1}')
            self.print(f'Args list is prepared, len: {len(args_list)}')
            lock = self.train_lists_locks[dataset_name] if label == 'train' else self.validation_lists_locks[dataset_name]
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit tasks to the executor
                futures = {executor.submit(self.process_dataset_indices, args): i for i, args in enumerate(args_list)}

                for future in concurrent.futures.as_completed(futures):
                    i = futures[future]  # Get the index of the completed future
                    self.print(f'Processing chunk {i + 1} of {len(args_list)}')
                    results = future.result()  # Get the result of the completed future

                    # Merge and shuffle results
                    relevant_chunks_ids_list = results
                    random.shuffle(relevant_chunks_ids_list)

                    # Add results to the queue
                    self.print(f'Will add {len(relevant_chunks_ids_list)} new items to {label} list of {dataset_name}')

                    if len(relevant_chunks_ids_list) == 0:
                        continue
                    with lock:
                        data_list.extend(relevant_chunks_ids_list)

            # Save current queue to cache file
            with lock:
                self.print(f'Will save list len({len(data_list)}) to {label} list of {dataset_name} to path {cache_file_path} ')
            self.save_list_to_file(lst=data_list, file_path=cache_file_path, dataset_name=dataset_name, label=label)

            # Create the lock file after processing is complete
            with open(lock_file, 'w') as lockfile:
                lockfile.write(f'Processed {lockfile}')

    def save_to_file(self, file_path, data):
        with open(file_path, 'wb') as f:
            msgpack.dump(data, f)

    def stop(self):
        self.stop_event.set()

    def print(self, msg):
        print(msg)
        self.output_queue.put(msg)

    def save_list_to_file(self, lst, file_path, label, dataset_name):
        lock = self.train_lists_locks[dataset_name] if label == 'train' else self.validation_lists_locks[dataset_name]
        with lock:  # Use a Lock from the Manager to synchronize access to the list
            list_copy = list(lst)

            # Save the list to the file
            with open(file_path, 'wb') as f:
                msgpack.dump(list_copy, f)


