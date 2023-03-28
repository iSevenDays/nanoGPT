import random
import threading
import time
from queue import Queue

colab = False
import concurrent
import chardet

cache_dir = '/Volumes/WD_RED_RW/machine_learning/cache'
#cache_dir = '/Volumes/homes/seven/Btrfs_main_backup/Macbook14M1Leipzig.local/Volumes/WD_RED_RW/machine_learning/cache'

import datasets
import os
from datasets.builder import DatasetGenerationError
from datasets import Features
from datasets import load_dataset, interleave_datasets, IterableDataset
from concurrent.futures import ThreadPoolExecutor


# Load the wiki dataset
wiki = load_dataset("wikitext", 'wikitext-103-v1', split="train", streaming=False)

# Define a generator function that yields examples from the wiki dataset
def wiki_gen():
    for example in wiki:
        yield example

# Create the IterableDataset using the generator function
iterable_wiki = IterableDataset.from_generator(
    generator=wiki_gen,
    features=Features(wiki.features),
)

global train_dataset, validation_dataset, datasets_ru, pile, multi_dataset
try:
    if pile:
        print('pile loaded')
except:
    pile = load_dataset("EleutherAI/the_pile_deduplicated", split="train", cache_dir=cache_dir)

def pile_gen():
    for example in pile:
        yield example

iterable_pile = IterableDataset.from_generator(
    generator=pile_gen,
    features=Features(pile.features),
)

all_datasets = []
try:
    if datasets_ru:
        print('datasets_ru loaded')
except:
    datasets_ru = []

DATA_LOAD_FORCE = False
try:
    if DATA_IS_LOADED:
        print("Data is already loaded")
    if DATA_LOAD_FORCE == True:
      raise Exception("Data load requested")
except:
    global all_datasets, datasets_ru
    train_dataset = None
    validation_dataset = None
    # Define the path to the folders containing the text files
    base = '/Users/seven/Downloads/corpus/'
    parent_folders = [
        "Arzamas",
        "Interfax",
        "Lenta",
        "NPlus1",
    ]
    if not colab:
      for year in range(2005, 2006):# 2018):
          parent_folders.append(f"proza_ru_{year}" )
    parent_folders = [] # TODO: ENABLE LATER
    # Define a function to recursively search for text files
    def find_text_files(folder_path):
        text_files = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.startswith('.'):
                    continue
                if file_name.endswith('.txt'):
                    file_path = os.path.join(root, file_name)
                    text_files.append(file_path)
            for sub_dir in dirs:
                sub_dir_path = os.path.join(root, sub_dir)
                text_files += find_text_files(sub_dir_path)
        return text_files

    def read_file_with_chardet(file_path):
        with open(file_path, "rb") as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(file_path, "r", encoding=encoding) as file:
            text = file.read()

        return {"text": text}

    def read_files_from_folder(folder_path, max_workers=8):
        # Find all the text files in the given folder and its subdirectories
        text_files = find_text_files(folder_path)

        # Create a ThreadPoolExecutor to process the files in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Use the executor to submit tasks for each file
            tasks = [executor.submit(read_file_with_chardet, file_path) for file_path in text_files]

            # Iterate through the tasks and yield their results
            for task in concurrent.futures.as_completed(tasks):
                yield task.result()

    # Create the dataset using the read_files_from_folder function
    if len(datasets_ru) == 0 and not colab:
        datasets_dict = datasets.DatasetDict()
        for parent_folder in parent_folders:
            try:
                print(f'Processing {parent_folder}')
                folder_path = os.path.join(base, parent_folder)
                dataset = datasets.Dataset.from_generator(
                    generator=lambda: read_files_from_folder(folder_path),
                    cache_dir=cache_dir
                    #features=Features.from_dict({"text": datasets.Value("string")}),
                )
                datasets_dict[parent_folder] = dataset
            except DatasetGenerationError:
                print(f'Will skip {parent_folder}')
                continue

            for k, v in datasets_dict.items():
              v = v.to_iterable_dataset(num_shards=128)
              datasets_ru.append((parent_folder, v))
    try:
        print(datasets_ru[0])
    except:
        pass

    all_datasets = datasets_ru if not colab else []
    all_datasets.append(('pile', pile))
    all_datasets.append(('wiki', wiki))
    #
    # multi_dataset = interleave_datasets(all_datasets)
    # shuffled_dataset = multi_dataset.shuffle(buffer_size=50_000)

    DATA_IS_LOADED = True

print(train_dataset)
datasets_dict = {}
for dataset_name, dataset in all_datasets:
    datasets_dict[dataset_name] = dataset
datasets_dict
from IrrelevantChecker import IrrelevantChecker
from is_irrelevant import is_irrelevant
import data_worker_module
import importlib
importlib.reload(data_worker_module)
from data_worker_module import DataWorker

relevance_checker = IrrelevantChecker(is_irrelevant)

data_worker = DataWorker(relevance_checker=relevance_checker, datasets=datasets_dict, load_cached_data=True)
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

def worker_train(dataset_name):
    data_worker.prefetch_train(dataset_name)

def worker_validation(dataset_name):
    data_worker.prefetch_validation(dataset_name)

with ThreadPoolExecutor(max_workers=8) as executor:
    # Start the prefetching threads

    futures_train = []
    futures_validation = []

    for dataset_name, dataset in all_datasets:
        executor.submit(worker_train, dataset_name)

    for dataset_name, dataset in all_datasets:
        executor.submit(worker_validation, dataset_name)

    # You can execute other code here without waiting for the processes to finish
    print("Processes are running asynchronously")


import random
# The context manager will wait for all processes to finish before exiting
print(f"All processes have finished {random.randint}")