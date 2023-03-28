import os
import pickle
import random
from queue import Queue
from typing import List

import msgpack


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
    def __init__(self, datasets: List[str], dataset_weights: dict[str, float],
                 parallel: bool = True,
                 cache_dir='cache'):
        self.validation_dataset = None
        self.train_dataset = None
        self.parallel = parallel
        # Create separate train and validation queues for each dataset
        self.train_queues = {dataset_name: Queue() for dataset_name in datasets}
        self.validation_queues = {dataset_name: Queue() for dataset_name in datasets}

        self.debug = False
        self.parallel = parallel
        self.cache_dir = cache_dir
        self.datasets = datasets
        self.dataset_weights = dataset_weights

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

        self.load_train_dataset_from_cache(dataset_name)
        self.load_validation_dataset_from_cache(dataset_name)

    def load_train_dataset_from_cache(self, dataset_name):
        train_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_train_cache.pkl")

        self.load_items_from_cache(self.train_queues[dataset_name], train_cache_file)
        print(f"{dataset_name} Train queue length: {self.train_queues[dataset_name].qsize()}")

    def load_validation_dataset_from_cache(self, dataset_name):
        validation_cache_file = os.path.join(self.cache_dir, f"{dataset_name}_validation_cache.pkl")
        self.load_items_from_cache(self.validation_queues[dataset_name], validation_cache_file)
        print(f"{dataset_name} Validation queue length: {self.validation_queues[dataset_name].qsize()}")

    def get_random_item(self, queue: Queue):
        queue_list = list(queue.queue)
        random_item = random.choice(queue_list)
        queue.queue.remove(random_item)
        return random_item

    def get_train_item(self, timeout=None):
        dataset_name = random.choices(list(self.train_queues.keys()), list(self.dataset_weights.values()))[0]
        return self.get_item(self.train_queues[dataset_name], timeout=timeout)

    def get_validation_item(self, timeout=None):
        dataset_name = random.choices(list(self.validation_queues.keys()), list(self.dataset_weights.values()))[0]
        return self.get_item(self.validation_queues[dataset_name], timeout=timeout)

    def get_item(self, queue, timeout=None, random_item=False):
        if queue.qsize() == 0:
            print('Will reload datasets, reason: queue is empty')
            for dataset_name, q in self.train_queues.items():
                if q == queue:
                    print(f'Will reload train queue for dataset {dataset_name}')
                    self.load_train_dataset_from_cache(dataset_name)
            for dataset_name, q in self.validation_queues.items():
                if q == queue:
                    print(f'Will reload validation queue for dataset {dataset_name}')
                    self.load_validation_dataset_from_cache(dataset_name)

            self.init_default_datasets()
        if random_item:
            item = self.get_random_item(queue)
        else:
            item = queue.get(timeout=timeout)

        return item

    def get_train_queue_size(self):
        num = 0
        for dataset_name in self.datasets:
            num += self.train_queues[dataset_name].qsize()
        return num

    def get_validation_queue_size(self):
        num = 0
        for dataset_name in self.datasets:
            num += self.validation_queues[dataset_name].qsize()
        return num

    def load_items_from_cache(self, queue, cache_file_path):
        cached_data = load_from_file(cache_file_path)
        if cached_data:
            random.shuffle(cached_data)
            for chunk_ids in cached_data:
                queue.put(chunk_ids)
