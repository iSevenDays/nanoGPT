from abc import ABC, abstractmethod

import nltk


class TextRelevanceChecker(ABC):
    @abstractmethod
    def is_relevant(self, text: str) -> bool:
        pass


class IrrelevantChecker(TextRelevanceChecker):
    def __init__(self, is_irrelevant_function):
        self.is_irrelevant_function = is_irrelevant_function
        self.init_datasets_for_relevance_checker()

    def init_datasets_for_relevance_checker(self):
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('wordnet')

    def is_relevant(self, text: str) -> bool:
        return not self.is_irrelevant_function(text)
