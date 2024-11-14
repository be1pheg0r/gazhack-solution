from typing import Tuple, List

import numpy
import pandas as pd

from bertopic import BERTopic
from numpy import ndarray
from sentence_transformers import SentenceTransformer

import os
import datetime

import json

BERTOPIC_CONFIG = {
    "embedding_model": SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2"),
    "n_gram_range": (1, 3),
    "min_topic_size": 5,
    "top_n_words": 5
}


class Tengine:

    def __init__(self, docs: list[str], docs_for_fit: list[str], seed_topic_list: list[list[str]] = None) -> None:
        if seed_topic_list:
            self.model = BERTopic(**BERTOPIC_CONFIG, seed_topic_list=seed_topic_list)
        self.model = BERTopic(**BERTOPIC_CONFIG)
        self._docs = docs
        self._edited = docs_for_fit
        self._seed_topic_list = seed_topic_list
        self._topics = None
        self._probs = None

    def __str__(self):
        config_table = [[key, value] for key, value in BERTOPIC_CONFIG.items()]
        res = (
            f"Topic Engine object\n"
            f"Model config:\n{config_table}\n"
        )
        return res

    def load_model(self, path: str) -> None:
        try:
            self.model.load(path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Model not found at {path}")

    def save_model(self, dir: str) -> None:
        if os.path.isdir(dir):
            base_name = "model"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{base_name}_{timestamp}.pkl"
            file_path = os.path.join(dir, file_name)

            counter = 1
            while os.path.exists(file_path):
                file_name = f"{base_name}_{timestamp}_{counter}.pkl"
                file_path = os.path.join(dir, file_name)
                counter += 1
        else:
            file_path = dir

        self.model.save(file_path, save_embedding_model=True)

    def fit(self) -> None:
        self.model.fit(self._edited)
        return None

    def fit_transform(self) -> tuple[list[int], ndarray | ndarray | None]:
        self._topics, self._probs = self.model.fit_transform(self._edited)
        return self._topics, self._probs

    def get_topics(self) -> list[int] | None:
        topics = self._topics
        return topics

    def reduce_outliers(self) -> None:
        if self._probs is not None:
            self._topics = self.model.reduce_outliers(self._edited, topics=self._topics, probabilities=self._probs)
        else:
            self._topics = self.model.reduce_outliers(self._edited, topics=self._topics)
        return None

    def topics_to_df(self) -> pd.DataFrame:
        topics = self.get_topics()
        df = pd.DataFrame({
            'Document': self._docs,
            'Topic': topics
        })
        return df

    def topics_names_to_df(self) -> pd.DataFrame:
        topics = self.get_topics()
        df = pd.DataFrame({
            'Document': self._docs,
            'Topic': [self.model.get_topic_info().loc[self.model.get_topic_info()['Topic'] == t, 'Representation'].values[0] for t
                      in topics]
        })
        return df

    def get_docs_with_topics(self) -> pd.DataFrame:
        topics = []
        for doc in self._docs:
            temp = []
            tops, probs = self.model.find_topics(doc, top_n=5)
            for t, p in zip(tops, probs):
                if p >= 0.7:
                    temp.append(t)

            if len(temp) == 0:
                temp.append(-1)
            topics.append(temp)

        df = pd.DataFrame({
            'Document': self._docs,
            'Topics': topics
        })

        return df

    def save_topics(self, path: str) -> None:
        topics = self.model.get_topic_info()["Representation"].tolist()

        if os.path.isdir(path):
            base_name = "topics"
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{base_name}_{timestamp}.json"
            file_path = os.path.join(path, file_name)

            counter = 1
            while os.path.exists(file_path):
                file_name = f"{base_name}_{timestamp}_{counter}.json"
                file_path = os.path.join(path, file_name)
                counter += 1
        else:
            file_path = path

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(topics, f, ensure_ascii=False, indent=4)

    def _load_topics(self, path: str) -> list[list[str]]:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                topics = json.load(f)
                return topics
        except FileNotFoundError:
            raise FileNotFoundError(f"Topics file not found at {path}")
