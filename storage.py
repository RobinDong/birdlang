import faiss
import numpy as np

from abc import ABC, abstractmethod

class Storage(ABC):
    @abstractmethod
    def put(self, embedding, sentence: str):
        pass

    @abstractmethod
    def build_index(self):
        pass

    @abstractmethod
    def get(self, embedding, k: int):
        pass

class FaissStorage(Storage):
    def __init__(self, dimention: int):
        self._index = faiss.IndexFlatL2(dimention)
        self._matrix = []
        self._map = {}
        self._seq_id = 0

    def put(self, embedding, sentence: str):
        self._map[self._seq_id] = sentence
        self._seq_id += 1
        self._matrix.append(embedding)

    def build_index(self):
        self._index.add(np.array(self._matrix))

    def get(self, embedding, k: int):
        scores, indices = self._index.search(np.array([embedding]), k)
        ans = []
        for idx in sorted(indices[0]):
            if idx == -1:
                continue
            ans.append(self._map[idx])
        return ans
