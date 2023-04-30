import faiss
import numpy as np
import pickle

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

    @abstractmethod
    def save(self, filename: str):
        pass

    @abstractmethod
    def load(self, filename: str):
        pass

class FaissStorage(Storage):
    def __init__(self, dimention: int = 1024):
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

    def save(self, filename: str):
        array = faiss.serialize_index(self._index)
        with open(filename, "wb") as fp:
            obj = {"seq_id": self._seq_id, "map": self._map, "index": array}
            fp.write(pickle.dumps(obj))

    def load(self, filename: str):
        with open(filename, "rb") as fp:
            obj = pickle.loads(fp.read())
        self._seq_id = obj["seq_id"]
        self._map = obj["map"]
        self._index = faiss.deserialize_index(obj["index"])
