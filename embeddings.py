from abc import ABC, abstractmethod
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

class Embeddings(ABC):
    @abstractmethod
    def generate_embedding(self, sentence: str):
        pass

class SentenceEmbeddings(Embeddings):
    def __init__(self):
        self._model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        #self._model = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-tuned-alpha-3b").half().cuda()

    def generate_embedding(self, sentence: str):
        return self._model.encode(sentence)
