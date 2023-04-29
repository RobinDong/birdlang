import time
import tqdm
import sys

from embeddings import SentenceEmbeddings
from storage import FaissStorage
from splitter import BirdSplitter
from llm import StablelmLLM, DollyLLM, T5LLM

with open("/home/robin/Documents/jsonEN/1101.json", "r") as fp:
    content = fp.read()

splitter = BirdSplitter(400)
sentences = splitter.split(content)
print(max([len(item) for item in sentences]))
print(min([len(item) for item in sentences]))

se = SentenceEmbeddings()
begin = time.time()
em = se.generate_embedding(sentences[0])
print(em.shape, time.time() - begin)

fs = FaissStorage(em.shape[0])
for sent in tqdm.tqdm(sentences):
    em = se.generate_embedding(sent)
    fs.put(se.generate_embedding(sent), sent)
fs.build_index()
query = "Is there any hawk in Asia?"
ans = fs.get(se.generate_embedding(query), 4)
print(ans)
#sl = StablelmLLM("StabilityAI/stablelm-tuned-alpha-3b")
#sl = DollyLLM("databricks/dolly-v2-3b")
sl = T5LLM("google/flan-t5-large")
resp = sl.generate("".join(ans), query)
print(resp)
