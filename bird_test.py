import time
import tqdm
import os
import sys

from embeddings import SentenceEmbeddings
from storage import FaissStorage
from splitter import BirdSplitter
from llm import StablelmLLM, DollyLLM, T5LLM

INDEX_FILE = ".birds.faiss"

if os.path.exists(INDEX_FILE):
    fs = FaissStorage()
    fs.load(INDEX_FILE)
    se = SentenceEmbeddings()
else:
    sentences = []
    data_dir = "/home/robin/Documents/jsonEN/"
    for filename in tqdm.tqdm(os.listdir(data_dir)):
        with open(data_dir + filename, "r") as fp:
            content = fp.read()

        splitter = BirdSplitter(400)
        res = splitter.split(content)
        if len(res) <= 0:
            continue
        if min([len(item) for item in res]) < 12:
            print(f"filename: {filename}")
            print(res)
        sentences += res

    se = SentenceEmbeddings()
    begin = time.time()
    em = se.generate_embedding(sentences[0])

    fs = FaissStorage(em.shape[0])
    for sent in tqdm.tqdm(sentences):
        em = se.generate_embedding(sent)
        fs.put(se.generate_embedding(sent), sent)
    fs.build_index()
    fs.save(INDEX_FILE)

for query in [
        "Is there any hawk living in Asia?",
        "What's the larget type of bird on the earth?",
        ]:
    ans = fs.get(se.generate_embedding(query), 5)
    print(ans)
    #sl = StablelmLLM("StabilityAI/stablelm-tuned-alpha-3b")
    #sl = DollyLLM("databricks/dolly-v2-3b")
    sl = T5LLM("lmsys/fastchat-t5-3b-v1.0")
    resp = sl.generate("".join(ans), query)
    print(f"Question: {query}")
    print(f"Answer: {resp}")
