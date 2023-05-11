import time
import tqdm
import os
import sys
import codecs

from embeddings import SentenceEmbeddings
from storage import FaissStorage
from splitter import BirdSplitter
from llm import StablelmLLM, DollyLLM, T5LLM

INDEX_FILE = ".birds.faiss"
ZH_INDEX_FILE = ".birds_zh.faiss"

def common_build(sentences, filename):
    se = SentenceEmbeddings()
    begin = time.time()
    em = se.generate_embedding(sentences[0])

    fs = FaissStorage(em.shape[0])
    for sent in tqdm.tqdm(sentences):
        em = se.generate_embedding(sent)
        fs.put(se.generate_embedding(sent), sent)
    fs.build_index()
    fs.save(filename)

def build_chinese():
    # Get chinese info
    print("Chinese info:")
    sentences = []
    data_dir = "/home/robin/Documents/json2/"
    for filename in tqdm.tqdm(os.listdir(data_dir)):
        with codecs.open(data_dir + filename, "r", encoding="utf-8", errors="ignore") as fp:
            for line in fp.readlines():
                line = line.strip()
                if line.find("英文名") >= 0 and len(line) > 30:
                    sentences.append(line)
    common_build(sentences, ZH_INDEX_FILE)

def build_english():
    print("English info:")
    sentences = []
    data_dir = "/home/robin/Documents/jsonEN/"
    for filename in tqdm.tqdm(os.listdir(data_dir)):
        with open(data_dir + filename, "r") as fp:
            content = fp.read()

        splitter = BirdSplitter(1024)
        res = splitter.split(content)
        if len(res) <= 0:
            continue
        if min([len(item) for item in res]) < 12:
            print(f"filename: {filename}")
            print(res)
        sentences += res
    common_build(sentences, INDEX_FILE)

if __name__ == "__main__":
    build_english()
    build_chinese()
