import time
import tqdm

from embeddings import SentenceEmbeddings
from storage import FaissStorage
from splitter import GutenbergSplitter

with open("/home/robin/Documents/pompeii.txt", "r") as fp:
    content = fp.read()

splitter = GutenbergSplitter(256)
sentences = splitter.split(content)
print(sentences)
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
query = "Who is Ariston?"
ans = fs.get(se.generate_embedding(query), 5)
print(ans)
