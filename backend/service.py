import time
import openai

from flask import Flask, request
from embeddings import SentenceEmbeddings
from storage import FaissStorage

INDEX_FILE = ".birds.faiss"
MAX_QUERIES = 5

app = Flask(__name__)
fs = FaissStorage()
fs.load(INDEX_FILE)
se = SentenceEmbeddings()

@app.route("/", methods=["POST", "GET"])
def whisper():
    print(request.method)
    if request.method == "POST":
        file = request.files["audio_data"]
        filename = f"/tmp/{file.filename}.wav"
        file.save(filename)
        begin = time.time()
        audio = open(filename, "rb")
        query = openai.Audio.transcribe("whisper-1", audio)["text"]
        print("Q:", query)
        em = se.generate_embedding(query)
        ans = "".join(fs.get(em, MAX_QUERIES))
        prompt = f"You are a friendly assistant. Please read below text and answer the questions clearly, directly and shortly:\n\n{ans}\n\n"
        resp = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ]
        )
        res = resp["choices"][0]["message"]["content"]
        print("A:", res)
        print("time:", time.time() - begin)
    else:
        res = "Hello world"
    return res

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
