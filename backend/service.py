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
        init_query = openai.Audio.transcribe("whisper-1", audio)["text"]
        print("I:", init_query)
        resp = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Translate this into English:\n\n{init_query}\n\n",
            temperature=0.3,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        query = resp["choices"][0]["text"]
        print("Q:", query)
        em = se.generate_embedding(query)
        ans = "".join(fs.get(em, MAX_QUERIES))
        print("Related: ", ans)
        prompt = f"You are a friendly assistant. Please read below text and answer the questions shortly by using Chinese language:\n\n{ans}\n\n"
        resp = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0301",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Question: " + query},
            ],
            temperature=0.3
        )
        res = resp["choices"][0]["message"]["content"]
        resp = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Translate this into Chinese:\n\n{res}\n\n",
            temperature=0.3,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        res = resp["choices"][0]["text"]
        print("A:", res)
        print("time:", time.time() - begin)
    else:
        res = "Hello world"
    return f"Q: {init_query}\nA: {res}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
