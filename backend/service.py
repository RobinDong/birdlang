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
    begin = time.time()
    if request.method == "POST":
        if len(request.files) > 0:
            file = request.files["audio_data"]
            filename = f"/tmp/{file.filename}.wav"
            file.save(filename)
            audio = open(filename, "rb")
            init_query = openai.Audio.transcribe("whisper-1", audio)["text"]
        else:
            print(request.form)
            init_query = request.form["question"]
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
        print("Q:", query.strip())
        em = se.generate_embedding(query)
        ans = "\n\n".join(fs.get(em, MAX_QUERIES))
        print("Related: ", ans)
        prompt = f"Read and understand below text in the triple backticks and answer the question by following rules: \n1. Answer the question shortly. \n2. Answer the question by English language at first, and then answer the question by Chinese language. \n3. If the text does not provide information for the question, just answer the question by using knowledge from web (include Wikipedia and eBird.org).' \n\n```{ans}```\n\nQuestion: {query}"
        resp = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-0301",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=200,
            top_p=1.0,
        )
        res = resp["choices"][0]["message"]["content"]
        print("A:", res)
        print("time:", time.time() - begin)
    else:
        res = "Hello world"
    return f"{res}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
