import time
import openai

from flask import Flask, request
from embeddings import SentenceEmbeddings
from storage import FaissStorage

INDEX_FILE = ".birds.faiss"
ZH_INDEX_FILE = ".birds_zh.faiss"
MAX_QUERIES = 5

app = Flask(__name__)
fs = FaissStorage()
fs.load(INDEX_FILE)
zh_fs = FaissStorage()
zh_fs.load(ZH_INDEX_FILE)
se = SentenceEmbeddings()

def translate(prompt):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0301",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=200,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return resp["choices"][0]["message"]["content"]

def chat_translate(eng_text, refs):
    prompt = f"Learning these knowledge about bird name translation:\n\n{refs}\n\nNow translate below english text in the triple backticks shortly to Chinese sentence without double quotes: \n\n```{eng_text}```\n\nAnswer:"
    resp = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0301",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=200,
        top_p=1.0,
    )
    return resp["choices"][0]["message"]["content"]

def openai_ask(refs, query):
    prompt = f"Read and understand below text in the triple backticks and answer the question by following rules: 1. Answer the question politely and completely. 2. If the text does not provide information for the question, just answer the question by using knowledge from Wikipedia or eBird.org. 3. Don't say \"not mentioned\" in the answer. \n\n```{refs}```\n\nQuestion: {query}\nAnswer: "
    resp = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo-0301",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=200,
        top_p=1.0,
    )
    return resp["choices"][0]["message"]["content"]

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
        query = translate(f"Translate this into English:\n\n{init_query}\n\n")
        print("Q:", query.strip())
        em = se.generate_embedding(query)
        refs = "\n\n".join(fs.get(em, MAX_QUERIES) + zh_fs.get(em, MAX_QUERIES))
        print("Related: ", refs)
        res = openai_ask(refs, query)
        print("A:", res)
        refs = "\n\n".join(zh_fs.get(em, MAX_QUERIES))
        zh_res = chat_translate(res, refs)
        print("ZH_A:", zh_res)
        print("time:", time.time() - begin)
    else:
        res = "Hello world"
    return f"{res}\n{zh_res}"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
