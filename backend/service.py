import time
import openai

from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def whisper():
    print(request.method)
    if request.method == "POST":
        file = request.files["audio_data"]
        filename = f"/tmp/{file.filename}.wav"
        file.save(filename)
        begin = time.time()
        audio = open(filename, "rb")
        result = openai.Audio.transcribe("whisper-1", audio)
        print("time:", time.time() - begin)
    else:
        print(request)
        result = {"text": "Hello world!"}
    print(result["text"])
    return result["text"]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
