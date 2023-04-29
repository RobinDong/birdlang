import json

from abc import ABC, abstractmethod

class Splitter(ABC):
    @abstractmethod
    def split(self, text: str):
        pass

class GutenbergSplitter(Splitter):
    def __init__(self, max_chunk: int = 1024):
        self._max_chunk = max_chunk

    def split(self, text: str):
        paras = text.split("\n\n")
        chunks = []
        for para in paras:
            sentence = []
            for char in para:
                if char == "\n" or char == "\r":
                    sentence.append(" ")
                    continue
                if char == " " and sentence and sentence[-1] == "." and len(sentence) >= self._max_chunk:
                    chunks.append("".join(sentence))
                    sentence = []
                    continue
                sentence.append(char)
        return chunks

class BirdSplitter(Splitter):
    def __init__(self, max_chunk: int = 1024):
        self._max_chunk = max_chunk

    def split(self, text: str):
        sentences = []
        js = json.loads(text)
        bird_name = None
        for key, value in js:
            if key == "en":
                bird_name = value
                continue
            if key in ("id", "cn", "sn"):
                continue
            assert bird_name != None
            head = f"The {key.lower()} of {bird_name}: "
            pieces = value.split(". ")
            sections = []
            for piece in pieces:
                sections.append(piece + ". ")
                if len(". ". join(sections)) > self._max_chunk:
                    sentences.append(head + "".join(sections))
                    sections = []
            if len(sections) > 0:
                sentences.append(head + "".join(sections))
        return sentences
