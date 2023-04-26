import torch

from abc import ABC, abstractmethod
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, pipeline

class LLM(ABC):
    @abstractmethod
    def generate(self, context: str, query: str) -> str:
        pass

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class StablelmLLM(LLM):
    def __init__(self, model_name: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForCausalLM.from_pretrained(model_name)
        self._model.half().cuda()

        self._prompt = """<|SYSTEM|>Please read below text, understand it and answer the questions.

        {context}

        <|USER|>{query}
        <|ASSISTANT|>"""

    def generate(self, context: str, query: str) -> str:
        prompt = self._prompt.format(context=context, query=query)
        prompt_len = len(prompt) - len("<|SYSTEM|>") - len("<|USER|>") - len("<|ASSISTANT|>")
        inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        tokens = self._model.generate(
          **inputs,
          max_new_tokens=20,
          temperature=1.0,
          do_sample=True,
          stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
          pad_token_id=self._tokenizer.eos_token_id
        )
        return self._tokenizer.decode(tokens[0], skip_special_tokens=True)[prompt_len:]

class DollyLLM(LLM):
    def __init__(self, model_name: str):
        self._pipe = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        self._prompt = """Please read below text, understand it and answer the questions.

        {context}

        Question: {query}
        Answer: """

    def generate(self, context: str, query: str) -> str:
        prompt = self._prompt.format(context=context, query=query)
        prompt_len = len(prompt)
        res = self._pipe(prompt)
        return res[0]["generated_text"]
