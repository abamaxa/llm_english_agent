from services.base import AbstractTextProcessor
from transformers import pipeline


class SimpleGrammarChecker(AbstractTextProcessor):
    """
    Provides a simple grammar checker service using a pre-trained Transformer model.

    The `SimpleGrammarChecker` class is an implementation of the `AbstractTextProcessor` interface,
    which processes input text and returns corrected text.

    The class uses the `prithivida/grammar_error_correcter_v1` Transformer model to perform grammar
    correction. It limits the maximum number of tokens in the output to MAX_TOKEN to avoid excessive
    processing time.
    """

    MAX_TOKENS = 500

    def __init__(self, model_name: str = "prithivida/grammar_error_correcter_v1"):
        self.model_name = model_name
        self.model = pipeline("text2text-generation", model=model_name)

    async def process(self, text: str) -> str:
        input_length = len(text.split())
        max_length = input_length * 1.2

        if max_length > self.MAX_TOKENS:
            max_length = self.MAX_TOKENS

        return self.model(text, max_length=max_length)[0]["generated_text"]
