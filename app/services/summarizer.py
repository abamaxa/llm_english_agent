from services.base import AbstractTextProcessor
from transformers import pipeline


class Summarizer(AbstractTextProcessor):
    """
    Summarizes the given text using a pre-trained T5 model.

    The Summarizer class is an implementation of the AbstractTextProcessor interface,
    which provides a standardized way to process text.

    The `process` method takes a string of text as input and returns a summary of that text.

    The summary is generated using the T5 model, and generates a summary half the length of
    the original, with a maximum length of MAX_TOKENS words and a minimum length of 5 words.

    MAX_TOKENS is set to 500 words to avoid excessive processing time.

    If the input text is less than 5 words, the method simply returns the original text.
    """

    MAX_TOKENS = 500

    def __init__(self, model_name: str = "t5-small") -> None:
        self.model_name = model_name
        self.model = pipeline("summarization", model=model_name)

    async def process(self, text: str) -> str:
        input_length = len(text.split())
        if len(text) < 5:
            print("This text is too short to summarize")
            return text

        max_length = max(30, input_length // 2)  # Minimum 30, maximum 500 tokens

        if max_length > self.MAX_TOKENS:
            max_length = self.MAX_TOKENS

        result = self.model(text, max_length=max_length, min_length=10)[0][
            "summary_text"
        ]

        return ".".join(result.split(" ."))
