from adaptors.openai_adaptor import OpenAIModel, system_message
from services.base import AbstractTextProcessor


class StandardEnglishChecker(AbstractTextProcessor):
    def __init__(self, grammar_checker: AbstractTextProcessor) -> None:
        self.model = OpenAIModel("standard-english")
        self.grammar_checker = grammar_checker

    async def process(self, text: str) -> str:
        # First, correct grammar
        corrected_text = await self.grammar_checker.process(text)

        # Construct prompt for OpenAI
        prompt = f"""Role: You are an expert in written Standard English.

    Context: Correct the following text:

    Original: {text}
    Grammatically corrected: {corrected_text}

    Instructions:

    Think step by step.

    First, decide if the meaning of the text is clear. If it contains ambiguities or context-specific nuances then
    output the word "AMBIGOUS" followed by a clear and concise explanation of the problem and stop.

    Otherwise, provide an improved version of the text, using standard English, explaining the changes made and
    offering additional suggestions for improvement. Focus on producing Standard English, including vocabulary
    and natural phrasing. Do not change the meaning of the text.

    Improved text:
    """
        # Get response from OpenAI
        response = await self.model.call_chat([system_message(prompt)], text)

        return response[0].strip()
