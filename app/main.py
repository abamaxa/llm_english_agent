import asyncio
import os

from services import RAGStyleChecker, SimpleGrammarChecker, Summarizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_user_command():
    return input(
        """
---------------------------------------------------------------------------------------------
Enter:

1 for write_properly: Enhances both grammar and style of the input message.
2 for write_the_same_grammar_fixed: Corrects only the grammatical errors in the input message.
3 to summarize: Provides a concise summary of the input message.

or anything else to quit: """
    )


def get_text():
    return input("Enter text: ")


async def main():
    print(
        """This is a demo English text improvement tool.

It uses OpenAI's GPT-3.5 models and the HuggingFace Transformers library to correct grammar and style.

It also uses a knowledge base of English language rules to improve the text.

The tool provides 3 functions:

● write_properly: Enhances both grammar and style of the input message.
● write_the_same_grammar_fixed: Corrects only the grammatical errors in the
input message.
● summarize: Provides a concise summary of the input message.

"""
    )
    print("Loading models...")

    command_processors = {
        "1": RAGStyleChecker(),
        "2": SimpleGrammarChecker(),
        "3": Summarizer(),
    }

    while True:
        option = get_user_command()
        if option not in command_processors:
            break

        text = get_text()
        if not text:
            continue

        corrected_text = await command_processors[option].process(text)

        print(f"\nResult: {corrected_text}")


if __name__ == "__main__":
    asyncio.run(main())
