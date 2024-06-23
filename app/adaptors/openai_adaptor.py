from abc import ABC, abstractmethod
from datetime import datetime
from os import getenv, makedirs
from pathlib import Path
from time import sleep

from openai import AsyncOpenAI, BadRequestError

MODEL = getenv("OPENAI_MODEL", "gpt-3.5-turbo")
ORG_ID = getenv("OPENAI_ORG_ID")
LOG_DIR = Path(getenv("LOG_DIR", Path(__file__).parent.parent.parent / "responses"))
API_CALL_RETRIES = int(getenv("API_CALL_RETRIES", 3))


client = AsyncOpenAI(api_key=getenv("OPENAI_API_KEY"), organization=ORG_ID)


class AIModel(ABC):
    @abstractmethod
    async def call_chat(messages: list[dict], context: any):
        pass


class OpenAIModel(AIModel):
    """
    This class is a wrapper around the OpenAI API.
    """

    def __init__(self, name: str, model: str = None, log_dir: str = None, **kwargs):
        """
        Initializes an instance of the OpenAIModel class.

        Args:
            name (str): A name used to identify the class instance, used to identify the log files.

            model (str, optional): The OpenAI model to use. Defaults to the value of the OPENAI_MODEL
                environment variable, or "gpt-3.5-turbo" if not set.

            log_dir (str, optional): The directory to store log files. Defaults to the value of the
                LOG_DIR environment variable, or a directory named "responses" in the parent directory
                of the current file if not set.

            **kwargs: Additional parameters to pass to the OpenAI API.
        """
        super().__init__()
        self.name = name
        self.model = model or MODEL
        self.model_params = kwargs
        self.log_dir = Path(log_dir or LOG_DIR)

    async def call_chat(self, messages: list[dict], context: any) -> str:
        """
        Calls the OpenAI chat API to generate a response based on the provided messages.

        Args:
            messages (list[dict]): A list of message dictionaries, where each dictionary has a "role" and "content" key.

            context (any): Additional context information that is returned when the call completes, this is useful as
                    the API is called asynchronously and its is intended to support multiple concurrent requests using
                    the asyncio.gather command. By returning the context, the caller can map the result to the original
                    request.

        Returns:
            str: The generated response from the OpenAI API.
            any: the context object passed in.
        """
        trys = API_CALL_RETRIES
        while trys:
            try:
                response = await client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **self.model_params,
                )
                break
            except BadRequestError as err:
                print(err)
                raise err

            except Exception as err:
                print(err)
                sleep(1)

            trys -= 1

        return self.extract_results(response), context

    def extract_results(self, response) -> str:
        """
        Extracts the results from the OpenAI API response and logs the response.

        Args:
            response (dict): The response from the OpenAI API.

        Returns:
            str: The extracted results from the response.
        """
        results = []
        for choice in response.choices or []:
            try:
                content = choice.message.content
                results.append(content)
            except Exception as err:
                print(err)
                print(choice)
                raise err

        results = "\n".join(results)

        self.log(results, response)

        return results

    def log(self, message: str, response: dict):
        """
        Logs the given message and response to a file.

        Args:
            message (str): The message to log.
            response (dict): The response to log.
        """
        now = datetime.now()
        key = f"{self.name}-{now.time().isoformat()}"

        filepath = self.log_dir / now.date().isoformat() / f"{key}.md"

        makedirs(filepath.parent, exist_ok=True)

        with open(filepath, "w") as f:
            f.write(message)

        filepath = self.log_dir / now.date().isoformat() / f"{key}.json"
        with open(filepath, "w") as f:
            f.write(response.to_json())


def system_message(content: str) -> dict:
    return {"role": "system", "content": content}


def user_message(content: str) -> dict:
    return {"role": "user", "content": content}


def assistant_response(content: str) -> dict:
    return {"role": "assistant", "content": content}


def ok() -> dict:
    return {"role": "assistant", "content": "Ok"}
