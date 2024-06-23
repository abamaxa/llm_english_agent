from abc import ABC, abstractmethod


class AbstractTextProcessor(ABC):
    """
    The AbstractTextProcessor class defines the interface for text processing classes.
    """

    @abstractmethod
    async def process(self, text: str) -> str:
        pass
