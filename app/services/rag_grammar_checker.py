from abc import abstractmethod

import faiss
from adaptors.openai_adaptor import OpenAIModel, system_message
from sentence_transformers import SentenceTransformer
from services.base import AbstractTextProcessor

# Create a simple knowledge base
knowledge_base = [
    "Use 'a' before consonant sounds and 'an' before vowel sounds.",
    "Always capitalize the first letter of a sentence and proper nouns.",
    "Use past tense for completed actions and present tense for current or habitual actions.",
    "Adjectives usually come before the noun they describe.",
    "The order of a basic positive sentence is Subject-Verb-Object. (Negative and question "
    "sentences may have a different order.)",  # noqa
    "Every sentence must have a subject and a verb. An object is optional. Note that an "
    "imperative sentence may have a verb only, but the subject is understood.",
    "The subject and verb must agree in number, that is a singular subject needs a singular "
    "verb and a plural subject needs a plural verb.",
    "When two singular subjects are connected by or, use a singular verb. The same is true "
    "for either/or and neither/nor.",
    "When using two or more adjectives together, the usual order is opinion-adjective + "
    "fact-adjective + noun. (There are some additional rules for the order of fact adjectives.)",
    "Treat collective nouns (e.g. committee, company, board of directors) as singular OR plural. "
    "In BrE a collective noun is usually treated as plural, needing a plural verb and pronoun. "
    "In AmE a collective noun is often treated as singular, needing a singular verb and pronoun.",
    "Use a comma before a coordinating conjunction, unless the conjunction is the first word in the sentence.",
    "The words its and it's are two different words with different meanings.",
    "The words your and you're are two different words with different meanings.",
    "The words there, their and they're are three different words with different meanings.",
    "The words we, our and we're are three different words with different meanings.",
    "The contraction he's can mean he is OR he has. Similarly, she's can mean she is OR she has, "
    "and it's can mean it is OR it has, and John's can mean John is OR John has.",
    "The contraction he'd can mean he had OR he would. Similarly, they'd can mean they had OR they would.",
    "Use the indefinite article a/an for countable nouns in general. Use the definite article the for "
    "specific countable nouns and all uncountable nouns.",
    "Use the indefinite article a with words beginning with a consonant sound. Use the indefinite article "
    "an with words beginning with a vowel sound. see When to Say a or an",
    "Use many or few with countable nouns. Use much/a lot or little for uncountable nouns.",
    "To show possession (who is the owner of something) use an apostrophe + s for singular owners, and "
    "s + apostrophe for plural owners.",
    "In general, use the active voice (Cats eat fish) in preference to the passive voice (Fish are eaten by cats).",
]


class BaseRAGStyleChecker(AbstractTextProcessor):
    """
    Base class for RAG-style text processors that use a knowledge base and OpenAI model to improve text.

    This class provides common functionality for retrieving relevant knowledge from a knowledge base,
    constructing prompts for the OpenAI model, and processing text.

    Subclasses must implement the `get_prompt()` method to define the specific prompt format for their
    use case.
    """

    def __init__(self) -> None:
        self.model = OpenAIModel(self.__class__.__name__)
        self.sentence_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = self._create_index()

    @abstractmethod
    def get_prompt(self, text: str, relevant_knowledge: list[str]) -> str:
        """
        Get the prompt for the OpenAI model based on the given text and relevant knowledge.

        This method must be implemented by subclasses of `BaseRAGStyleChecker` to define the specific
        prompt format for their use case. The prompt should include the original text to be improved,
        as well as the relevant language rules from the knowledge base that should be applied.

        Args:
            text (str): The original text to be improved.
            relevant_knowledge (List[str]): A list of relevant language rules from the knowledge base.

        Returns:
            str: The prompt for the OpenAI model.
        """
        raise NotImplementedError

    def _create_index(self):
        """
        Creates a vector index for fast similarity search over the knowledge base.

        The knowledge base is encoded using the SentenceTransformer model, and the
        resulting embeddings are used to create a FAISS IndexFlatL2 index. This index
        allows for efficient nearest neighbor search to find the most relevant knowledge
        base entries for a given query.
        """
        # Encode knowledge base
        knowledge_embeddings = self.sentence_encoder.encode(knowledge_base)

        # Create FAISS index for fast similarity search
        index = faiss.IndexFlatL2(knowledge_embeddings.shape[1])
        index.add(knowledge_embeddings.astype("float32"))

        return index

    def retrieve_relevant_knowledge(self, query, top_k=2) -> list[str]:
        # do a vector seach and pick the top 2 matches to suggest as relevant rules
        query_embedding = self.sentence_encoder.encode([query])
        _, indices = self.index.search(query_embedding.astype("float32"), top_k)
        return [knowledge_base[i] for i in indices[0]]

    async def process(self, text: str) -> str:
        # Retrieve relevant knowledge
        relevant_knowledge = self.retrieve_relevant_knowledge(text)

        # Construct prompt for OpenAI
        prompt = self.get_prompt(text, relevant_knowledge)

        # Get response from OpenAI
        response = await self.model.call_chat([system_message(prompt)], text)

        return response[0].strip()


class RAGStyleChecker(BaseRAGStyleChecker):
    def get_prompt(self, text: str, relevant_knowledge: str) -> str:
        return f"""Role: You are an expert in written Standard English.

    Context: Your task is to improve the grammar and style of the following text, taking account of
    the Relevant language rules given below:

    Original: {text}

    Relevant language rules:
    {' '.join(relevant_knowledge)}

    Instructions:

    Think step by step.

    First, decide if the meaning of the text is clear. If it contains ambiguities or context-specific nuances then
    output the word "AMBIGOUS TEXT" followed by a clear and concise explanation of the problem and stop.

    Otherwise, provide an improved version of the text, using the Relevant language rules, explaining the
    changes made and offering additional suggestions for improvement.

    Focus on producing Standard English, including vocabulary and natural phrasing.

    Do not change the meaning of the text.

    Improved text:
    """


class RAGGrammarChecker(BaseRAGStyleChecker):
    def get_prompt(self, text: str, relevant_knowledge: str) -> str:
        return f"""As an expert in English syntax, morphology and semantics but know nothing else
    about the English language.

    Fix any syntax, morphology or semantics errors in the following text:

    Original: {text}

    Relevant language rules:
    {' '.join(relevant_knowledge)}

    Do not change the meaning or style of the text.

    Do not change the spelling of any words unless it is necessary to comply with the Relevant language rules.

    Focus only on the syntax, morphology and semantics of the text.

    Example:
    Original: When I was at skool, they learned me up how to talk proper
    Corrected: When I was at skool, they taught me how to talk.

    Corrected text:
    """
