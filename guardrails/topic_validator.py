
import re

OFF_TOPIC_PATTERNS = [
    r"\btell me a joke\b",
    r"\bwrite\b.{0,20}\bcode\b",
    r"\bwho won\b.*\b(ipl|match|game|election|world cup)\b",
    r"\bstock price\b|\bshare price\b",
    r"\bweather\b(?!.*\b(effect|impact)\b)",
    r"\brecipe for\b",
    r"\bmovie recommendation\b|\bwhat should i watch\b",
    r"\bwrite (a )?(poem|song|story)\b",
    r"\btranslate\b.*\bto\b",
]

OUT_OF_SCOPE_MESSAGE = (
    "I'm a healthcare knowledge assistant, so I can only help with questions "
    "about medical terminology, patient care guidelines, common diseases, and "
    "healthcare insurance terms based on the documents I have access to. Could "
    "you rephrase your question around one of those topics?"
)

NOT_IN_KB_MESSAGE = (
    "I couldn't find anything about that in the healthcare knowledge base I have "
    "access to, so I don't want to guess. Could you rephrase, or ask about medical "
    "terminology, patient care guidelines, common diseases, or insurance terms?"
)

# FAISS L2 distance — lower means closer/more relevant. Above this,
# treat the retrieval as "nothing relevant found."
NO_MATCH_DISTANCE_THRESHOLD = 1.1

_compiled = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]


def is_obviously_off_topic(text: str) -> bool:
    return any(p.search(text) for p in _compiled)


def get_off_topic_response() -> str:
    return OUT_OF_SCOPE_MESSAGE


def validate_against_retrieval(results: list[tuple[object, float]]) -> bool:
    """
    `results` is the output of vector_store.similarity_search_with_score():
    a list of (document, distance_score) tuples. Returns True if at least
    one result is close enough to count as "in scope."
    """
    if not results:
        return False
    return any(score <= NO_MATCH_DISTANCE_THRESHOLD for _, score in results)


def get_not_in_kb_response() -> str:
    return NOT_IN_KB_MESSAGE
