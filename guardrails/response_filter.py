
import re

# Patterns suggesting the model gave personalized clinical advice
# despite input-side guardrails.
UNSAFE_OUTPUT_PATTERNS = [
    r"\byou should take\b.*\b(mg|ml|tablet|pill|dose|capsule)s?\b",
    r"\bi (would )?diagnose\b",
    r"\byou (have|are suffering from)\b\s+\w+.*\b(disease|condition|disorder|cancer)\b",
    r"\btake \d+\s?(mg|ml|tablets?|pills?|capsules?)\b",
    r"\bi recommend (you )?(take|start|stop) (taking )?\b",
]

# Phrases suggesting the model is speaking from general/opinion
# knowledge rather than the retrieved context — a soft hallucination
# signal, not an automatic block.
UNGROUNDED_PHRASES = [
    r"\bin my opinion\b",
    r"\bi believe\b",
    r"\bgenerally speaking, doctors (often |usually )?recommend\b",
    r"\bas far as i know\b",
    r"\bfrom what i understand\b",
]

FALLBACK_MESSAGE = (
    "I generated a response, but it may have drifted into personalized medical "
    "advice, which I'm not able to provide. Please consult a qualified healthcare "
    "professional for guidance specific to you."
)

_unsafe_compiled = [re.compile(p, re.IGNORECASE) for p in UNSAFE_OUTPUT_PATTERNS]
_ungrounded_compiled = [re.compile(p, re.IGNORECASE) for p in UNGROUNDED_PHRASES]


def contains_unsafe_output(text: str) -> bool:
    """Hard check — if True, the response MUST be replaced, not shown."""
    return any(p.search(text) for p in _unsafe_compiled)


def contains_ungrounded_language(text: str) -> bool:
    """Soft check — worth logging/flagging for prompt-tuning, not blocking on its own."""
    return any(p.search(text) for p in _ungrounded_compiled)


def filter_response(answer: str) -> tuple[str, bool]:
    """
    Runs the full output-side check.

    Returns (final_answer, was_blocked). If was_blocked is True, the
    caller should log the original answer for guardrail-quality
    review (do NOT show the original unsafe text to the user), and
    persist `final_answer` (the fallback) as what's actually shown.
    """
    if contains_unsafe_output(answer):
        return FALLBACK_MESSAGE, True
    return answer, False
