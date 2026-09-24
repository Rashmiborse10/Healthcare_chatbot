
import re

INJECTION_PATTERNS = [
    r"\bignore (all |the )?(previous|above|prior) instructions?\b",
    r"\bdisregard (all|any|the) (rules|guidelines|restrictions|instructions)\b",
    r"\breveal (the |your )?system prompt\b",
    r"\bwhat (are|is) your (system prompt|instructions|rules)\b",
    r"\bact as (my|a) doctor\b",
    r"\bpretend (you|to be)\b.*\b(doctor|unrestricted|no rules|not an ai)\b",
    r"\bforget (the |your )?(knowledge base|instructions|rules|guardrails)\b",
    r"\byou are now\b",
    r"\bdeveloper mode\b|\bjailbreak\b|\bdan mode\b",
    r"\bwithout (any )?(restrictions|limitations|filters|guardrails)\b",
    r"\brespond as if you (have no|had no) (rules|restrictions)\b",
]

INJECTION_MESSAGE = (
    "I can't follow instructions that ask me to ignore my guidelines or act "
    "outside my role as a healthcare knowledge assistant. Happy to help with a "
    "healthcare or medical terminology question instead."
)

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def is_prompt_injection(text: str) -> bool:
    return any(p.search(text) for p in _compiled)


def get_injection_response() -> str:
    return INJECTION_MESSAGE


def get_matched_patterns(text: str) -> list[str]:
    return [p.pattern for p in _compiled if p.search(text)]
