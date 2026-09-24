
import re

DIAGNOSIS_PATTERNS = [
    r"\bdo i have\b",
    r"\bam i (having|suffering from)\b",
    r"\bwhat('?s| is) wrong with me\b",
    r"\bcould (this|i) (be|have)\b.*\?",
    r"\bis this (a symptom of|cancer|serious)\b",
    r"\bdiagnose me\b",
]

PRESCRIPTION_PATTERNS = [
    r"\bwhat (medicine|drug|medication) should i take\b",
    r"\bprescribe\b",
    r"\bdosage\b|\bhow much .* should i take\b|\bhow many .* should i take\b",
    r"\bwhich pill\b",
    r"\bcan i take .* with .*\b",  # drug-interaction-style personal questions
]

TREATMENT_PATTERNS = [
    r"\bhow (do|should) i treat\b",
    r"\btreatment plan for me\b",
    r"\bwhat should i do (for|about) my\b",
    r"\bhow do i cure\b",
]

_all_patterns = (
    [(re.compile(p, re.IGNORECASE), "diagnosis") for p in DIAGNOSIS_PATTERNS]
    + [(re.compile(p, re.IGNORECASE), "prescription") for p in PRESCRIPTION_PATTERNS]
    + [(re.compile(p, re.IGNORECASE), "treatment") for p in TREATMENT_PATTERNS]
)

REFUSAL_MESSAGES = {
    "diagnosis": (
        "I cannot diagnose medical conditions. Please consult a qualified "
        "healthcare professional who can properly evaluate your symptoms."
    ),
    "prescription": (
        "I cannot recommend medications or dosages. Please speak with a doctor "
        "or pharmacist about what's appropriate for you."
    ),
    "treatment": (
        "I can't create a personal treatment plan — that needs to come from a "
        "healthcare professional who knows your medical history."
    ),
}


def check_medical_guardrail(text: str) -> str | None:
    """Returns a category-specific refusal message, or None if the request is safe
    (e.g. a general definitional question like 'What is diabetes?')."""
    for pattern, category in _all_patterns:
        if pattern.search(text):
            return REFUSAL_MESSAGES[category]
    return None


def get_matched_category(text: str) -> str | None:
    for pattern, category in _all_patterns:
        if pattern.search(text):
            return category
    return None
