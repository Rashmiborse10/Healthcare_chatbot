
import re

EMERGENCY_PATTERNS = [
    r"\bchest pain\b",
    r"\bheart attack\b",
    r"\bstroke\b",
    r"\bcan'?t breathe\b|\bdifficulty breathing\b|\bshortness of breath\b|\bgasping for air\b",
    r"\bsevere bleeding\b|\bbleeding (heavily|a lot|won'?t stop|profusely)\b",
    r"\bsuicidal\b|\bkill myself\b|\bwant to die\b|\bend my life\b|\bsuicide\b",
    r"\bunconscious\b|\bnot breathing\b|\bnon[- ]responsive\b|\bpassed out\b|\bcollapsed\b",
    r"\banaphylaxis\b|\bsevere allergic reaction\b|\bthroat (is )?closing\b",
    r"\bseizure\b|\bconvuls",
    r"\boverdose\b|\btook too many pills\b",
    r"\bself[- ]harm\b|\bhurting myself\b|\bcutting myself\b",
    r"\bchild (is )?(not breathing|unresponsive|choking)\b",
]

EMERGENCY_MESSAGE = (
    "This sounds like it could be a medical emergency. I'm not able to help with "
    "emergencies — please call your local emergency number right now (e.g. 911 in "
    "the US, 112 in the EU, 108 in India) or go to the nearest emergency room. "
    "If this involves thoughts of suicide or self-harm, you can also reach a crisis "
    "line such as 988 (US) or your country's equivalent. Please reach out for "
    "immediate help."
)

_compiled = [re.compile(p, re.IGNORECASE) for p in EMERGENCY_PATTERNS]


def is_emergency(text: str) -> bool:
    """Returns True if `text` matches any emergency pattern."""
    return any(p.search(text) for p in _compiled)


def get_emergency_response() -> str:
    return EMERGENCY_MESSAGE


def get_matched_patterns(text: str) -> list[str]:
    """For logging/debugging: which patterns fired (never expose to the end user)."""
    return [p.pattern for p in _compiled if p.search(text)]
