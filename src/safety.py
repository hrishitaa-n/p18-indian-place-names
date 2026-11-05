"""Output guardrail: blocks profanity, sexual terms, hate/caste slurs in generated names.
Short terms are matched as whole words (so 'Assam' and 'Rashitpur' pass);
long unambiguous terms are also matched as substrings (so 'Chutiyapur' is caught)."""
import re

BLOCK_WORDS = {
    # English
    "sex", "porn", "rape", "shit", "damn", "bitch", "bastard", "whore", "slut", "dick",
    "cock", "cunt", "ass", "piss", "fuck", "nazi", "hitler",
    # transliterated Hindi / Hinglish vulgarities
    "gand", "gaand", "gandu", "lund", "lauda", "randi", "harami", "kutta", "kamina",
    "saala", "chut", "chod", "chutiya", "madarchod", "behenchod", "bhenchod", "bhosdi",
    # communal / caste slurs
    "chamar", "bhangi", "kafir", "katua",
}
BLOCK_SUBSTR = ("chutiya", "madarchod", "behenchod", "bhenchod", "bhosdi", "fuck", "nigger")

def is_safe(name: str) -> bool:
    words = re.findall(r"[a-z]+", name.lower())
    if any(w in BLOCK_WORDS for w in words):
        return False
    squashed = "".join(words)
    return not any(s in squashed for s in BLOCK_SUBSTR)

def filter_names(names):
    """Returns (safe_names, number_blocked)."""
    safe = [n for n in names if is_safe(n)]
    return safe, len(names) - len(safe)
