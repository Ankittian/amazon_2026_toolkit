"""
Generic string normalization for business names and addresses.

Deliberately pattern-based rather than country-keyed: the PS explicitly
warns the test set includes France, unseen in training, and forbids
hard-coding to {US, India}. Everything here is a general regex/rule that
should transfer to any Latin-script business name or address, not a lookup
table keyed by country.
"""

import re

# Legal-entity suffix normalization. Keys are patterns (word-boundary safe),
# values are a single canonical token. This list is generic corporate-suffix
# vocabulary (English + common Romance-language cognates), not tied to one
# country, since France appears only at test time.
_SUFFIX_MAP = {
    r"\bcorporation\b": "corp",
    r"\bcorp\.?\b": "corp",
    r"\bincorporated\b": "inc",
    r"\binc\.?\b": "inc",
    r"\blimited\b": "ltd",
    r"\bltd\.?\b": "ltd",
    r"\bpvt\.?\b": "pvt",
    r"\bprivate\b": "pvt",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bl\.l\.c\.?\b": "llc",
    r"\bco\.?\b": "co",
    r"\bcompany\b": "co",
    r"\bsarl\b": "sarl",          # France: société à responsabilité limitée
    r"\bsociete\b": "societe",
    r"\bsa\b": "sa",              # France: société anonyme
    r"\bplc\b": "plc",
    r"\bgmbh\b": "gmbh",
}

_ADDRESS_ABBREV_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bblvd\.?\b": "boulevard",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bfl\.?\b": "floor",
    r"\bno\.?\b": "number",
    r"\bnr\.?\b": "near",
    r"\bpo box\b": "post office box",
}


def _apply_map(text: str, mapping: dict) -> str:
    for pattern, repl in mapping.items():
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
    return text


def basic_clean(text: str) -> str:
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)   # strip punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name: str) -> str:
    text = basic_clean(name)
    text = _apply_map(text, _SUFFIX_MAP)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(addr: str) -> str:
    text = basic_clean(addr)
    text = _apply_map(text, _ADDRESS_ABBREV_MAP)
    # drop generic landmark filler words that don't identify a location
    text = re.sub(r"\b(near|opposite|behind|next to)\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_set(text: str) -> set:
    return set(t for t in text.split() if t)


def blocking_key(normalized_name: str, country: str, prefix_len: int = 4) -> str:
    """
    Coarse blocking key: country + first N alphanumeric chars of the
    normalized name, with common suffix tokens stripped first so
    'acme corp' and 'acme inc' land in the same block.
    """
    core = re.sub(r"\b(corp|inc|ltd|pvt|llc|llp|co|sarl|sa|plc|gmbh)\b", "", normalized_name)
    core = re.sub(r"\s+", "", core)
    country_key = (country or "unk").strip().lower()
    return f"{country_key}::{core[:prefix_len]}"
