"""Text normalization, script detection, and conservative Indic transliteration helpers."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
LATIN = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[\w]+(?:['’][\w]+)?|[^\w\s]", re.UNICODE)
ENGLISH_HINTS = {
    "a", "an", "and", "are", "at", "but", "by", "can", "do", "for", "from", "good",
    "have", "he", "her", "hey", "how", "i", "in", "is", "it", "like", "love", "me",
    "my", "no", "not", "of", "on", "or", "our", "please", "so", "that", "the", "this",
    "to", "was", "we", "what", "when", "where", "with", "you", "your",
}


def normalize(text: str, *, lowercase: bool = True) -> str:
    """Apply only loss-minimizing normalization; preserve emoji and punctuation."""
    text = unicodedata.normalize("NFC", str(text))
    text = re.sub(r"https?://\S+|www\.\S+", " <URL> ", text)
    text = re.sub(r"@[\w_]+", " <USER> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower() if lowercase else text


def words(text: str) -> list[str]:
    return WORD_RE.findall(text)


def script_of(word: str) -> str:
    """Return deva, latin, mixed, or other at the word level."""
    deva, latin = bool(DEVANAGARI.search(word)), bool(LATIN.search(word))
    if deva and latin:
        return "mixed"
    if deva:
        return "deva"
    if latin:
        return "latin"
    return "other"


def language_of(word: str, english_hints: set[str] | None = None) -> str:
    """A deliberately simple weak LID heuristic; replace with gold/predicted LID when available."""
    script = script_of(word)
    if script == "deva":
        return "HI"
    if script == "mixed":
        return "MIX"
    if script == "latin":
        return "EN" if word.casefold() in (english_hints or ENGLISH_HINTS) else "ROM"
    return "OTHER"


@lru_cache(maxsize=100_000)
def devanagari_to_itrans(word: str) -> str:
    """Transliterate Devanagari to ITRANS when available, otherwise keep input unchanged."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        return transliterate(word, sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    except (ImportError, ValueError):
        return word


@lru_cache(maxsize=100_000)
def itrans_to_devanagari(word: str) -> str:
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate

        return transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
    except (ImportError, ValueError):
        return word


def transliteration_pair(word: str) -> tuple[str, str] | None:
    """Return (original, transliterated) only when the source is clearly Devanagari."""
    if script_of(word) == "deva":
        other = devanagari_to_itrans(word)
        return (word, other) if other != word else None
    return None


def augment_with_transliteration(text: str) -> list[str]:
    """Make one minimally augmented view with Devanagari words rendered in ITRANS."""
    out = []
    for token in words(text):
        out.append(devanagari_to_itrans(token) if script_of(token) == "deva" else token)
    roman = " ".join(out)
    return [text, roman] if roman != text else [text]


def tag_language(text: str) -> str:
    """Prefix each lexical item with a weak language/script tag for segmentation guidance."""
    tagged = []
    for token in words(text):
        kind = language_of(token)
        tag = {"EN": "<EN>", "ROM": "<ROM>", "HI": "<HI>", "MIX": "<MIX>", "OTHER": "<OTHER>"}[kind]
        tagged.append(f"{tag} {token}" if re.search(r"\w", token, re.UNICODE) else token)
    return " ".join(tagged)


def phone_proxy(text: str) -> str:
    """Create a transparent grapheme-to-phone proxy for cross-script experiments.

    This is not a production G2P system: ITRANS approximates Devanagari sounds, and English
    spellings are left in a coarse Latin phone alphabet. Results must be described as a proxy.
    """
    # Digraphs are collapsed before single-character mapping, retaining a reversible-ish alphabet.
    pairs = {"kh": "K", "gh": "G", "ch": "C", "jh": "J", "th": "T", "dh": "D",
             "ph": "P", "bh": "B", "sh": "S", "ai": "E", "au": "O", "aa": "A",
             "ee": "I", "oo": "U", "ng": "N", "ny": "Y", "chh": "Q"}
    singles = dict(zip("abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"))
    rendered = []
    for token in words(text):
        form = devanagari_to_itrans(token) if script_of(token) == "deva" else token.casefold()
        if not re.search(r"[a-z]", form):
            rendered.append(form)
            continue
        i, phones = 0, []
        keys = sorted(pairs, key=len, reverse=True)
        while i < len(form):
            found = next((k for k in keys if form.startswith(k, i)), None)
            if found:
                phones.append(pairs[found]); i += len(found)
            elif form[i].isalpha():
                phones.append(singles.get(form[i], form[i])); i += 1
            else:
                phones.append(form[i]); i += 1
        rendered.append(" ".join(phones))
    return " | ".join(rendered)

