"""Bundled text resources: the sentence corpus for synthetic lines and the lexicon.

`accuscan resources` downloads the pinned sources (see configs/resources.toml),
strips the distributor's header and footer from each book, and writes:

- src/accuscan/resources/corpus/pg<ID>.txt.gz  (bundled books only)
- web/public/lexicon/words.txt.gz             ("word<TAB>count" lines, most frequent first)

Both outputs are committed, so training never needs the network for text.
"""

from __future__ import annotations

import gzip
import re
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .download import fetch
from .paths import CONFIG_DIR, LEXICON_PATH, RESOURCES_DIR

RESOURCES_CONFIG = CONFIG_DIR / "resources.toml"
CORPUS_DIR = RESOURCES_DIR / "corpus"
SUPPLEMENT_PATH = RESOURCES_DIR / "supplement.txt"
LEXICON_SIZE = 50_000
WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
_START = re.compile(r"^\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
_END = re.compile(r"^\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*$", re.M)
# Regular inflections used to extend words seen in the books with forms ENABLE lists.
_SUFFIXES = ("s", "es", "ed", "d", "ing", "er", "ers", "est", "ly", "ness", "ment", "ments")


@dataclass(frozen=True)
class Book:
    id: str
    title: str
    author: str
    url: str
    sha256: str
    bundle: bool


def load_books(path: Path = RESOURCES_CONFIG) -> tuple[list[Book], tuple[str, str]]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    books = [Book(**b) for b in raw["book"]]
    wordlist = (raw["wordlist"]["url"], raw["wordlist"]["sha256"])
    return books, wordlist


def strip_boilerplate(text: str) -> str:
    """Body between the distributor's START and END markers, normalised to LF."""
    text = text.replace("\r\n", "\n").lstrip("﻿")
    start = _START.search(text)
    end = _END.search(text)
    if not start or not end or end.start() <= start.end():
        raise ValueError("start/end markers not found")
    return text[start.end() : end.start()].strip() + "\n"


def count_words(texts: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(w.lower() for w in WORD_RE.findall(text))
    return counts


def inflections(word: str, vocabulary: set[str]) -> set[str]:
    stems = {word}
    if word.endswith(("e", "y")):
        stems.add(word[:-1])
    found = set()
    for stem in stems:
        for suffix in (*_SUFFIXES, "ies", "ied"):
            if stem + suffix in vocabulary:
                found.add(stem + suffix)
    return found


def build_lexicon(
    counts: Counter[str], vocabulary: set[str], supplement: list[str], size: int = LEXICON_SIZE
) -> list[tuple[str, int]]:
    """Words from the vocabulary seen in the corpus, their inflections, and the supplement.

    Ranked by corpus count (inflections and supplement words get a count of 1
    unless seen), truncated to `size`.
    """
    seen = {w for w in counts if w in vocabulary}
    words = set(seen)
    for word in seen:
        words |= inflections(word, vocabulary)
    words |= set(supplement)
    ranked = sorted(words, key=lambda w: (-max(1, counts.get(w, 0)), w))[:size]
    return [(w, max(1, counts.get(w, 0))) for w in ranked]


def read_supplement(path: Path = SUPPLEMENT_PATH) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip().lower() for ln in lines if ln.strip() and not ln.startswith("#")]


def write_lexicon(entries: list[tuple[str, int]], path: Path = LEXICON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{w}\t{c}\n" for w, c in entries).encode("utf-8")
    # mtime=0 keeps the archive byte-identical across rebuilds.
    path.write_bytes(gzip.compress(body, compresslevel=9, mtime=0))


def read_lexicon(path: Path = LEXICON_PATH) -> dict[str, int]:
    text = gzip.decompress(path.read_bytes()).decode("utf-8")
    out: dict[str, int] = {}
    for line in text.splitlines():
        word, count = line.split("\t")
        out[word] = int(count)
    return out


def corpus_texts(directory: Path = CORPUS_DIR) -> dict[str, str]:
    """Bundled book bodies keyed by file stem (e.g. 'pg1342')."""
    return {
        p.name.removesuffix(".txt.gz"): gzip.decompress(p.read_bytes()).decode("utf-8")
        for p in sorted(directory.glob("*.txt.gz"))
    }


def build_resources(log: Callable[[str], None] = print) -> None:
    books, (wordlist_url, wordlist_sha) = load_books()
    vocabulary = {
        w.strip()
        for w in fetch(wordlist_url, wordlist_sha).decode("utf-8").splitlines()
        if w.strip().isalpha()
    }
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    texts = []
    for book in books:
        body = strip_boilerplate(fetch(book.url, book.sha256).decode("utf-8-sig"))
        texts.append(body)
        if book.bundle:
            out = CORPUS_DIR / f"pg{book.id}.txt.gz"
            out.write_bytes(gzip.compress(body.encode("utf-8"), compresslevel=9, mtime=0))
        log(f"{book.title}: {len(body):,} characters")
    entries = build_lexicon(count_words(texts), vocabulary, read_supplement())
    write_lexicon(entries)
    log(f"lexicon: {len(entries):,} words -> {LEXICON_PATH}")
