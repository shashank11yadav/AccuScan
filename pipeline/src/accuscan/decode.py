"""CTC decoding shared with the browser (`web/src/ml/decode.ts`).

- `greedy_decode`: best path, repeats collapsed, blanks removed.
- `beam_search`: CTC prefix beam search, optionally constrained so every run of
  letters is a word from a lexicon, with a unigram prior on completed words.
- `align_confidences`: Viterbi alignment of a label sequence, giving each
  character the highest posterior among the frames it was aligned to.

The TypeScript port follows the same iteration order and tie-breaking so that
both produce the same text for `shared/fixtures/decode.json`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from .charset import BLANK, CHARSET

NEG_INF = -math.inf
DEFAULT_BEAM_WIDTH = 16
DEFAULT_LM_WEIGHT = 0.3
DEFAULT_PRUNE = 1e-3
# A dictionary reading may be at most this many nats per word less likely than
# the best unconstrained path; otherwise the greedy reading is shown instead.
DEFAULT_MAX_DROP_PER_WORD = 10.0

LogProbs = NDArray[np.floating]


@dataclass(frozen=True)
class Decoded:
    text: str
    confidences: list[float]
    fallback: bool = False


def log_add(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi = max(a, b)
    return hi + math.log1p(math.exp(-abs(a - b)))


def _labels_to_text(labels: Iterable[int]) -> str:
    return "".join(CHARSET[i] for i in labels)


def count_words(text: str) -> int:
    """Number of maximal runs of letters."""
    words = 0
    inside = False
    for ch in text:
        letter = is_letter(ch)
        if letter and not inside:
            words += 1
        inside = letter
    return words


def greedy_decode(log_probs: LogProbs) -> Decoded:
    """log_probs: [T, C]. np.argmax returns the first maximum, as the TS port does."""
    best = np.argmax(log_probs, axis=1)
    labels: list[int] = []
    confidences: list[float] = []
    previous = -1
    for t, k in enumerate(best.tolist()):
        p = math.exp(float(log_probs[t, k]))
        if k != BLANK and k != previous:
            labels.append(k)
            confidences.append(p)
        elif k != BLANK and k == previous:
            confidences[-1] = max(confidences[-1], p)
        previous = k
    return Decoded(_labels_to_text(labels), confidences)


def align_confidences(log_probs: LogProbs, labels: list[int]) -> list[float]:
    """Per-label max posterior along the Viterbi CTC alignment (0.0s if impossible)."""
    n_labels = len(labels)
    if n_labels == 0:
        return []
    t_max = log_probs.shape[0]
    ext = [BLANK]
    for label in labels:
        ext += [label, BLANK]
    s_max = len(ext)
    score = [[NEG_INF] * s_max for _ in range(t_max)]
    back = [[0] * s_max for _ in range(t_max)]
    score[0][0] = float(log_probs[0, ext[0]])
    if s_max > 1:
        score[0][1] = float(log_probs[0, ext[1]])
    for t in range(1, t_max):
        for s in range(s_max):
            best, arg = score[t - 1][s], s
            if s >= 1 and score[t - 1][s - 1] > best:
                best, arg = score[t - 1][s - 1], s - 1
            if s >= 2 and ext[s] != BLANK and ext[s] != ext[s - 2] and score[t - 1][s - 2] > best:
                best, arg = score[t - 1][s - 2], s - 2
            if best == NEG_INF:
                continue
            score[t][s] = best + float(log_probs[t, ext[s]])
            back[t][s] = arg
    last = t_max - 1
    end_candidates = [s_max - 1, s_max - 2]
    end = end_candidates[0]
    if score[last][s_max - 2] > score[last][s_max - 1]:
        end = end_candidates[1]
    if score[last][end] == NEG_INF:
        return [0.0] * n_labels
    confidences = [0.0] * n_labels
    s = end
    for t in range(last, -1, -1):
        if ext[s] != BLANK:
            idx = (s - 1) // 2
            confidences[idx] = max(confidences[idx], math.exp(float(log_probs[t, ext[s]])))
        s = back[t][s] if t > 0 else s
    return confidences


@dataclass
class TrieNode:
    children: dict[str, TrieNode] = field(default_factory=dict)
    logprob: float | None = None  # set when the path from the root spells a word


def is_letter(ch: str) -> bool:
    return ("a" <= ch <= "z") or ("A" <= ch <= "Z")


class Lexicon:
    """Lower-case word trie with unigram log-probabilities."""

    def __init__(self, root: TrieNode, size: int) -> None:
        self.root = root
        self.size = size

    @classmethod
    def from_counts(cls, counts: Mapping[str, int]) -> Lexicon:
        total = float(sum(counts.values()))
        root = TrieNode()
        for word, count in counts.items():
            node = root
            for ch in word.lower():
                node = node.children.setdefault(ch, TrieNode())
            node.logprob = math.log(count / total)
        return cls(root, len(counts))

    def logprob(self, word: str) -> float | None:
        node: TrieNode | None = self.root
        for ch in word.lower():
            node = node.children.get(ch) if node else None
        return node.logprob if node else None


@dataclass
class _Beam:
    node: TrieNode | None
    lm: float
    pb: float = NEG_INF
    pnb: float = NEG_INF

    def total(self) -> float:
        return log_add(self.pb, self.pnb)


def _advance(
    lexicon: Lexicon | None, node: TrieNode | None, lm: float, ch: str, lm_weight: float
) -> tuple[TrieNode | None, float] | None:
    """Trie state after appending ch, or None if the lexicon forbids it."""
    if lexicon is None or node is None:
        return None, lm
    root = lexicon.root
    if is_letter(ch):
        child = node.children.get(ch.lower())
        return (child, lm) if child is not None else None
    if ch == "'" and node is not root and "'" in node.children:
        return node.children["'"], lm
    if node is root:
        return root, lm
    if node.logprob is not None:
        return root, lm + lm_weight * node.logprob
    return None


def _finish(lexicon: Lexicon | None, beam: _Beam, lm_weight: float) -> float | None:
    """Final score with the trailing word completed, or None if it is not a word."""
    if lexicon is None or beam.node is lexicon.root:
        return beam.total() + beam.lm
    if beam.node is None or beam.node.logprob is None:
        return None
    return beam.total() + beam.lm + lm_weight * beam.node.logprob


def _rank_key(item: tuple[tuple[int, ...], _Beam]) -> tuple[float, tuple[int, ...]]:
    prefix, beam = item
    return (-(beam.total() + beam.lm), prefix)


def beam_search(
    log_probs: LogProbs,
    lexicon: Lexicon | None,
    beam_width: int = DEFAULT_BEAM_WIDTH,
    lm_weight: float = DEFAULT_LM_WEIGHT,
    prune: float = DEFAULT_PRUNE,
    max_drop_per_word: float = DEFAULT_MAX_DROP_PER_WORD,
) -> Decoded:
    """CTC prefix beam search.

    With a lexicon, falls back to the greedy reading (fallback=True) when no
    beam ends on a word, or when the best one is implausible next to the
    unconstrained best path, so an unknown word is shown as read rather than
    replaced by an empty or unrelated string.
    """
    log_prune = math.log(prune)
    start_node = lexicon.root if lexicon else None
    beams: list[tuple[tuple[int, ...], _Beam]] = [((), _Beam(start_node, 0.0, pb=0.0))]
    n_classes = log_probs.shape[1]
    for t in range(log_probs.shape[0]):
        row = [float(v) for v in log_probs[t]]
        candidates = [c for c in range(1, n_classes) if row[c] >= log_prune]
        nxt: dict[tuple[int, ...], _Beam] = {}
        for prefix, beam in beams:
            total = beam.total()
            same = nxt.get(prefix)
            if same is None:
                same = nxt[prefix] = _Beam(beam.node, beam.lm)
            same.pb = log_add(same.pb, total + row[BLANK])
            last = prefix[-1] if prefix else -1
            for c in candidates:
                p = row[c]
                if c == last:
                    same.pnb = log_add(same.pnb, beam.pnb + p)
                    extend = beam.pb + p
                else:
                    extend = total + p
                if extend == NEG_INF:
                    continue
                state = _advance(lexicon, beam.node, beam.lm, CHARSET[c], lm_weight)
                if state is None:
                    continue
                key = (*prefix, c)
                target = nxt.get(key)
                if target is None:
                    target = nxt[key] = _Beam(state[0], state[1])
                target.pnb = log_add(target.pnb, extend)
        beams = sorted(nxt.items(), key=_rank_key)[:beam_width]

    best_labels: tuple[int, ...] | None = None
    best_score = NEG_INF
    best_acoustic = NEG_INF
    for prefix, beam in beams:
        final = _finish(lexicon, beam, lm_weight)
        if final is not None and (best_labels is None or final > best_score):
            best_labels, best_score, best_acoustic = prefix, final, beam.total()
    if lexicon is not None:
        greedy = greedy_decode(log_probs)
        greedy_path = float(np.max(log_probs, axis=1).sum())
        allowed = max_drop_per_word * max(1, count_words(greedy.text))
        if best_labels is None or best_acoustic < greedy_path - allowed:
            return Decoded(greedy.text, greedy.confidences, fallback=True)
    if best_labels is None:  # only reachable with an empty input
        return Decoded("", [])
    labels = list(best_labels)
    return Decoded(_labels_to_text(labels), align_confidences(log_probs, labels))
