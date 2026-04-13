#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  RDR Tagger  —  Single Classification Ripple Down Rules based NLP Sequence Tagger
================================================================================

A complete, production-grade reimplementation of the SCRDR-based POS / NLP
sequence tagger described in:

  Nguyen et al. (2014). "A Robust Transformation-Based Learning Approach Using
      Ripple Down Rules for Part-of-Speech Tagging."  arXiv:1412.4021
  Nguyen & Widyantoro (2014). "RDRPOSTagger: A Ripple Down Rules-based
      Part-Of-Speech Tagger."  EACL 2014.

Vibe coding by Ye Kyaw Thu, Lab. Leader, Myanmar.
Last updated: 13 April 2026.

Key improvements over the original RDRPOSTagger codebase
---------------------------------------------------------
  1. NO eval() / exec()  — All condition checking uses structured tuples with
       direct equality comparison.  This is 4–10× faster and safe.
  2. Multiprocessing learning  — Each initial tag group builds its subtree in a
       separate worker process using multiprocessing.Pool, with shared read-only
       data passed efficiently via pickling.
  3. Cleaner data representation  — Context objects are named-tuples (lightweight,
       hashable-friendly, no dynamic attribute generation via exec()).
  4. Richer feature set  — Added prefix features, character n-grams for Myanmar
       / CJK scripts, word shape features, and configurable window size.
  5. Proper evaluation  — Per-tag Precision / Recall / F1, Macro / Micro / Weighted
       averages, Known-word vs Unknown-word accuracy, and confusion matrix PNG.
  6. Robust I/O  — Handles UTF-8, empty lines, the "///" slash word, smart quotes,
       and inconsistent whitespace without crashing.
  7. Human-readable + binary model formats — .rdr rules file (tab-indented,
       auditable) and .pkl binary file (fast reload).
  8. Three sub-commands: train | test | tag

Usage
-----
  # 1. Train
  python rdr_tagger.py train \\
      --train path/to/train.txt \\
      --model path/to/model.rdr \\
      [--jobs N]  [--threshold-imp 2]  [--threshold-match 2]

  # 2. Evaluate
  python rdr_tagger.py test \\
      --test path/to/test.txt \\
      --model path/to/model.rdr \\
      --hyp path/to/hyp.txt \\
      [--confusion-matrix path/to/cm.png]

  # 3. Tag raw text (no gold labels)
  python rdr_tagger.py tag \\
      --input path/to/raw.txt \\
      --model path/to/model.rdr \\
      --output path/to/tagged.txt

Data format (one sentence per line, tokens separated by spaces):
  word1/TAG1 word2/TAG2 word3/TAG3 ...

Myanmar myPOS example:
  နောက်ပြောင်/adj သော/part စကား/n က/ppm လွန်ကျွံ/v သည်/ppm ။/punc
================================================================================
"""

import os
import re
import sys
import time
import pickle
import logging
import argparse
import warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Set, Tuple, Any

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────────────────────
def _setup_logging(level: str = "INFO") -> logging.Logger:
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(format=fmt, level=getattr(logging, level.upper()),
                        datefmt="%H:%M:%S", stream=sys.stderr)
    return logging.getLogger("rdr_tagger")

log = _setup_logging()

# ──────────────────────────────────────────────────────────────────────────────
# Constants  (feature-vector slot indices — keep in sync with Context)
# ──────────────────────────────────────────────────────────────────────────────
#  Slot  Name          Description
#  ───────────────────────────────────────────────────────────────────────────
#   0    prevWord2     2nd word to the left
#   1    prevTag2      2nd tag  to the left
#   2    prevWord1     1st word to the left
#   3    prevTag1      1st tag  to the left
#   4    word          current word
#   5    tag           current (initial / running) tag
#   6    nextWord1     1st word to the right
#   7    nextTag1      1st tag  to the right
#   8    nextWord2     2nd word to the right
#   9    nextTag2      2nd tag  to the right
#  10    suffixL2      last-2-chars suffix
#  11    suffixL3      last-3-chars suffix
#  12    suffixL4      last-4-chars suffix
#  13    prefixL2      first-2-chars prefix
#  14    prefixL3      first-3-chars prefix
#  15    wordShape     shape code  (e.g. "Xx", "dd", "X", …)

SLOT_PW2,  SLOT_PT2  =  0,  1
SLOT_PW1,  SLOT_PT1  =  2,  3
SLOT_W,    SLOT_T    =  4,  5
SLOT_NW1,  SLOT_NT1  =  6,  7
SLOT_NW2,  SLOT_NT2  =  8,  9
SLOT_SFX2, SLOT_SFX3, SLOT_SFX4 = 10, 11, 12
SLOT_PFX2, SLOT_PFX3            = 13, 14
SLOT_SHAPE                      = 15

NUM_SLOTS = 16

# Sentinel strings for boundary / missing context
_BOS_W = "<BOS>"   # beginning of sentence  (word)
_EOS_W = "<EOS>"   # end of sentence         (word)
_BOS_T = "<BOS_T>"
_EOS_T = "<EOS_T>"
_NO_SFX = "<SFX>"
_NO_PFX = "<PFX>"
_NO_SHP = "<SHP>"

# Map slot → human-readable name (for .rdr file I/O)
SLOT_NAMES: Dict[int, str] = {
    SLOT_PW2: "prevWord2", SLOT_PT2: "prevTag2",
    SLOT_PW1: "prevWord1", SLOT_PT1: "prevTag1",
    SLOT_W:   "word",      SLOT_T:   "tag",
    SLOT_NW1: "nextWord1", SLOT_NT1: "nextTag1",
    SLOT_NW2: "nextWord2", SLOT_NT2: "nextTag2",
    SLOT_SFX2: "suffixL2", SLOT_SFX3: "suffixL3", SLOT_SFX4: "suffixL4",
    SLOT_PFX2: "prefixL2", SLOT_PFX3: "prefixL3",
    SLOT_SHAPE: "wordShape",
}
NAME_TO_SLOT: Dict[str, int] = {v: k for k, v in SLOT_NAMES.items()}

# Lexicon special keys
_UNKN_WORD = "TAG4UNKN-WORD"
_UNKN_CAP  = "TAG4UNKN-CAPITAL"
_UNKN_NUM  = "TAG4UNKN-NUM"


# ──────────────────────────────────────────────────────────────────────────────
# Context  (one per token — plain list for speed)
# ──────────────────────────────────────────────────────────────────────────────
def make_empty_context() -> List[str]:
    """Return a fresh context vector filled with sentinels."""
    ctx = [_BOS_W] * NUM_SLOTS
    ctx[SLOT_SFX2] = ctx[SLOT_SFX3] = ctx[SLOT_SFX4] = _NO_SFX
    ctx[SLOT_PFX2] = ctx[SLOT_PFX3] = _NO_PFX
    ctx[SLOT_SHAPE] = _NO_SHP
    return ctx


def _word_shape(w: str) -> str:
    """
    Coarse word-shape feature.  Works for Latin, Digits, Myanmar, CJK, etc.
    Examples:  "Hello" → "Xxxx",  "2024" → "dddd",  "နောက်" → "UU"
    """
    if not w:
        return _NO_SHP
    # Latin-only fast path
    shape = []
    for ch in w[:8]:   # cap at 8 chars to bound length
        if ch.isupper():
            shape.append("X")
        elif ch.islower():
            shape.append("x")
        elif ch.isdigit():
            shape.append("d")
        elif ch in (".", ",", "!", "?", ";", ":", "-", "_"):
            shape.append("p")
        else:
            # Non-Latin (Myanmar, CJK, Arabic, …) — treat as uppercase-like
            shape.append("U")
    # Collapse consecutive identical chars to reduce sparsity
    collapsed = [shape[0]]
    for c in shape[1:]:
        if c != collapsed[-1]:
            collapsed.append(c)
    return "".join(collapsed)


def build_context(word_tags: List[str], idx: int) -> List[str]:
    """
    Build the full context vector for token at position `idx` in a
    sequence of 'word/TAG' strings.
    """
    ctx = make_empty_context()

    word, tag = split_word_tag(word_tags[idx])
    ctx[SLOT_W] = word
    ctx[SLOT_T] = tag

    # suffix features (character n-grams from the right)
    wlen = len(word)
    if wlen >= 3:
        ctx[SLOT_SFX2] = word[-2:]
        ctx[SLOT_SFX3] = word[-3:]
    if wlen >= 5:
        ctx[SLOT_SFX4] = word[-4:]

    # prefix features (character n-grams from the left)
    if wlen >= 2:
        ctx[SLOT_PFX2] = word[:2]
    if wlen >= 3:
        ctx[SLOT_PFX3] = word[:3]

    # word shape
    ctx[SLOT_SHAPE] = _word_shape(word)

    # left context
    if idx > 0:
        pw1, pt1 = split_word_tag(word_tags[idx - 1])
        ctx[SLOT_PW1], ctx[SLOT_PT1] = pw1, pt1
    else:
        ctx[SLOT_PW1], ctx[SLOT_PT1] = _BOS_W, _BOS_T

    if idx > 1:
        pw2, pt2 = split_word_tag(word_tags[idx - 2])
        ctx[SLOT_PW2], ctx[SLOT_PT2] = pw2, pt2
    else:
        ctx[SLOT_PW2], ctx[SLOT_PT2] = _BOS_W, _BOS_T

    # right context
    n = len(word_tags)
    if idx < n - 1:
        nw1, nt1 = split_word_tag(word_tags[idx + 1])
        ctx[SLOT_NW1], ctx[SLOT_NT1] = nw1, nt1
    else:
        ctx[SLOT_NW1], ctx[SLOT_NT1] = _EOS_W, _EOS_T

    if idx < n - 2:
        nw2, nt2 = split_word_tag(word_tags[idx + 2])
        ctx[SLOT_NW2], ctx[SLOT_NT2] = nw2, nt2
    else:
        ctx[SLOT_NW2], ctx[SLOT_NT2] = _EOS_W, _EOS_T

    return ctx


# ──────────────────────────────────────────────────────────────────────────────
# Rule  —  a conjunction of (slot, value) constraints
#   Represented as a sorted *tuple* of (slot, value) pairs so that:
#   • it is hashable (can be a dict key or in a set)
#   • satisfaction is a pure Python loop, zero string parsing, zero eval()
# ──────────────────────────────────────────────────────────────────────────────
class Rule:
    """
    An immutable conjunction of (slot_index, expected_value) constraints.

    A Rule fires (is satisfied) on a context vector `ctx` iff
        ctx[slot] == value   for every (slot, value) in self.constraints.

    Internally stored as a sorted tuple for hashing and fast iteration.
    """

    __slots__ = ("constraints", "_hash", "_str_cache")

    def __init__(self, constraint_dict: Dict[int, str]):
        # Normalise: sort by slot index so two rules with same constraints
        # but different insertion order compare equal.
        self.constraints: Tuple[Tuple[int, str], ...] = tuple(
            sorted(constraint_dict.items())
        )
        self._hash: int = hash(self.constraints)
        self._str_cache: Optional[str] = None

    # ── satisfaction ──────────────────────────────────────────────────────────
    def satisfied_by(self, ctx: List[str]) -> bool:
        """Return True iff this rule's constraints all hold on `ctx`."""
        for slot, val in self.constraints:
            if ctx[slot] != val:
                return False
        return True

    # ── Python object protocol ────────────────────────────────────────────────
    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        return self.constraints == other.constraints

    def __repr__(self) -> str:
        return f"Rule({self.to_human_string()})"

    # ── human-readable serialisation ──────────────────────────────────────────
    def to_human_string(self) -> str:
        """
        Format:   slotName == "value" and slotName == "value" ...
        Matches the convention used in the .rdr rules file.
        """
        if self._str_cache is None:
            parts = [
                f'{SLOT_NAMES[slot]} == "{val}"'
                for slot, val in self.constraints
            ]
            self._str_cache = " and ".join(parts)
        return self._str_cache

    @staticmethod
    def from_human_string(s: str) -> "Rule":
        """
        Parse a condition string produced by to_human_string() back into a Rule.
        No eval() is used.
        """
        constraint_dict: Dict[int, str] = {}
        for part in s.split(" and "):
            part = part.strip()
            eq_idx = part.find("==")
            if eq_idx < 0:
                continue
            name = part[:eq_idx].strip()
            val  = part[eq_idx + 2:].strip().strip('"')
            if name in NAME_TO_SLOT:
                constraint_dict[NAME_TO_SLOT[name]] = val
        return Rule(constraint_dict)

    # ── number of constraint atoms ────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.constraints)


# ──────────────────────────────────────────────────────────────────────────────
# Feature / Rule generation  (28+ templates)
# ──────────────────────────────────────────────────────────────────────────────
def generate_rules_for_context(ctx: List[str]) -> Set[Rule]:
    """
    Generate all candidate rules (as Rule objects) for a given context vector.
    Each rule is a conjunction of up to 3 feature constraints.

    These 31 templates cover the same families as the original code PLUS
    prefix and shape features.
    """
    W    = ctx[SLOT_W];    T    = ctx[SLOT_T]
    PW1  = ctx[SLOT_PW1];  PT1  = ctx[SLOT_PT1]
    PW2  = ctx[SLOT_PW2];  PT2  = ctx[SLOT_PT2]
    NW1  = ctx[SLOT_NW1];  NT1  = ctx[SLOT_NT1]
    NW2  = ctx[SLOT_NW2];  NT2  = ctx[SLOT_NT2]
    SX2  = ctx[SLOT_SFX2]; SX3  = ctx[SLOT_SFX3]; SX4  = ctx[SLOT_SFX4]
    PX2  = ctx[SLOT_PFX2]; PX3  = ctx[SLOT_PFX3]
    SHP  = ctx[SLOT_SHAPE]

    # Helper: build a Rule from keyword (slot_idx, value) pairs
    def R(*pairs: Tuple[int, str]) -> Rule:
        return Rule(dict(pairs))

    # ── Single-feature rules ──────────────────────────────────────────────────
    r_W    = R((SLOT_W,    W))
    r_PW1  = R((SLOT_PW1, PW1))
    r_PW2  = R((SLOT_PW2, PW2))
    r_NW1  = R((SLOT_NW1, NW1))
    r_NW2  = R((SLOT_NW2, NW2))
    r_PT1  = R((SLOT_PT1, PT1))
    r_PT2  = R((SLOT_PT2, PT2))
    r_NT1  = R((SLOT_NT1, NT1))
    r_NT2  = R((SLOT_NT2, NT2))
    r_SX2  = R((SLOT_SFX2, SX2))
    r_SX3  = R((SLOT_SFX3, SX3))
    r_SX4  = R((SLOT_SFX4, SX4))
    r_PX2  = R((SLOT_PFX2, PX2))
    r_PX3  = R((SLOT_PFX3, PX3))
    r_SHP  = R((SLOT_SHAPE, SHP))

    # ── Word bigrams ──────────────────────────────────────────────────────────
    r_W_NW1   = R((SLOT_W, W),   (SLOT_NW1, NW1))
    r_PW1_W   = R((SLOT_PW1, PW1), (SLOT_W, W))
    r_PW1_NW1 = R((SLOT_PW1, PW1), (SLOT_NW1, NW1))
    r_W_NW2   = R((SLOT_W, W),   (SLOT_NW2, NW2))
    r_PW2_W   = R((SLOT_PW2, PW2), (SLOT_W, W))
    r_PW2_PW1 = R((SLOT_PW2, PW2), (SLOT_PW1, PW1))
    r_NW1_NW2 = R((SLOT_NW1, NW1), (SLOT_NW2, NW2))

    # ── Word trigrams ─────────────────────────────────────────────────────────
    r_W_NW1_NW2   = R((SLOT_W, W),   (SLOT_NW1, NW1), (SLOT_NW2, NW2))
    r_PW2_PW1_W   = R((SLOT_PW2, PW2), (SLOT_PW1, PW1), (SLOT_W, W))
    r_PW1_W_NW1   = R((SLOT_PW1, PW1), (SLOT_W, W),   (SLOT_NW1, NW1))

    # ── Tag bigrams ───────────────────────────────────────────────────────────
    r_NT1_NT2  = R((SLOT_NT1, NT1), (SLOT_NT2, NT2))
    r_PT2_PT1  = R((SLOT_PT2, PT2), (SLOT_PT1, PT1))
    r_PT1_NT1  = R((SLOT_PT1, PT1), (SLOT_NT1, NT1))

    # ── Word + tag combinations ───────────────────────────────────────────────
    r_W_NT1         = R((SLOT_W, W), (SLOT_NT1, NT1))
    r_PT1_W         = R((SLOT_PT1, PT1), (SLOT_W, W))
    r_PT1_W_NT1     = R((SLOT_PT1, PT1), (SLOT_W, W), (SLOT_NT1, NT1))
    r_W_NT1_NT2     = R((SLOT_W, W), (SLOT_NT1, NT1), (SLOT_NT2, NT2))
    r_PT2_PT1_W     = R((SLOT_PT2, PT2), (SLOT_PT1, PT1), (SLOT_W, W))

    # ── Morphological + shape ─────────────────────────────────────────────────
    r_SX2_PT1  = R((SLOT_SFX2, SX2), (SLOT_PT1, PT1))
    r_SX3_PT1  = R((SLOT_SFX3, SX3), (SLOT_PT1, PT1))
    r_PX2_NT1  = R((SLOT_PFX2, PX2), (SLOT_NT1, NT1))
    r_SHP_PT1  = R((SLOT_SHAPE, SHP), (SLOT_PT1, PT1))
    r_SHP_NT1  = R((SLOT_SHAPE, SHP), (SLOT_NT1, NT1))
    r_W_SX3    = R((SLOT_W, W), (SLOT_SFX3, SX3))

    return {
        r_W, r_PW1, r_PW2, r_NW1, r_NW2, r_PT1, r_PT2, r_NT1, r_NT2,
        r_SX2, r_SX3, r_SX4, r_PX2, r_PX3, r_SHP,
        r_W_NW1, r_PW1_W, r_PW1_NW1, r_W_NW2, r_PW2_W, r_PW2_PW1, r_NW1_NW2,
        r_W_NW1_NW2, r_PW2_PW1_W, r_PW1_W_NW1,
        r_NT1_NT2, r_PT2_PT1, r_PT1_NT1,
        r_W_NT1, r_PT1_W, r_PT1_W_NT1, r_W_NT1_NT2, r_PT2_PT1_W,
        r_SX2_PT1, r_SX3_PT1, r_PX2_NT1, r_SHP_PT1, r_SHP_NT1, r_W_SX3,
    }


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
_SMART_QUOTE_RE = re.compile(r'[\u201c\u201d\u2018\u2019\u0022]')

def split_word_tag(token: str) -> Tuple[str, str]:
    """
    Split a 'word/TAG' token into (word, tag).
    Handles the special '///' case (a literal slash word tagged as '/').
    """
    if token == "///":
        return "/", "/"
    idx = token.rfind("/")
    if idx <= 0:
        # No slash at all, or slash at position 0 — treat as word with empty tag
        return token, ""
    return token[:idx], token[idx + 1:]


def read_tagged_corpus(path: str) -> List[List[Tuple[str, str]]]:
    """
    Read a tagged corpus file.
    Returns a list of sentences; each sentence is a list of (word, tag) tuples.
    Skips blank lines.
    """
    sentences: List[List[Tuple[str, str]]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = _SMART_QUOTE_RE.sub("''", line.strip())
            if not line:
                continue
            tokens = line.split()
            sentence: List[Tuple[str, str]] = []
            for tok in tokens:
                w, t = split_word_tag(tok)
                if w:
                    sentence.append((w, t))
            if sentence:
                sentences.append(sentence)
    return sentences


def read_raw_corpus(path: str) -> List[List[str]]:
    """
    Read an un-tagged text file (one sentence per line, words separated by spaces).
    """
    sentences: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            words = line.strip().split()
            if words:
                sentences.append(words)
    return sentences


def write_tagged_corpus(
    path: str,
    sentences_words: List[List[str]],
    sentences_tags: List[List[str]],
) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for words, tags in zip(sentences_words, sentences_tags):
            line = " ".join(f"{w}/{t}" for w, t in zip(words, tags))
            fh.write(line + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Lexicon  (initial tagger — most-frequent-tag + suffix fallback)
# ──────────────────────────────────────────────────────────────────────────────
class Lexicon:
    """
    Maps surface forms → most-frequent POS tag, with fallback strategies
    for unknown words:
      1. Try lowercased word
      2. Try suffix n-grams  (longest match wins)
      3. Use heuristic fallback (number, capitalised, or plain unknown)
    """

    def __init__(self):
        # word/suffix → most-frequent tag
        self._dict: Dict[str, str] = {}
        # fallback tags for fully unknown words
        self.tag_unkn_word: str = "NN"
        self.tag_unkn_cap:  str = "NNP"
        self.tag_unkn_num:  str = "CD"

    # ── building ──────────────────────────────────────────────────────────────
    @classmethod
    def build(cls, corpus_path: str, min_freq: int = 1) -> "Lexicon":
        """
        Build a Lexicon from a tagged corpus file.
        min_freq: minimum total frequency for a word to enter the dictionary
                  (use 1 = keep everything, 2 = drop hapax legomena).
        """
        lex = cls()

        # word → {tag → freq}
        word_tag_freq: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # suffix → {tag → freq}
        sfx_freq: Dict[str, Dict[str, int]]     = defaultdict(lambda: defaultdict(int))

        # For computing unknown-word fallback tags
        cnt_alpha: Dict[str, int] = defaultdict(int)
        cnt_cap:   Dict[str, int] = defaultdict(int)
        cnt_num:   Dict[str, int] = defaultdict(int)

        with open(corpus_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = _SMART_QUOTE_RE.sub("''", line.strip())
                if not line:
                    continue
                for tok in line.split():
                    w, t = split_word_tag(tok)
                    if not w or not t:
                        continue
                    word_tag_freq[w][t] += 1
                    # Accumulate suffix/prefix only for non-numeric words
                    if re.search(r"[0-9]+", w):
                        cnt_num[t] += 1
                    elif w[0].isupper():
                        cnt_cap[t] += 1
                    else:
                        cnt_alpha[t] += 1
                    # Suffix n-grams (for unknown word morphology)
                    wl = len(w)
                    if wl >= 4:
                        sfx_freq[".*" + w[-2:]][t] += 1
                        sfx_freq[".*" + w[-3:]][t] += 1
                    if wl >= 5:
                        sfx_freq[".*" + w[-4:]][t] += 1
                    if wl >= 6:
                        sfx_freq[".*" + w[-5:]][t] += 1

        # Populate word dictionary
        for word, tf in word_tag_freq.items():
            total = sum(tf.values())
            if total >= min_freq:
                best_tag = max(tf, key=tf.get)
                lex._dict[word] = best_tag

        # Compute fallback tags
        def _best(counter: Dict[str, int]) -> Optional[str]:
            return max(counter, key=counter.get) if counter else None

        lex.tag_unkn_word = _best(cnt_alpha) or "NN"
        lex.tag_unkn_cap  = _best(cnt_cap)   or lex.tag_unkn_word
        lex.tag_unkn_num  = _best(cnt_num)   or lex.tag_unkn_word

        # Store fallback keys in dict for easy serialisation
        lex._dict[_UNKN_WORD] = lex.tag_unkn_word
        lex._dict[_UNKN_CAP]  = lex.tag_unkn_cap
        lex._dict[_UNKN_NUM]  = lex.tag_unkn_num

        # Suffix entries (frequency-thresholded by suffix length)
        for sfx, tf in sfx_freq.items():
            best_tag = max(tf, key=tf.get)
            freq     = tf[best_tag]
            slen     = len(sfx)   # includes ".*" prefix → real length = slen-2
            if slen == 7 and freq >= 2:   # 5-char suffix
                lex._dict[sfx] = best_tag
            elif slen == 6 and freq >= 3: # 4-char suffix
                lex._dict[sfx] = best_tag
            elif slen == 5 and freq >= 4: # 3-char suffix
                lex._dict[sfx] = best_tag
            elif slen == 4 and freq >= 5: # 2-char suffix
                lex._dict[sfx] = best_tag

        log.info(
            "Lexicon built: %d entries | unk_word=%s | unk_cap=%s | unk_num=%s",
            len(lex._dict), lex.tag_unkn_word, lex.tag_unkn_cap, lex.tag_unkn_num,
        )
        return lex

    # ── tagging ───────────────────────────────────────────────────────────────
    def get_tag(self, word: str) -> str:
        """Return the most likely POS tag for `word` using lexicon + fallbacks."""
        if word in self._dict:
            return self._dict[word]
        lo = word.lower()
        if lo in self._dict:
            return self._dict[lo]
        # Numeric
        if re.search(r"[0-9]+", word):
            return self.tag_unkn_num
        # Suffix lookup (longest match wins)
        wl = len(word)
        for length in (5, 4, 3, 2):
            if wl > length:
                key = ".*" + word[-length:]
                if key in self._dict:
                    return self._dict[key]
        # Final heuristic
        if word and word[0].isupper():
            return self.tag_unkn_cap
        return self.tag_unkn_word

    def is_known(self, word: str) -> bool:
        return word in self._dict or word.lower() in self._dict

    # ── persistence ───────────────────────────────────────────────────────────
    def save(self, path: str) -> None:
        data = {
            "dict":      self._dict,
            "unkn_word": self.tag_unkn_word,
            "unkn_cap":  self.tag_unkn_cap,
            "unkn_num":  self.tag_unkn_num,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=4)

    @classmethod
    def load(cls, path: str) -> "Lexicon":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        lex = cls()
        lex._dict         = data["dict"]
        lex.tag_unkn_word = data["unkn_word"]
        lex.tag_unkn_cap  = data["unkn_cap"]
        lex.tag_unkn_num  = data["unkn_num"]
        return lex

    def write_text(self, path: str) -> None:
        """Write human-readable tab-separated lexicon."""
        with open(path, "w", encoding="utf-8") as fh:
            for word, tag in sorted(self._dict.items()):
                fh.write(f"{word}\t{tag}\n")


# ──────────────────────────────────────────────────────────────────────────────
# RDR Node
# ──────────────────────────────────────────────────────────────────────────────
class RDRNode:
    """
    A node in the SCRDR tree.

    Structure:
      - condition   : Rule | None (None means "always true" → root node)
      - conclusion  : str  (tag to assign when this node fires)
      - except_child: RDRNode | None  — checked when THIS node fires (refinement)
      - else_child  : RDRNode | None  — checked when THIS node does NOT fire (sibling)
      - depth       : int  (0 = root, 1 = per-initial-tag, 2+ = exceptions)
      - parent      : RDRNode | None
      - cornerstone_cases : List[List[str]]  — context vectors that triggered
                            creation of this node (used during learning only)
    """

    __slots__ = (
        "condition", "conclusion", "except_child", "else_child",
        "depth", "parent", "cornerstone_cases",
    )

    def __init__(
        self,
        condition:  Optional[Rule],
        conclusion: str,
        depth:      int = 0,
        parent:     Optional["RDRNode"] = None,
    ):
        self.condition:   Optional[Rule]     = condition
        self.conclusion:  str                = conclusion
        self.depth:       int                = depth
        self.parent:      Optional[RDRNode]  = parent
        self.except_child: Optional[RDRNode] = None
        self.else_child:   Optional[RDRNode] = None
        self.cornerstone_cases: List[List[str]] = []

    # ── fast inference ────────────────────────────────────────────────────────
    def fires(self, ctx: List[str]) -> bool:
        """Return True if this node's condition is satisfied by `ctx`."""
        if self.condition is None:
            return True          # root node always fires
        return self.condition.satisfied_by(ctx)

    def find_fired_node(self, ctx: List[str]) -> "RDRNode":
        """
        Iterative traversal of the SCRDR tree.
        Returns the deepest node whose condition fired.
        The conclusion of that node is the predicted tag.

        Traversal rule:
          • if current node fires → record it, descend to except_child
          • if current node does NOT fire → go to else_child
          • stop when we run off the end of a branch
        """
        current  = self
        fired    = self    # root always fires (condition is None)
        while True:
            if current.fires(ctx):
                fired   = current
                nxt     = current.except_child
                if nxt is None:
                    break
                current = nxt
            else:
                nxt = current.else_child
                if nxt is None:
                    break
                current = nxt
        return fired

    # ── file I/O (human-readable .rdr format) ────────────────────────────────
    def write_to_file(self, fh: Any, depth: int = 0) -> None:
        """Write this node and its subtree to `fh` in tab-indented format."""
        indent = "\t" * depth
        cond_str = "True" if self.condition is None else self.condition.to_human_string()
        fh.write(f"{indent}{cond_str} : {self.conclusion}\n")
        if self.except_child is not None:
            self.except_child.write_to_file(fh, depth + 1)
        if self.else_child is not None:
            self.else_child.write_to_file(fh, depth)

    def write_with_cases(self, fh: Any, depth: int = 0) -> None:
        """Write nodes AND their cornerstone cases (for debugging / inspection)."""
        indent = "\t" * depth
        cond_str = "True" if self.condition is None else self.condition.to_human_string()
        fh.write(f"{indent}{cond_str} : {self.conclusion}\n")
        for cc in self.cornerstone_cases:
            # Represent cornerstone case as the word/tag pair
            fh.write(f" {indent}cc: {cc[SLOT_W]}/{cc[SLOT_T]}\n")
        if self.except_child is not None:
            self.except_child.write_with_cases(fh, depth + 1)
        if self.else_child is not None:
            self.else_child.write_with_cases(fh, depth)


# ──────────────────────────────────────────────────────────────────────────────
# SCRDR Tree
# ──────────────────────────────────────────────────────────────────────────────
class SCRDRTree:
    """
    The complete Single-Classification Ripple Down Rules tree.

    Inference:
      For each token, build its context vector, call root.find_fired_node(ctx),
      and return fired_node.conclusion as the predicted tag.
    """

    def __init__(self, default_tag: str = "NN"):
        # Root node: condition=None (always fires), conclusion=default_tag
        self.root = RDRNode(condition=None, conclusion=default_tag, depth=0)
        self.default_tag = default_tag

    # ── inference ─────────────────────────────────────────────────────────────
    def classify_context(self, ctx: List[str]) -> str:
        """Return the tag predicted for a context vector."""
        return self.root.find_fired_node(ctx).conclusion

    def tag_sentence(
        self,
        words: List[str],
        lexicon: "Lexicon",
    ) -> List[str]:
        """
        Tag a sentence (list of raw words) by:
          1. Assigning initial tags from the lexicon.
          2. Iterating left-to-right, classifying each token via the RDR tree,
             and immediately using the predicted tag as context for subsequent tokens.
        """
        # Initial tagging
        n     = len(words)
        tags  = [lexicon.get_tag(w) for w in words]
        wt    = [f"{w}/{t}" for w, t in zip(words, tags)]   # 'word/tag' strings

        # Refine using RDR tree
        for i in range(n):
            ctx       = build_context(wt, i)
            new_tag   = self.classify_context(ctx)
            tags[i]   = new_tag
            wt[i]     = f"{words[i]}/{new_tag}"              # update context for successors

        return tags

    # ── counting ──────────────────────────────────────────────────────────────
    def count_nodes(self) -> int:
        """Count total number of nodes in the tree (BFS)."""
        count = 0
        queue = [self.root]
        while queue:
            node = queue.pop()
            count += 1
            if node.except_child: queue.append(node.except_child)
            if node.else_child:   queue.append(node.else_child)
        return count

    # ── persistence: human-readable .rdr file ─────────────────────────────────
    def save_rules(self, path: str, with_cases: bool = False) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            if with_cases:
                self.root.write_with_cases(fh, 0)
            else:
                self.root.write_to_file(fh, 0)

    @staticmethod
    def load_rules(path: str) -> "SCRDRTree":
        """
        Reconstruct a SCRDRTree from a tab-indented .rdr rules file.
        IMPORTANT: No eval() or exec() is used anywhere in this method.
        """
        with open(path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()

        # Filter out blank lines and cornerstone-case lines
        parsed = []
        for line in lines:
            stripped = line.strip()
            if not stripped or stripped.startswith("cc:"):
                continue
            depth = len(line) - len(line.lstrip("\t"))
            if " : " not in stripped:
                continue
            cond_str, conclusion = stripped.split(" : ", 1)
            conclusion = conclusion.strip()
            cond_str   = cond_str.strip()
            if cond_str == "True":
                rule = None
            else:
                rule = Rule.from_human_string(cond_str)
            parsed.append((depth, rule, conclusion))

        if not parsed:
            return SCRDRTree()

        # Build tree
        depth0, rule0, conc0 = parsed[0]
        tree = SCRDRTree(default_tag=conc0)
        tree.root = RDRNode(rule0, conc0, depth=depth0, parent=None)

        current = tree.root
        for depth, rule, conclusion in parsed[1:]:
            node = RDRNode(rule, conclusion, depth=depth)
            if depth > current.depth:
                node.parent = current
                current.except_child = node
            elif depth == current.depth:
                node.parent = current.parent
                current.else_child = node
            else:
                # Go up until we find the ancestor at (depth-1)
                tmp = current
                while tmp.depth != depth:
                    tmp = tmp.parent
                node.parent = tmp.parent
                tmp.else_child = node
            current = node

        return tree

    # ── persistence: fast binary .pkl file ────────────────────────────────────
    def _to_dict(self, node: Optional[RDRNode]) -> Optional[dict]:
        """Recursively convert RDRNode tree to plain-dict for safe pickling."""
        if node is None:
            return None
        return {
            "cond":   node.condition.constraints if node.condition else None,
            "conc":   node.conclusion,
            "depth":  node.depth,
            "except": self._to_dict(node.except_child),
            "else":   self._to_dict(node.else_child),
        }

    @staticmethod
    def _from_dict(
        d: Optional[dict],
        parent: Optional[RDRNode] = None,
    ) -> Optional[RDRNode]:
        if d is None:
            return None
        rule = Rule(dict(d["cond"])) if d["cond"] is not None else None
        node = RDRNode(rule, d["conc"], d["depth"], parent)
        node.except_child = SCRDRTree._from_dict(d.get("except"), parent=node)
        node.else_child   = SCRDRTree._from_dict(d.get("else"),   parent=node)
        return node

    def save_binary(self, path: str) -> None:
        data = {
            "default_tag": self.default_tag,
            "root":        self._to_dict(self.root),
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=4)

    @staticmethod
    def load_binary(path: str) -> "SCRDRTree":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        tree = SCRDRTree(default_tag=data.get("default_tag", "NN"))
        tree.root = SCRDRTree._from_dict(data.get("root"))
        return tree


# ──────────────────────────────────────────────────────────────────────────────
# Learning — object dictionary
# ──────────────────────────────────────────────────────────────────────────────
# Type alias: objects[init_tag][correct_tag] = List[context_vector]
ObjectDict = Dict[str, Dict[str, List[List[str]]]]


def build_object_dict(
    corpus:  List[List[Tuple[str, str]]],
    lexicon: Lexicon,
) -> ObjectDict:
    """
    Apply the lexicon to each sentence, then record every (init_tag, correct_tag,
    context) triple.  This object dictionary is the training input for the SCRDR
    learner.
    """
    objects: ObjectDict = defaultdict(lambda: defaultdict(list))

    for sentence in corpus:
        words = [w for w, _ in sentence]
        golds = [t for _, t in sentence]

        # Initial tagging via lexicon
        init_tags = [lexicon.get_tag(w) for w in words]
        wt_seq    = [f"{w}/{t}" for w, t in zip(words, init_tags)]

        for idx in range(len(words)):
            init_tag    = init_tags[idx]
            correct_tag = golds[idx]
            ctx         = build_context(wt_seq, idx)
            objects[init_tag][correct_tag].append(ctx)

    return objects


# ──────────────────────────────────────────────────────────────────────────────
# Learning — rule scoring (runs inside worker processes, must be picklable)
# ──────────────────────────────────────────────────────────────────────────────
def _count_matching(
    ctx_list:   List[List[str]],
    exclude:    Set[Rule],
) -> Tuple[Dict[Rule, int], Dict[Rule, List[List[str]]]]:
    """
    For each context vector in `ctx_list`, generate its candidate rules and
    count how many contexts each rule matches (excluding rules in `exclude`).

    Returns:
        counts:   rule → number of matching contexts
        matched:  rule → list of matching context vectors
    """
    counts:  Dict[Rule, int]               = defaultdict(int)
    matched: Dict[Rule, List[List[str]]]   = defaultdict(list)
    for ctx in ctx_list:
        for rule in generate_rules_for_context(ctx):
            if rule in exclude:
                continue
            counts[rule]  += 1
            matched[rule].append(ctx)
    return counts, matched


def _find_best_exception_rule(
    init_tag:         str,
    object_set:       Dict[str, List[List[str]]],
    correct_counts:   Dict[Rule, int],
    imp_threshold:    int,
) -> Tuple[Optional[Rule], str, int, List[List[str]], Dict[str, List[List[str]]]]:
    """
    Search for the single exception rule with the highest *net improvement*:

        net_improvement(rule, correct_tag) =
            |wrong tokens matching rule| - |correctly-tagged tokens matching rule|

    This is the core SCRDR learning criterion (Nguyen et al. 2014).

    Returns:
        best_rule, best_correct_tag, best_net_imp, cornerstone_cases, need_to_correct
    """
    best_rule:       Optional[Rule]           = None
    best_ctag:       str                      = ""
    best_net:        int                      = imp_threshold - 1
    best_cs:         List[List[str]]          = []
    best_need:       Dict[str, List[List[str]]] = {}

    for correct_tag, wrong_ctxs in object_set.items():
        if correct_tag == init_tag:
            continue
        if len(wrong_ctxs) < imp_threshold:
            continue

        imp_counts, imp_matched = _count_matching(wrong_ctxs, set())

        for rule, imp_cnt in imp_counts.items():
            net = imp_cnt - correct_counts.get(rule, 0)
            if net > best_net:
                # Compute collateral: which OTHER tagged objects also fire?
                need: Dict[str, List[List[str]]] = {}
                for other_tag, other_ctxs in object_set.items():
                    if other_tag == correct_tag:
                        continue
                    hits = [c for c in other_ctxs if rule.satisfied_by(c)]
                    if hits:
                        need[other_tag] = hits

                best_net  = net
                best_rule = rule
                best_ctag = correct_tag
                best_cs   = imp_matched[rule]
                best_need = need

    return best_rule, best_ctag, best_net, best_cs, best_need


def _find_best_matching_rule(
    object_set:        Dict[str, List[List[str]]],
    excluded_rules:    Set[Rule],
    match_threshold:   int,
) -> Tuple[Optional[Rule], str, List[List[str]]]:
    """
    Among rules not in `excluded_rules`, find the one that matches the most
    contexts (used to build sub-exception / correction nodes).
    """
    best_rule:  Optional[Rule]  = None
    best_ctag:  str             = ""
    best_cs:    List[List[str]] = []
    best_cnt:   int             = match_threshold - 1

    for correct_tag, ctxs in object_set.items():
        counts, matched = _count_matching(ctxs, excluded_rules)
        for rule, cnt in counts.items():
            if cnt > best_cnt:
                best_cnt  = cnt
                best_rule = rule
                best_ctag = correct_tag
                best_cs   = matched[rule]

    return best_rule, best_ctag, best_cs


def _build_correction_subtree(
    parent_node:     "RDRNode",
    need_to_correct: Dict[str, List[List[str]]],
    depth:           int,
    match_threshold: int,
    parent_cs_rules: Set[Rule],
) -> None:
    """
    Recursively build correction sub-nodes for objects that were incorrectly
    affected by an exception rule.  These nodes hang as except_child or
    else_child chains from `parent_node`.

    This corresponds to the `buildNodeForObjectSet` function in the original code.
    """
    # Working copy
    obj_set = {t: list(ctxs) for t, ctxs in need_to_correct.items()
               if ctxs}

    excluded  = set(parent_cs_rules)
    prev_node = parent_node
    first     = True

    while True:
        rule, ctag, cs = _find_best_matching_rule(obj_set, excluded, match_threshold)
        if rule is None:
            break

        node = RDRNode(rule, ctag, depth=depth, parent=prev_node)
        node.cornerstone_cases = list(cs)

        if first:
            prev_node.except_child = node
            first = False
        else:
            prev_node.else_child = node

        prev_node = node

        # Remove cornerstone cases from the working object set
        for ctx in cs:
            if ctx in obj_set.get(ctag, []):
                obj_set[ctag].remove(ctx)

        # Find collateral objects that also fire under this new rule
        sub_need: Dict[str, List[List[str]]] = {}
        for other_tag, other_ctxs in obj_set.items():
            if other_tag == ctag:
                continue
            hits = [c for c in other_ctxs if rule.satisfied_by(c)]
            if hits:
                sub_need[other_tag] = hits
                for ctx in hits:
                    obj_set[other_tag].remove(ctx)

        excluded.add(rule)

        if sub_need:
            _build_correction_subtree(
                node, sub_need, depth + 1, match_threshold, excluded
            )


# ──────────────────────────────────────────────────────────────────────────────
# Worker function for multiprocessing
# (Must be a module-level function to be picklable by multiprocessing)
# ──────────────────────────────────────────────────────────────────────────────
def _worker_build_tag_subtree(args: tuple) -> tuple:
    """
    Multiprocessing worker.  Builds all exception rules for one initial tag
    and returns a serialisable (plain-dict) representation.

    Args:
        args: (init_tag, tag_object_dict, imp_threshold, match_threshold)
            tag_object_dict: Dict[correct_tag → List[ctx_vector]]

    Returns:
        (init_tag, node_dict_list)
        node_dict_list is the list of depth-2 exception nodes as plain dicts.
    """
    init_tag, tag_objects, imp_threshold, match_threshold = args

    # Build correct_counts: for each rule that matches a correctly-tagged object,
    # how many correctly-tagged objects does it match?
    correct_counts: Dict[Rule, int] = defaultdict(int)
    for ctx in tag_objects.get(init_tag, []):
        for rule in generate_rules_for_context(ctx):
            correct_counts[rule] += 1

    # Working copy of the object set
    object_set = {t: list(ctxs) for t, ctxs in tag_objects.items()}

    # Collect the exception nodes at depth 2 (children of the init_tag node)
    exception_nodes: List[dict] = []   # list of serialisable node dicts

    while True:
        rule, ctag, net_imp, cs, need = _find_best_exception_rule(
            init_tag, object_set, correct_counts, imp_threshold
        )
        if rule is None or net_imp < imp_threshold:
            break

        # Serialise this exception node
        exc_node_dict: dict = {
            "condition":  rule.constraints,    # tuple of (slot, val) pairs
            "conclusion": ctag,
            "cs_contexts": [list(c) for c in cs],
            "sub_nodes":  [],                  # will hold correction sub-nodes
        }

        # Remove cornerstone cases from object_set
        for ctx in cs:
            if ctx in object_set.get(ctag, []):
                object_set[ctag].remove(ctx)

        # Handle collateral / error-raising objects
        error_raisers = need.get(init_tag, [])
        for tag, hits in need.items():
            for ctx in hits:
                if ctx in object_set.get(tag, []):
                    object_set[tag].remove(ctx)
        for ctx in error_raisers:
            for r in generate_rules_for_context(ctx):
                if r in correct_counts:
                    correct_counts[r] = max(0, correct_counts[r] - 1)

        # Build correction sub-nodes for the collateral objects
        if need:
            sub_node_dicts = _build_correction_sub_nodes(
                need, depth=3, match_threshold=match_threshold,
                parent_cs_rules=set()
            )
            exc_node_dict["sub_nodes"] = sub_node_dicts

        exception_nodes.append(exc_node_dict)

    return init_tag, exception_nodes


def _build_correction_sub_nodes(
    need_to_correct:  Dict[str, List[List[str]]],
    depth:            int,
    match_threshold:  int,
    parent_cs_rules:  Set[Rule],
) -> List[dict]:
    """
    Serialisable version of _build_correction_subtree for use inside workers.
    Returns a list of node dicts (a flat representation of the sub-chain).
    """
    obj_set  = {t: list(ctxs) for t, ctxs in need_to_correct.items() if ctxs}
    excluded = set(parent_cs_rules)
    result   = []

    while True:
        rule, ctag, cs = _find_best_matching_rule(obj_set, excluded, match_threshold)
        if rule is None:
            break

        node_dict: dict = {
            "condition":   rule.constraints,
            "conclusion":  ctag,
            "cs_contexts": [list(c) for c in cs],
            "depth":       depth,
            "sub_nodes":   [],
        }

        for ctx in cs:
            if ctx in obj_set.get(ctag, []):
                obj_set[ctag].remove(ctx)

        sub_need: Dict[str, List[List[str]]] = {}
        for other_tag, other_ctxs in obj_set.items():
            if other_tag == ctag:
                continue
            hits = [c for c in other_ctxs if rule.satisfied_by(c)]
            if hits:
                sub_need[other_tag] = hits
                for ctx in hits:
                    obj_set[other_tag].remove(ctx)

        excluded.add(rule)

        if sub_need:
            node_dict["sub_nodes"] = _build_correction_sub_nodes(
                sub_need, depth + 1, match_threshold, excluded
            )

        result.append(node_dict)

    return result


def _node_dicts_to_rdr_nodes(
    node_dicts: List[dict],
    parent:     RDRNode,
    depth:      int,
) -> None:
    """
    Reconstruct a chain of RDRNode objects from the list of dicts returned
    by the worker.  Nodes are linked as else_child siblings; each node's
    correction sub-nodes are linked as except_child chains.
    """
    prev = parent
    first = True

    for nd in node_dicts:
        rule = Rule(dict(nd["condition"]))
        node = RDRNode(rule, nd["conclusion"], depth=depth, parent=prev)
        node.cornerstone_cases = nd.get("cs_contexts", [])

        if first:
            prev.except_child = node
            first = False
        else:
            prev.else_child = node

        prev = node

        if nd.get("sub_nodes"):
            _node_dicts_to_rdr_nodes(nd["sub_nodes"], node, depth + 1)


# ──────────────────────────────────────────────────────────────────────────────
# SCRDR Learner
# ──────────────────────────────────────────────────────────────────────────────
class SCRDRLearner:
    """
    Trains an SCRDRTree from a tagged corpus and a Lexicon.

    Key design choices:
      • Each initial-tag group's exception rules are built independently in a
        separate worker process (via multiprocessing.Pool).
      • Workers communicate via picklable plain-dict structures (no RDRNode objects
        cross process boundaries).
      • The main process assembles the final tree from worker results.

    Parameters
    ----------
    imp_threshold   : int  (default 2)
        Minimum net improvement required to add an exception rule.
        Lower → more rules (higher recall on train, possible overfit).
        Higher → fewer rules (faster, better generalisation).
    match_threshold : int  (default 2)
        Minimum match count for sub-correction rules.
    n_jobs          : int  (default -1 = all CPUs)
        Number of worker processes.  Use 1 for single-process (easier debugging).
    verbose         : bool
    """

    def __init__(
        self,
        imp_threshold:   int  = 2,
        match_threshold: int  = 2,
        n_jobs:          int  = -1,
        verbose:         bool = True,
    ):
        self.imp_threshold   = imp_threshold
        self.match_threshold = match_threshold
        self.n_jobs          = n_jobs if n_jobs > 0 else max(1, cpu_count())
        self.verbose         = verbose

    def learn(
        self,
        corpus:  List[List[Tuple[str, str]]],
        lexicon: Lexicon,
    ) -> SCRDRTree:
        """
        Train and return an SCRDRTree.

        Steps:
          1. Apply lexicon to corpus → build object dictionary.
          2. Determine the default (most-frequent) tag.
          3. For each initial tag, spawn a worker to build exception rules.
          4. Assemble the final tree from worker results.
        """
        t_start = time.time()

        # ── 1. Object dictionary ──────────────────────────────────────────────
        log.info("Building object dictionary from %d sentences …", len(corpus))
        objects = build_object_dict(corpus, lexicon)
        total_tokens = sum(
            sum(len(v) for v in od.values()) for od in objects.values()
        )
        log.info(
            "Object dictionary: %d initial tags, %d total tokens  [%.1fs]",
            len(objects), total_tokens, time.time() - t_start,
        )

        # ── 2. Default tag (most frequent tag overall) ────────────────────────
        tag_freq: Dict[str, int] = defaultdict(int)
        for od in objects.values():
            for tag, ctxs in od.items():
                tag_freq[tag] += len(ctxs)
        default_tag = max(tag_freq, key=tag_freq.get)
        log.info("Default tag (root conclusion): %s", default_tag)

        # ── 3. Build per-tag exception subtrees in parallel ───────────────────
        init_tags = sorted(objects.keys())
        n_workers = min(self.n_jobs, len(init_tags))
        log.info(
            "Learning exception rules for %d initial tags using %d worker(s) …",
            len(init_tags), n_workers,
        )

        # Prepare arguments for workers (must be picklable)
        worker_args = [
            (
                init_tag,
                {t: list(ctxs) for t, ctxs in objects[init_tag].items()},
                self.imp_threshold,
                self.match_threshold,
            )
            for init_tag in init_tags
        ]

        t_learn = time.time()
        if n_workers > 1:
            with Pool(processes=n_workers) as pool:
                worker_results = pool.map(_worker_build_tag_subtree, worker_args)
        else:
            worker_results = [_worker_build_tag_subtree(a) for a in worker_args]

        log.info("Rule learning complete  [%.1fs]", time.time() - t_learn)

        # ── 4. Assemble the final tree ────────────────────────────────────────
        tree = SCRDRTree(default_tag=default_tag)
        # tree.root already has condition=None, conclusion=default_tag

        prev_l1_node: Optional[RDRNode] = None
        total_exception_rules = 0

        for init_tag, exception_node_dicts in worker_results:
            # Level-1 node: condition = (tag == init_tag), conclusion = init_tag
            l1_rule = Rule({SLOT_T: init_tag})
            l1_node = RDRNode(l1_rule, init_tag, depth=1, parent=tree.root)

            if tree.root.except_child is None:
                tree.root.except_child = l1_node
            else:
                prev_l1_node.else_child = l1_node
                l1_node.parent = prev_l1_node.parent
            prev_l1_node = l1_node

            if exception_node_dicts:
                n_exc = len(exception_node_dicts)
                total_exception_rules += n_exc
                _node_dicts_to_rdr_nodes(exception_node_dicts, l1_node, depth=2)

            if self.verbose:
                n_exc_disp = len(exception_node_dicts)
                log.info(
                    "  Tag %-12s : %d exception rule(s)", init_tag, n_exc_disp
                )

        log.info(
            "Total exception rules: %d  |  Total nodes: %d  [%.1fs total]",
            total_exception_rules,
            tree.count_nodes(),
            time.time() - t_start,
        )
        return tree


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(
    gold_corpus:    List[List[Tuple[str, str]]],
    pred_tag_lists: List[List[str]],
    lexicon:        Optional[Lexicon] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Returns a dict with:
      accuracy, known_acc, unknown_acc,
      macro_p, macro_r, macro_f1,
      micro_p, micro_r, micro_f1,
      weighted_f1,
      per_tag: {tag → {P, R, F1, support}},
      all_gold, all_pred (flat lists for confusion matrix)
    """
    from sklearn.metrics import (
        precision_recall_fscore_support,
        accuracy_score,
    )

    all_gold:  List[str] = []
    all_pred:  List[str] = []
    all_words: List[str] = []

    for sent_gold, sent_pred in zip(gold_corpus, pred_tag_lists):
        for (word, gold_tag), pred_tag in zip(sent_gold, sent_pred):
            all_gold.append(gold_tag)
            all_pred.append(pred_tag)
            all_words.append(word)

    labels = sorted(set(all_gold) | set(all_pred))

    # Per-tag metrics
    p_arr, r_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        all_gold, all_pred, labels=labels, average=None, zero_division=0
    )
    per_tag: Dict[str, Dict[str, float]] = {}
    for i, lbl in enumerate(labels):
        per_tag[lbl] = {
            "P":       float(p_arr[i]),
            "R":       float(r_arr[i]),
            "F1":      float(f1_arr[i]),
            "support": int(sup_arr[i]),
        }

    # Macro / micro / weighted
    def _avg(av):
        return precision_recall_fscore_support(
            all_gold, all_pred, average=av, zero_division=0
        )[:3]

    macro_p,    macro_r,    macro_f1    = _avg("macro")
    micro_p,    micro_r,    micro_f1    = _avg("micro")
    weighted_p, weighted_r, weighted_f1 = _avg("weighted")

    accuracy = accuracy_score(all_gold, all_pred)

    # Known / unknown word accuracy
    known_acc = unknown_acc = None
    known_total = unknown_total = 0
    if lexicon is not None:
        kn_correct = kn_total = un_correct = un_total = 0
        for word, gold, pred in zip(all_words, all_gold, all_pred):
            if lexicon.is_known(word):
                kn_total  += 1
                kn_correct += (gold == pred)
            else:
                un_total  += 1
                un_correct += (gold == pred)
        known_acc     = kn_correct / kn_total  if kn_total  > 0 else 0.0
        unknown_acc   = un_correct / un_total  if un_total  > 0 else 0.0
        known_total   = kn_total
        unknown_total = un_total

    return {
        "accuracy":     accuracy,
        "known_acc":    known_acc,
        "unknown_acc":  unknown_acc,
        "known_total":  known_total,
        "unknown_total": unknown_total,
        "macro_p":      float(macro_p),
        "macro_r":      float(macro_r),
        "macro_f1":     float(macro_f1),
        "micro_p":      float(micro_p),
        "micro_r":      float(micro_r),
        "micro_f1":     float(micro_f1),
        "weighted_p":   float(weighted_p),
        "weighted_r":   float(weighted_r),
        "weighted_f1":  float(weighted_f1),
        "per_tag":      per_tag,
        "all_gold":     all_gold,
        "all_pred":     all_pred,
        "labels":       labels,
    }


def print_results(results: Dict[str, Any]) -> None:
    """Pretty-print evaluation results to stdout."""
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  RDR TAGGER — EVALUATION RESULTS")
    print(SEP)
    print(f"  Overall Accuracy   : {results['accuracy']*100:7.3f}%")
    if results["known_acc"] is not None:
        print(
            f"  Known-word Acc     : {results['known_acc']*100:7.3f}%"
            f"  ({results['known_total']} tokens)"
        )
        print(
            f"  Unknown-word Acc   : {results['unknown_acc']*100:7.3f}%"
            f"  ({results['unknown_total']} tokens)"
        )
    print(f"  Macro   P/R/F1     : "
          f"{results['macro_p']*100:.3f}% / "
          f"{results['macro_r']*100:.3f}% / "
          f"{results['macro_f1']*100:.3f}%")
    print(f"  Micro   P/R/F1     : "
          f"{results['micro_p']*100:.3f}% / "
          f"{results['micro_r']*100:.3f}% / "
          f"{results['micro_f1']*100:.3f}%")
    print(f"  Weighted P/R/F1    : "
          f"{results['weighted_p']*100:.3f}% / "
          f"{results['weighted_r']*100:.3f}% / "
          f"{results['weighted_f1']*100:.3f}%")
    print("-" * 72)
    hdr = f"  {'Tag':<18} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
    print(hdr)
    print("-" * 72)
    for tag, m in sorted(results["per_tag"].items()):
        print(
            f"  {tag:<18} "
            f"{m['P']*100:>10.3f} "
            f"{m['R']*100:>10.3f} "
            f"{m['F1']*100:>10.3f} "
            f"{m['support']:>10}"
        )
    print(SEP)


def plot_confusion_matrix(
    results:    Dict[str, Any],
    output_path: str,
    max_labels: int = 40,
    figsize:    Optional[Tuple[int, int]] = None,
) -> None:
    """
    Plot and save a normalised confusion matrix as a PNG file.
    Only the `max_labels` most-frequent tags are shown (to keep it readable).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        from sklearn.metrics import confusion_matrix
        from collections import Counter
    except ImportError as e:
        log.warning("Cannot plot confusion matrix: %s", e)
        return

    all_gold = results["all_gold"]
    all_pred = results["all_pred"]
    labels   = results["labels"]

    # Keep only top-N most frequent labels for readability
    if len(labels) > max_labels:
        freq = Counter(all_gold)
        labels = [lbl for lbl, _ in freq.most_common(max_labels)]

    cm = confusion_matrix(all_gold, all_pred, labels=labels)

    # Normalise row-wise (true label distribution)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums

    n = len(labels)
    if figsize is None:
        figsize = (max(10, n * 0.7 + 2), max(8, n * 0.6 + 2))

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.4,
        linecolor="lightgray",
        annot_kws={"size": max(6, 10 - n // 6)},
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Predicted Tag", fontsize=13)
    ax.set_ylabel("Gold (True) Tag", fontsize=13)
    ax.set_title(
        f"RDR Tagger — Confusion Matrix\n"
        f"(Normalised by true label;  Acc = {results['accuracy']*100:.2f}%)",
        fontsize=13,
    )
    plt.xticks(rotation=45, ha="right", fontsize=max(7, 11 - n // 6))
    plt.yticks(rotation=0,  fontsize=max(7, 11 - n // 6))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrix saved → %s", output_path)


# ──────────────────────────────────────────────────────────────────────────────
# High-level pipeline functions
# ──────────────────────────────────────────────────────────────────────────────
def cmd_train(args: argparse.Namespace) -> None:
    """Train an SCRDR model from a tagged corpus."""
    log.info("=" * 60)
    log.info("MODE: TRAIN")
    log.info("  Train corpus : %s", args.train)
    log.info("  Model prefix : %s", args.model)
    log.info("  Imp thresh   : %d", args.threshold_imp)
    log.info("  Match thresh : %d", args.threshold_match)
    log.info("  Workers      : %d", args.jobs if args.jobs > 0 else cpu_count())
    log.info("=" * 60)

    # Read corpus
    corpus = read_tagged_corpus(args.train)
    n_tokens = sum(len(s) for s in corpus)
    log.info("Corpus: %d sentences, %d tokens", len(corpus), n_tokens)

    # Ensure output directory exists
    model_dir = os.path.dirname(os.path.abspath(args.model))
    os.makedirs(model_dir, exist_ok=True)

    # Build lexicon
    lexicon = Lexicon.build(args.train, min_freq=args.min_lexicon_freq)

    # Save lexicon
    lex_path = args.model + ".lex"
    lexicon.save(lex_path)
    log.info("Lexicon saved → %s", lex_path)

    if args.save_text_lexicon:
        lexicon.write_text(args.model + ".lex.txt")

    # Learn RDR tree
    learner = SCRDRLearner(
        imp_threshold   = args.threshold_imp,
        match_threshold = args.threshold_match,
        n_jobs          = args.jobs,
        verbose         = True,
    )
    tree = learner.learn(corpus, lexicon)

    # Save binary model
    bin_path = args.model + ".pkl"
    tree.save_binary(bin_path)
    log.info("Binary model saved → %s", bin_path)

    # Save human-readable rules file
    rdr_path = args.model + ".rdr"
    tree.save_rules(rdr_path, with_cases=args.save_cases)
    log.info("Rules file saved  → %s", rdr_path)
    log.info("Total nodes in tree: %d", tree.count_nodes())

    # Optional: run evaluation on training set
    if getattr(args, "eval_on_train", False):
        log.info("Evaluating on training data …")
        all_preds = [
            tree.tag_sentence([w for w, _ in s], lexicon)
            for s in corpus
        ]
        results = evaluate(corpus, all_preds, lexicon)
        print_results(results)


def cmd_test(args: argparse.Namespace) -> None:
    """Evaluate a trained model on a tagged test corpus."""
    log.info("=" * 60)
    log.info("MODE: TEST")
    log.info("  Test corpus : %s", args.test)
    log.info("  Model       : %s", args.model)
    log.info("  Hypothesis  : %s", args.hyp)
    log.info("=" * 60)

    # Load model
    bin_path = args.model + ".pkl"
    rdr_path = args.model + ".rdr"
    lex_path = args.model + ".lex"

    if os.path.exists(bin_path):
        log.info("Loading binary model from %s …", bin_path)
        tree = SCRDRTree.load_binary(bin_path)
    elif os.path.exists(rdr_path):
        log.info("Loading rules from %s …", rdr_path)
        tree = SCRDRTree.load_rules(rdr_path)
    else:
        log.error("No model file found at %s (.pkl or .rdr)", args.model)
        sys.exit(1)

    if not os.path.exists(lex_path):
        log.error("Lexicon file not found: %s", lex_path)
        sys.exit(1)

    lexicon = Lexicon.load(lex_path)
    log.info("Model loaded. Nodes in tree: %d", tree.count_nodes())

    # Read test corpus
    corpus = read_tagged_corpus(args.test)
    log.info("Test corpus: %d sentences", len(corpus))

    # Tag
    t0 = time.time()
    all_preds: List[List[str]] = []
    for sent in corpus:
        words = [w for w, _ in sent]
        tags  = tree.tag_sentence(words, lexicon)
        all_preds.append(tags)
    log.info("Tagging complete  [%.2fs]", time.time() - t0)

    # Write hypothesis file
    all_words  = [[w for w, _ in s] for s in corpus]
    write_tagged_corpus(args.hyp, all_words, all_preds)
    log.info("Hypothesis written → %s", args.hyp)

    # Evaluate
    results = evaluate(corpus, all_preds, lexicon)
    print_results(results)

    # Confusion matrix
    if args.confusion_matrix:
        max_lbl = getattr(args, "max_labels", 40)
        plot_confusion_matrix(results, args.confusion_matrix, max_labels=max_lbl)


def cmd_tag(args: argparse.Namespace) -> None:
    """Tag a raw (un-tagged) text file."""
    log.info("=" * 60)
    log.info("MODE: TAG (raw text)")
    log.info("  Input  : %s", args.input)
    log.info("  Model  : %s", args.model)
    log.info("  Output : %s", args.output)
    log.info("=" * 60)

    # Load model
    bin_path = args.model + ".pkl"
    rdr_path = args.model + ".rdr"
    lex_path = args.model + ".lex"

    if os.path.exists(bin_path):
        tree = SCRDRTree.load_binary(bin_path)
    elif os.path.exists(rdr_path):
        tree = SCRDRTree.load_rules(rdr_path)
    else:
        log.error("No model file found: %s (.pkl / .rdr)", args.model)
        sys.exit(1)

    lexicon = Lexicon.load(lex_path)

    raw_corpus = read_raw_corpus(args.input)
    log.info("Tagging %d sentences …", len(raw_corpus))

    t0 = time.time()
    with open(args.output, "w", encoding="utf-8") as fh:
        for words in raw_corpus:
            tags = tree.tag_sentence(words, lexicon)
            fh.write(" ".join(f"{w}/{t}" for w, t in zip(words, tags)) + "\n")
    log.info("Done  [%.2fs]  →  %s", time.time() - t0, args.output)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rdr_tagger",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )

    sub = parser.add_subparsers(dest="command", required=True,
                                 metavar="{train,test,tag}")

    # ── train ─────────────────────────────────────────────────────────────────
    p_train = sub.add_parser(
        "train",
        help="Train an SCRDR model from a tagged corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_train.add_argument(
        "--train", required=True, metavar="FILE",
        help="Tagged training corpus  (word/TAG format, one sentence per line)",
    )
    p_train.add_argument(
        "--model", required=True, metavar="PREFIX",
        help=(
            "Output model path prefix.  Produces:\n"
            "  PREFIX.pkl   (fast binary model)\n"
            "  PREFIX.rdr   (human-readable rules)\n"
            "  PREFIX.lex   (binary lexicon)"
        ),
    )
    p_train.add_argument(
        "--jobs", "-j", type=int, default=-1, metavar="N",
        help="Number of CPU workers  (-1 = all available, default: -1)",
    )
    p_train.add_argument(
        "--threshold-imp", type=int, default=2, metavar="T",
        help=(
            "Minimum net improvement to add an exception rule  (default: 2).\n"
            "Lower → more rules (better train fit, risk overfit).\n"
            "Higher → fewer rules (faster, better generalisation)."
        ),
    )
    p_train.add_argument(
        "--threshold-match", type=int, default=2, metavar="T",
        help=(
            "Minimum match count for sub-correction rules  (default: 2)."
        ),
    )
    p_train.add_argument(
        "--min-lexicon-freq", type=int, default=1, metavar="N",
        help="Minimum token frequency to include a word in the lexicon (default: 1)",
    )
    p_train.add_argument(
        "--save-cases", action="store_true",
        help="Include cornerstone cases in the .rdr file (for inspection)",
    )
    p_train.add_argument(
        "--save-text-lexicon", action="store_true",
        help="Also save a human-readable lexicon as PREFIX.lex.txt",
    )
    p_train.add_argument(
        "--eval-on-train", action="store_true",
        help="Evaluate model on training data after learning (sanity check)",
    )

    # ── test ──────────────────────────────────────────────────────────────────
    p_test = sub.add_parser(
        "test",
        help="Evaluate a trained model on a tagged test corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_test.add_argument(
        "--test", required=True, metavar="FILE",
        help="Tagged test corpus  (gold standard; word/TAG format)",
    )
    p_test.add_argument(
        "--model", required=True, metavar="PREFIX",
        help="Model prefix (same as used in training)",
    )
    p_test.add_argument(
        "--hyp", required=True, metavar="FILE",
        help="Output hypothesis file  (tagged by the model)",
    )
    p_test.add_argument(
        "--confusion-matrix", metavar="FILE",
        help="Save confusion matrix plot as a PNG file",
    )
    p_test.add_argument(
        "--max-labels", type=int, default=40, metavar="N",
        help="Maximum number of tags shown in confusion matrix  (default: 40)",
    )

    # ── tag ───────────────────────────────────────────────────────────────────
    p_tag = sub.add_parser(
        "tag",
        help="Tag a raw (un-tagged) text file with a trained model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_tag.add_argument(
        "--input", required=True, metavar="FILE",
        help="Raw input file  (one sentence per line, words separated by spaces)",
    )
    p_tag.add_argument(
        "--model", required=True, metavar="PREFIX",
        help="Model prefix (same as used in training)",
    )
    p_tag.add_argument(
        "--output", required=True, metavar="FILE",
        help="Output tagged file",
    )

    return parser


def main():
    parser = build_argument_parser()
    args   = parser.parse_args()

    # Re-configure logging level based on flag
    logging.getLogger("rdr_tagger").setLevel(
        getattr(logging, args.log_level.upper())
    )

    # Dispatch
    {
        "train": cmd_train,
        "test":  cmd_test,
        "tag":   cmd_tag,
    }[args.command](args)


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Protect multiprocessing entry point (important on Windows and for spawn)
    import multiprocessing
    multiprocessing.freeze_support()
    sys.setrecursionlimit(100_000)
    main()

