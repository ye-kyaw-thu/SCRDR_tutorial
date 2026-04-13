#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  scrdr_tokenizer.py  v3.0
  Single Classification Ripple Down Rules (SCRDR) Word Segmenter
================================================================================

Changelog v3.0  (major accuracy improvement)
─────────────────────────────────────────────
  CORE FIX — Word-Vocabulary + Greedy-Longest-Match initial tagger:
    The original RDRsegmenter (DataPreprocessor.java) builds a vocabulary of
    all training words, then initially segments raw text with greedy longest
    match before running the SCRDR learner.  Our v1/v2 code used only
    character/syllable frequency for initial tagging — producing ~50% accuracy.
    With vocabulary-based greedy-longest-match, initial accuracy on KNOWN words
    is 100%.  The SCRDR tree then only needs to correct unknown-word boundaries.

  BUG-4 FIXED — Post-inference linguistic constraint:
    During inference, Myanmar combining characters (vowel signs, medials, asat,
    misc signs) that are predicted B or S by the tree are silently corrected to
    I.  These characters can NEVER start or stand alone as a word in Myanmar.
    This eliminates artefacts like `တယ ်` (where asat `်` became S).

  BUG-1/2/3 retained from v2.0:
    BUG-1: Myanmar combining chars always get initial tag I (not most-frequent).
    BUG-2: Spaces in raw input are treated as word separators, not chars to tag.
    BUG-3: bies_to_words() no longer emits whitespace-only S tokens as words.

  Syllable mode (--syllable) is now also dramatically improved through the same
  vocabulary-based greedy-longest-match approach.

Written by Ye Kyaw Thu, Language Understanding Lab., Myanmar.
Last updated: 13 April 2026.

Reference Code:  
https://github.com/datquocnguyen/RDRsegmenter

Reference Paper:

A Fast and Accurate Vietnamese Word Segmenter (Nguyen et al., LREC 2018)
https://aclanthology.org/L18-1410/
https://arxiv.org/abs/1709.06307

Architecture
────────────
  Training:
    1. Read segmented corpus.
    2. Build WordVocabulary (maps word_string → BIES sequence).
    3. Build Lexicon (freq-based fallback for OOV chars/syllables).
    4. Build object dictionary using WordVocabulary initial tags.
    5. SCRDR learner builds exception tree in parallel workers.
    6. Save model (binary .pkl + human-readable .rdr).

  Inference:
    1. Load model + lexicon.
    2. For each raw sentence: syllabify if needed; apply greedy-longest-match
       initial tags from vocabulary; run RDR tree left-to-right with running
       tag updates; apply post-inference Myanmar constraints.
    3. Convert BIES/BI tags → word list → join with separator.

Usage
─────
  python scrdr_tokenizer.py train \\
      --train corpus.txt --model model/mySeg \\
      --jobs 16 [--syllable] [--scheme BIES]

  python scrdr_tokenizer.py test \\
      --input gold.txt --model model/mySeg \\
      --output hyp.txt [--confusion-matrix cm.png]

  python scrdr_tokenizer.py segment \\
      --input raw.txt --model model/mySeg --output out.txt [--separator " "]

Data format: space-separated words, one sentence per line.
  Myanmar: ကျွန်တော် မျက်မှန် တစ် လက် လုပ် ချင် ပါ တယ်
  Chinese: 北京 是 中国 的 首都
================================================================================
"""

import os, re, sys, time, pickle, logging, unicodedata, argparse, warnings
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Set, Tuple, Any

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────────────────────────────────────
def _setup_logging(level="INFO"):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        level=getattr(logging, level.upper()),
        datefmt="%H:%M:%S", stream=sys.stderr)
    return logging.getLogger("scrdr_tokenizer")
log = _setup_logging()

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
SCHEME_BIES = "BIES";  SCHEME_BI = "BI"
MODE_CHAR = "char";    MODE_SYLLABLE = "syllable"
TAG_B = "B"; TAG_I = "I"; TAG_E = "E"; TAG_S = "S"

CT_SPACE="SP"; CT_DIGIT="DG"; CT_LATIN_U="LU"; CT_LATIN_L="LL"
CT_MY_CON="MC"; CT_MY_VOW="MV"; CT_MY_MED="MM"; CT_MY_AST="MA"
CT_MY_SGN="MG"; CT_MY_DIG="MD"; CT_MY_PUN="MP"; CT_MY_EXT="ME"
CT_CJK="CJ"; CT_ARABIC="AR"; CT_DEVA="DV"; CT_THAI="TH"
CT_PUNC="PU"; CT_OTHER="OT"

# Myanmar combining types — NEVER word-initial, NEVER standalone word
_MY_COMBINING = frozenset([CT_MY_VOW, CT_MY_MED, CT_MY_AST, CT_MY_SGN])

def get_char_type(ch: str) -> str:
    if not ch: return CT_OTHER
    cp = ord(ch)
    if ch.isspace(): return CT_SPACE
    if 0x1000 <= cp <= 0x109F:
        if cp <= 0x1021 or cp in (0x1025,0x1026,0x1027,0x1029,0x102A,0x103F,0x104E):
            return CT_MY_CON
        if 0x102B <= cp <= 0x1035: return CT_MY_VOW
        if cp in (0x1036,0x1037,0x1038): return CT_MY_SGN
        if cp in (0x1039,0x103A): return CT_MY_AST
        if 0x103B <= cp <= 0x103E: return CT_MY_MED
        if 0x1040 <= cp <= 0x1049: return CT_MY_DIG
        if 0x104A <= cp <= 0x104F: return CT_MY_PUN
        return CT_MY_CON
    if (0xA9E0<=cp<=0xA9FF) or (0xAA60<=cp<=0xAA7F): return CT_MY_EXT
    if 0x105E<=cp<=0x1060: return CT_MY_MED
    if (0x4E00<=cp<=0x9FFF or 0x3400<=cp<=0x4DBF or 0x20000<=cp<=0x2A6DF
            or 0x3040<=cp<=0x30FF or 0xFF00<=cp<=0xFFEF
            or 0xAC00<=cp<=0xD7AF or 0x3000<=cp<=0x303F): return CT_CJK
    if 0x0E00<=cp<=0x0E7F: return CT_THAI
    if 0x0600<=cp<=0x06FF: return CT_ARABIC
    if 0x0900<=cp<=0x097F: return CT_DEVA
    if cp < 0x80:
        if ch.isdigit(): return CT_DIGIT
        if ch.isupper(): return CT_LATIN_U
        if ch.islower(): return CT_LATIN_L
        return CT_PUNC
    cat = unicodedata.category(ch)
    if cat.startswith("L"): return CT_LATIN_L
    if cat.startswith("N"): return CT_DIGIT
    if cat.startswith(("P","S")): return CT_PUNC
    if cat.startswith("Z"): return CT_SPACE
    return CT_OTHER

def can_word_init(ct: str) -> str:
    return "N" if ct in _MY_COMBINING else "Y"

# ──────────────────────────────────────────────────────────────────────────────
# Myanmar syllabifier  (sylbreak-style regex)
# ──────────────────────────────────────────────────────────────────────────────
_MY_SYL_RE = re.compile(
    r'(?:'
    r'[\u1000-\u102A\u103F\u104E]'       # base consonant / independent vowel
    r'(?:\u1039[\u1000-\u102A])*'         # stacked consonants (virama + consonant)
    r'[\u103B-\u103E]*'                   # medials
    r'[\u102B-\u1032\u1036-\u1038\u103A]*'  # vowel signs, signs, asat
    r')'
    r'|[\u1040-\u1049]+'                  # Myanmar digits (grouped)
    r'|[\u104A-\u104F]'                   # Myanmar punctuation (one each)
    r'|[\uA9E0-\uA9FF\uAA60-\uAA7F]'     # Myanmar Extended A/B
    r'|[0-9]+'                            # ASCII digits (grouped)
    r'|[a-zA-Z]+'                         # Latin letters (grouped)
    r'|[^\s]',                            # fallback: any non-space char
    re.UNICODE
)

def myanmar_syllabify(text: str) -> List[str]:
    """Break a Myanmar string into syllables (and Latin/digit groups)."""
    return [m.group() for m in _MY_SYL_RE.finditer(text) if not m.group().isspace()]

# ──────────────────────────────────────────────────────────────────────────────
# Feature slot layout  (19 slots)
# ──────────────────────────────────────────────────────────────────────────────
SL_PU2,SL_PU1=0,1; SL_CU,SL_NU1=2,3; SL_NU2=4
SL_CT,SL_PT1=5,6; SL_PT2,SL_NT1=7,8; SL_NT2=9
SL_CTP,SL_PTP1=10,11; SL_NTP1=12
SL_BGL,SL_BGR=13,14; SL_TRI=15; SL_PBGL,SL_NBGR=16,17; SL_CWI=18
NUM_SLOTS=19

_BOS_U="<BOS>"; _EOS_U="<EOS>"; _BOS_T="<BOST>"; _EOS_T="<EOST>"
_NO_BI="<NOB>"; _NO_TR="<NOT>"

SLOT_NAMES: Dict[int,str] = {
    SL_PU2:"prevUnit2", SL_PU1:"prevUnit1",
    SL_CU:"currUnit",   SL_NU1:"nextUnit1", SL_NU2:"nextUnit2",
    SL_CT:"currTag",    SL_PT1:"prevTag1",  SL_PT2:"prevTag2",
    SL_NT1:"nextTag1",  SL_NT2:"nextTag2",
    SL_CTP:"currType",  SL_PTP1:"prevType1",SL_NTP1:"nextType1",
    SL_BGL:"bigramL",   SL_BGR:"bigramR",
    SL_TRI:"trigramC",
    SL_PBGL:"prevBigram",SL_NBGR:"nextBigram",
    SL_CWI:"canWordInit",
}
NAME_TO_SLOT: Dict[str,int] = {v:k for k,v in SLOT_NAMES.items()}

def _type_of(unit:str, mode:str)->str:
    if mode==MODE_SYLLABLE:
        ch0=unit[0] if unit else ''
        ct=get_char_type(ch0)
        if ct in (CT_MY_CON,CT_MY_EXT): return "SM"
        if ct==CT_MY_PUN: return "SP"
        if ct==CT_MY_DIG: return "SD"
        if ct==CT_DIGIT:  return "LD"
        if ct in (CT_LATIN_U,CT_LATIN_L): return "LA"
        if ct==CT_CJK: return "CJ"
        return "PU"
    return get_char_type(unit[0] if unit else '')

def build_unit_context(unit_tags:List[str], idx:int, mode:str=MODE_CHAR)->List[str]:
    def _sp(tok):
        p=tok.rfind("/"); return (tok[:p],tok[p+1:]) if p>=0 else (tok,"")
    n=len(unit_tags); ctx=[_BOS_U]*NUM_SLOTS
    cu,ct=_sp(unit_tags[idx])
    ctx[SL_CU]=cu; ctx[SL_CT]=ct
    ctx[SL_CTP]=_type_of(cu,mode)
    ctx[SL_CWI]=can_word_init(get_char_type(cu[0]) if cu else CT_OTHER)
    if idx>0:
        pu1,pt1=_sp(unit_tags[idx-1])
        ctx[SL_PU1]=pu1; ctx[SL_PT1]=pt1; ctx[SL_PTP1]=_type_of(pu1,mode)
    else: ctx[SL_PU1]=_BOS_U; ctx[SL_PT1]=_BOS_T; ctx[SL_PTP1]=CT_OTHER
    if idx>1:
        pu2,pt2=_sp(unit_tags[idx-2])
        ctx[SL_PU2]=pu2; ctx[SL_PT2]=pt2
    else: ctx[SL_PU2]=_BOS_U; ctx[SL_PT2]=_BOS_T
    if idx<n-1:
        nu1,nt1=_sp(unit_tags[idx+1])
        ctx[SL_NU1]=nu1; ctx[SL_NT1]=nt1; ctx[SL_NTP1]=_type_of(nu1,mode)
    else: ctx[SL_NU1]=_EOS_U; ctx[SL_NT1]=_EOS_T; ctx[SL_NTP1]=CT_OTHER
    if idx<n-2:
        nu2,nt2=_sp(unit_tags[idx+2])
        ctx[SL_NU2]=nu2; ctx[SL_NT2]=nt2
    else: ctx[SL_NU2]=_EOS_U; ctx[SL_NT2]=_EOS_T
    ctx[SL_BGL]=ctx[SL_PU1]+ctx[SL_CU]  if ctx[SL_PU1]!=_BOS_U else _NO_BI
    ctx[SL_BGR]=ctx[SL_CU]+ctx[SL_NU1]  if ctx[SL_NU1]!=_EOS_U else _NO_BI
    ctx[SL_TRI]=(ctx[SL_PU1]+ctx[SL_CU]+ctx[SL_NU1]
                 if ctx[SL_PU1]!=_BOS_U and ctx[SL_NU1]!=_EOS_U else _NO_TR)
    ctx[SL_PBGL]=ctx[SL_PU2]+ctx[SL_PU1] if ctx[SL_PU2]!=_BOS_U else _NO_BI
    ctx[SL_NBGR]=ctx[SL_NU1]+ctx[SL_NU2] if ctx[SL_NU2]!=_EOS_U else _NO_BI
    return ctx

# ──────────────────────────────────────────────────────────────────────────────
# Rule
# ──────────────────────────────────────────────────────────────────────────────
class Rule:
    __slots__=("constraints","_hash","_str_cache")
    def __init__(self,d:Dict[int,str]):
        self.constraints=tuple(sorted(d.items())); self._hash=hash(self.constraints); self._str_cache=None
    def satisfied_by(self,ctx:List[str])->bool:
        for s,v in self.constraints:
            if ctx[s]!=v: return False
        return True
    def __hash__(self): return self._hash
    def __eq__(self,o): return self.constraints==o.constraints
    def __len__(self): return len(self.constraints)
    def to_str(self)->str:
        if self._str_cache is None:
            self._str_cache=" and ".join(f'{SLOT_NAMES[s]} == "{v}"' for s,v in self.constraints)
        return self._str_cache
    @staticmethod
    def from_str(s:str)->"Rule":
        d:Dict[int,str]={}
        for p in s.split(" and "):
            p=p.strip(); eq=p.find("==")
            if eq<0: continue
            nm=p[:eq].strip(); val=p[eq+2:].strip().strip('"')
            if nm in NAME_TO_SLOT: d[NAME_TO_SLOT[nm]]=val
        return Rule(d)

# ──────────────────────────────────────────────────────────────────────────────
# Fast raw-tuple rule generator (hot-path; no Rule object overhead)
# ──────────────────────────────────────────────────────────────────────────────
def _gen_raw(ctx:List[str])->List[tuple]:
    CU=ctx[SL_CU]; CT=ctx[SL_CT]
    PU1=ctx[SL_PU1]; PT1=ctx[SL_PT1]
    PU2=ctx[SL_PU2]; PT2=ctx[SL_PT2]
    NU1=ctx[SL_NU1]; NT1=ctx[SL_NT1]
    NU2=ctx[SL_NU2]; NT2=ctx[SL_NT2]
    CTP=ctx[SL_CTP]; PTP1=ctx[SL_PTP1]; NTP1=ctx[SL_NTP1]
    BGL=ctx[SL_BGL]; BGR=ctx[SL_BGR]; TRI=ctx[SL_TRI]
    PBGL=ctx[SL_PBGL]; NBGR=ctx[SL_NBGR]; CWI=ctx[SL_CWI]
    return [
        # single features
        ((SL_CU,CU),),((SL_PU1,PU1),),((SL_PU2,PU2),),
        ((SL_NU1,NU1),),((SL_NU2,NU2),),
        ((SL_CTP,CTP),),((SL_PTP1,PTP1),),((SL_NTP1,NTP1),),
        ((SL_PT1,PT1),),((SL_PT2,PT2),),((SL_NT1,NT1),),((SL_NT2,NT2),),
        ((SL_BGL,BGL),),((SL_BGR,BGR),),((SL_TRI,TRI),),
        ((SL_PBGL,PBGL),),((SL_NBGR,NBGR),),((SL_CWI,CWI),),
        # 2-feature (pre-sorted by slot idx)
        ((SL_PU1,PU1),(SL_CU,CU)),((SL_CU,CU),(SL_NU1,NU1)),
        ((SL_PU2,PU2),(SL_CU,CU)),((SL_CU,CU),(SL_NU2,NU2)),
        ((SL_CU,CU),(SL_PT1,PT1)),((SL_CU,CU),(SL_NT1,NT1)),
        ((SL_PU1,PU1),(SL_PT1,PT1)),((SL_NU1,NU1),(SL_NT1,NT1)),
        ((SL_PT1,PT1),(SL_NT1,NT1)),((SL_PT2,PT2),(SL_PT1,PT1)),
        ((SL_NT1,NT1),(SL_NT2,NT2)),
        ((SL_CTP,CTP),(SL_CU,CU)),((SL_CTP,CTP),(SL_NT1,NT1)),
        ((SL_PTP1,PTP1),(SL_CU,CU)),((SL_NTP1,NTP1),(SL_CU,CU)),
        ((SL_CTP,CTP),(SL_PTP1,PTP1)),((SL_CTP,CTP),(SL_NTP1,NTP1)),
        ((SL_CTP,CTP),(SL_PT1,PT1)),
        ((SL_BGL,BGL),(SL_NT1,NT1)),((SL_BGR,BGR),(SL_PT1,PT1)),
        ((SL_BGL,BGL),(SL_CTP,CTP)),((SL_BGR,BGR),(SL_CTP,CTP)),
        ((SL_CWI,CWI),(SL_CTP,CTP)),((SL_CWI,CWI),(SL_PT1,PT1)),
        ((SL_CWI,CWI),(SL_NT1,NT1)),((SL_PTP1,PTP1),(SL_CWI,CWI)),
        # 3-feature
        ((SL_PU1,PU1),(SL_CU,CU),(SL_NU1,NU1)),
        ((SL_PU2,PU2),(SL_PU1,PU1),(SL_CU,CU)),
        ((SL_CU,CU),(SL_NU1,NU1),(SL_NU2,NU2)),
        ((SL_PT2,PT2),(SL_PT1,PT1),(SL_CU,CU)),
        ((SL_CU,CU),(SL_PT1,PT1),(SL_NT1,NT1)),
        ((SL_PT1,PT1),(SL_CU,CU),(SL_NT1,NT1)),
        ((SL_PTP1,PTP1),(SL_CTP,CTP),(SL_NTP1,NTP1)),
        ((SL_CTP,CTP),(SL_PT1,PT1),(SL_NT1,NT1)),
        ((SL_TRI,TRI),(SL_PT1,PT1)),
        ((SL_CWI,CWI),(SL_CTP,CTP),(SL_PT1,PT1)),
        ((SL_CWI,CWI),(SL_PT1,PT1),(SL_NT1,NT1)),
    ]

def generate_rules_for_context(ctx:List[str])->Set[Rule]:
    return {Rule(dict(rt)) for rt in _gen_raw(ctx)}

# ──────────────────────────────────────────────────────────────────────────────
# BIES conversion helpers
# ──────────────────────────────────────────────────────────────────────────────
def _word_to_bies(word:str, mode:str=MODE_CHAR)->List[Tuple[str,str]]:
    units = myanmar_syllabify(word) if mode==MODE_SYLLABLE else list(word)
    if not units: return []
    n=len(units)
    pairs=[]
    for i,u in enumerate(units):
        if n==1:       pairs.append((u,TAG_S))
        elif i==0:     pairs.append((u,TAG_B))
        elif i==n-1:   pairs.append((u,TAG_E))
        else:          pairs.append((u,TAG_I))
    return pairs

def words_to_bies(words:List[str], mode:str=MODE_CHAR)->List[Tuple[str,str]]:
    pairs=[]
    for w in words: pairs.extend(_word_to_bies(w,mode))
    return pairs

def bies_to_words(pairs:List[Tuple[str,str]])->List[str]:
    """Convert (unit, BIES) pairs → word strings.  BUG-3: skip whitespace S tokens."""
    words=[]; buf=[]
    for unit,label in pairs:
        if label==TAG_S:
            if buf: words.append("".join(buf)); buf=[]
            if not unit.isspace(): words.append(unit)
        elif label==TAG_B:
            if buf: words.append("".join(buf))
            buf=[unit]
        elif label==TAG_I: buf.append(unit)
        elif label==TAG_E:
            buf.append(unit); words.append("".join(buf)); buf=[]
        else:
            if buf: words.append("".join(buf)); buf=[]
            if not unit.isspace(): words.append(unit)
    if buf: words.append("".join(buf))
    return words

def tags_to_words(units:List[str],tags:List[str],scheme:str=SCHEME_BIES)->List[str]:
    return bies_to_words(list(zip(units,tags)))

# ──────────────────────────────────────────────────────────────────────────────
# WordVocabulary — the KEY improvement in v3
# ──────────────────────────────────────────────────────────────────────────────
class WordVocabulary:
    """
    Stores all training words as char/syllable sequences with their BIES labels.
    Implements greedy-longest-match initial tagging (exactly as RDRsegmenter's
    DataPreprocessor.java).

    Why this dramatically improves accuracy:
    • Known words (the vast majority of training tokens) → perfect initial BIES tags
    • Object dictionary: init_tag == gold_tag for known words → ZERO entries → fewer rules needed
    • Tree only corrects unknown word boundaries
    • Initial accuracy: ~50% (frequency-based) → ~90%+ (vocabulary-based)
    """

    def __init__(self, mode:str=MODE_CHAR):
        self.mode    = mode
        # key = word string (char mode) or '|'-joined syllables (syllable mode)
        # value = list of BIES labels for each unit
        self._vocab: Dict[str, List[str]] = {}
        self.max_units: int = 0   # max word length in units

    @classmethod
    def build(cls, corpus:List[List[str]], mode:str=MODE_CHAR)->"WordVocabulary":
        voc = cls(mode)
        for words in corpus:
            for word in words:
                if not word or word.isspace(): continue
                units = myanmar_syllabify(word) if mode==MODE_SYLLABLE else list(word)
                if not units: continue
                n = len(units)
                key = "|".join(units) if mode==MODE_SYLLABLE else word
                if key not in voc._vocab:
                    bies=[]
                    for i,_ in enumerate(units):
                        if n==1:       bies.append(TAG_S)
                        elif i==0:     bies.append(TAG_B)
                        elif i==n-1:   bies.append(TAG_E)
                        else:          bies.append(TAG_I)
                    voc._vocab[key]=bies
                    voc.max_units = max(voc.max_units, n)
        log.info("WordVocabulary: %d word types | max_units=%d | mode=%s",
                 len(voc._vocab), voc.max_units, mode)
        return voc

    def tag_units(self, units:List[str], fallback_lexicon=None)->List[str]:
        """
        Apply greedy longest-match to assign initial BIES tags.
        Falls back to:
          - Myanmar combining chars → I  (BUG-1 constraint)
          - Myanmar punctuation    → S
          - Unknown chars          → lexicon.get_tag() or S
        Returns list of BIES tags (same length as `units`).
        """
        n=len(units); tags=[None]*n; i=0
        while i<n:
            matched=False
            for length in range(min(self.max_units, n-i), 0, -1):
                chunk=units[i:i+length]
                key="|".join(chunk) if self.mode==MODE_SYLLABLE else "".join(chunk)
                if key in self._vocab:
                    seq=self._vocab[key]
                    for j,tag in enumerate(seq): tags[i+j]=tag
                    i+=length; matched=True; break
            if not matched:
                # Unknown unit: use linguistic / lexicon fallback
                unit=units[i]; ch0=unit[0] if unit else ''
                ct=get_char_type(ch0)
                if ct in _MY_COMBINING:
                    # BUG-1 fix: Myanmar combining chars are NEVER word-initial
                    tags[i]=TAG_I
                elif ct==CT_MY_PUN:
                    tags[i]=TAG_S
                elif fallback_lexicon is not None:
                    tags[i]=fallback_lexicon.get_tag(unit)
                else:
                    # Safe default: treat unknown as standalone word
                    tags[i]=TAG_S
                i+=1
        return tags

    def save(self, path:str)->None:
        with open(path,"wb") as fh:
            pickle.dump({"vocab":self._vocab,"max_units":self.max_units,"mode":self.mode},fh,protocol=4)

    @classmethod
    def load(cls, path:str)->"WordVocabulary":
        with open(path,"rb") as fh:
            d=pickle.load(fh)
        voc=cls(d.get("mode",MODE_CHAR))
        voc._vocab=d["vocab"]; voc.max_units=d["max_units"]
        return voc


# ──────────────────────────────────────────────────────────────────────────────
# Lexicon  (frequency-based fallback; combining-char constraints)
# ──────────────────────────────────────────────────────────────────────────────
class Lexicon:
    """
    Frequency-based fallback tagger for units not found in WordVocabulary.
    Applies BUG-1 constraint: Myanmar combining chars → always I.
    """
    def __init__(self, scheme:str=SCHEME_BIES, mode:str=MODE_CHAR):
        self.scheme=scheme; self.mode=mode
        self._u2t: Dict[str,str]={}; self._t2t: Dict[str,str]={}
        self.default_tag: str=TAG_I

    @classmethod
    def build(cls, corpus:List[List[str]], scheme:str=SCHEME_BIES, mode:str=MODE_CHAR)->"Lexicon":
        lex=cls(scheme,mode)
        uf: Dict[str,Dict[str,int]]=defaultdict(lambda: defaultdict(int))
        tf: Dict[str,Dict[str,int]]=defaultdict(lambda: defaultdict(int))
        for words in corpus:
            pairs=words_to_bies(words,mode)
            for unit,label in pairs:
                ch0=unit[0] if unit else ''
                ct=get_char_type(ch0)
                # BUG-1: Myanmar combining chars ALWAYS get I in frequency counts too
                if ct in _MY_COMBINING: label=TAG_I
                uf[unit][label]+=1
                tf[ct][label]+=1
        for u,freq in uf.items(): lex._u2t[u]=max(freq,key=freq.get)
        for ct,freq in tf.items(): lex._t2t[ct]=max(freq,key=freq.get)
        overall: Dict[str,int]=defaultdict(int)
        for freq in uf.values():
            for l,c in freq.items(): overall[l]+=c
        lex.default_tag=max(overall,key=overall.get) if overall else TAG_I
        log.info("Lexicon: %d units | %d types | default=%s | scheme=%s | mode=%s",
                 len(lex._u2t),len(lex._t2t),lex.default_tag,scheme,mode)
        return lex

    def get_tag(self, unit:str)->str:
        """Return initial BIES tag. BUG-1 and BUG-2 applied here."""
        if not unit or unit.isspace(): return TAG_S
        ch0=unit[0]
        ct=get_char_type(ch0)
        if ct in _MY_COMBINING: return TAG_I   # BUG-1 fix
        if ct==CT_MY_PUN: return self._u2t.get(unit, TAG_S)
        if unit in self._u2t: return self._u2t[unit]
        lo=unit.lower()
        if lo in self._u2t: return self._u2t[lo]
        if ct in self._t2t:
            t=self._t2t[ct]
            if ct in _MY_COMBINING and t in (TAG_B,TAG_S): return TAG_I
            return t
        return self.default_tag

    def is_known(self, unit:str)->bool:
        return unit in self._u2t or unit.lower() in self._u2t

    def save(self,path:str)->None:
        with open(path,"wb") as fh:
            pickle.dump({"u2t":self._u2t,"t2t":self._t2t,"default":self.default_tag,
                         "scheme":self.scheme,"mode":self.mode},fh,protocol=4)

    @classmethod
    def load(cls, path:str)->"Lexicon":
        with open(path,"rb") as fh: d=pickle.load(fh)
        lex=cls(d.get("scheme",SCHEME_BIES),d.get("mode",MODE_CHAR))
        lex._u2t=d["u2t"]; lex._t2t=d["t2t"]; lex.default_tag=d["default"]
        return lex

    def write_text(self,path:str)->None:
        with open(path,"w",encoding="utf-8") as fh:
            fh.write(f"# scheme={self.scheme}  mode={self.mode}  default={self.default_tag}\n")
            for u,t in sorted(self._u2t.items()): fh.write(repr(u)+"\t"+t+"\n")


# ──────────────────────────────────────────────────────────────────────────────
# RDR Node
# ──────────────────────────────────────────────────────────────────────────────
class RDRNode:
    __slots__=("condition","conclusion","except_child","else_child","depth","parent","cornerstone_cases")
    def __init__(self,condition,conclusion,depth=0,parent=None):
        self.condition=condition; self.conclusion=conclusion
        self.depth=depth; self.parent=parent
        self.except_child=None; self.else_child=None
        self.cornerstone_cases:List[List[str]]=[]
    def fires(self,ctx:List[str])->bool:
        return self.condition is None or self.condition.satisfied_by(ctx)
    def find_fired_node(self,ctx:List[str])->"RDRNode":
        cur=fired=self
        while True:
            if cur.fires(ctx):
                fired=cur
                if cur.except_child: cur=cur.except_child
                else: break
            else:
                if cur.else_child: cur=cur.else_child
                else: break
        return fired
    def write_to_file(self,fh,depth=0):
        indent="\t"*depth
        cond="True" if self.condition is None else self.condition.to_str()
        fh.write(f"{indent}{cond} : {self.conclusion}\n")
        if self.except_child: self.except_child.write_to_file(fh,depth+1)
        if self.else_child:   self.else_child.write_to_file(fh,depth)


# ──────────────────────────────────────────────────────────────────────────────
# SCRDR Tree
# ──────────────────────────────────────────────────────────────────────────────
class SCRDRTree:
    def __init__(self, default_tag:str=TAG_I):
        self.root=RDRNode(None,default_tag,0); self.default_tag=default_tag

    def classify_context(self,ctx:List[str])->str:
        return self.root.find_fired_node(ctx).conclusion

    def _apply_constraints(self, unit:str, predicted_tag:str, mode:str)->str:
        """
        BUG-4 fix: post-inference linguistic constraint for Myanmar.
        Combining chars (MV, MM, MA, MG) can NEVER be B or S.
        They are always word-internal (I) or word-final (E).
        """
        ch0=unit[0] if unit else ''
        ct=get_char_type(ch0)
        if ct in _MY_COMBINING and predicted_tag in (TAG_B,TAG_S):
            return TAG_I   # force to I (tree's I→E exception rules still apply normally)
        return predicted_tag

    def _initial_tag_sentence(self, units:List[str], vocab:"WordVocabulary",
                               lexicon:"Lexicon")->List[str]:
        """
        Apply vocabulary-based greedy longest-match initial tagging.
        Falls back to lexicon for units not in vocabulary.
        """
        return vocab.tag_units(units, fallback_lexicon=lexicon)

    def _segment_chunk(self, chunk_str:str, vocab:"WordVocabulary",
                       lexicon:"Lexicon")->List[str]:
        """Segment a single whitespace-free string (char mode)."""
        if not chunk_str: return []
        chars=list(chunk_str)
        init_tags=self._initial_tag_sentence(chars, vocab, lexicon)
        seq=[f"{c}/{t}" for c,t in zip(chars,init_tags)]
        pred=[]
        for i in range(len(chars)):
            ctx=build_unit_context(seq, i, MODE_CHAR)
            tag=self.classify_context(ctx)
            tag=self._apply_constraints(chars[i], tag, MODE_CHAR)  # BUG-4
            pred.append(tag)
            seq[i]=f"{chars[i]}/{tag}"
        return bies_to_words(list(zip(chars,pred)))

    def _segment_chunk_syl(self, chunk_str:str, vocab:"WordVocabulary",
                           lexicon:"Lexicon")->List[str]:
        """Segment a single whitespace-free string (syllable mode)."""
        if not chunk_str: return []
        syls=myanmar_syllabify(chunk_str)
        if not syls: return []
        init_tags=self._initial_tag_sentence(syls, vocab, lexicon)
        seq=[f"{s}/{t}" for s,t in zip(syls,init_tags)]
        pred=[]
        for i in range(len(syls)):
            ctx=build_unit_context(seq, i, MODE_SYLLABLE)
            tag=self.classify_context(ctx)
            pred.append(tag)
            seq[i]=f"{syls[i]}/{tag}"
        return bies_to_words(list(zip(syls,pred)))

    def segment_words_list(self, raw_text:str, vocab:"WordVocabulary",
                           lexicon:"Lexicon")->List[str]:
        """Segment a raw string into a list of words."""
        fn=self._segment_chunk_syl if lexicon.mode==MODE_SYLLABLE else self._segment_chunk
        return fn(raw_text, vocab, lexicon)

    def segment_sentence(self, raw_text:str, vocab:"WordVocabulary",
                         lexicon:"Lexicon", separator:str=" ")->str:
        """
        Segment a raw sentence (may contain spaces — BUG-2 fix).
        Split on whitespace first; each chunk is a definite token group.
        """
        result=[]
        for chunk in re.split(r'\s+', raw_text.strip()):
            if chunk: result.extend(self.segment_words_list(chunk, vocab, lexicon))
        return separator.join(result)

    def count_nodes(self)->int:
        q=[self.root]; n=0
        while q:
            nd=q.pop(); n+=1
            if nd.except_child: q.append(nd.except_child)
            if nd.else_child:   q.append(nd.else_child)
        return n

    def save_rules(self, path:str)->None:
        with open(path,"w",encoding="utf-8") as fh: self.root.write_to_file(fh,0)

    @staticmethod
    def load_rules(path:str)->"SCRDRTree":
        with open(path,"r",encoding="utf-8") as fh: lines=fh.readlines()
        parsed=[]
        for line in lines:
            s=line.strip()
            if not s or s.startswith("cc:") or s.startswith("#"): continue
            depth=len(line)-len(line.lstrip("\t"))
            if " : " not in s: continue
            c,conc=s.split(" : ",1)
            rule=None if c.strip()=="True" else Rule.from_str(c.strip())
            parsed.append((depth,rule,conc.strip()))
        if not parsed: return SCRDRTree()
        tree=SCRDRTree(parsed[0][2])
        tree.root=RDRNode(parsed[0][1],parsed[0][2],parsed[0][0])
        cur=tree.root
        for depth,rule,conc in parsed[1:]:
            nd=RDRNode(rule,conc,depth)
            if depth>cur.depth:   nd.parent=cur; cur.except_child=nd
            elif depth==cur.depth: nd.parent=cur.parent; cur.else_child=nd
            else:
                tmp=cur
                while tmp.depth!=depth: tmp=tmp.parent
                nd.parent=tmp.parent; tmp.else_child=nd
            cur=nd
        return tree

    def _to_dict(self,nd):
        if nd is None: return None
        return {"cond":nd.condition.constraints if nd.condition else None,
                "conc":nd.conclusion,"depth":nd.depth,
                "except":self._to_dict(nd.except_child),
                "else":self._to_dict(nd.else_child)}

    @staticmethod
    def _from_dict(d,parent=None):
        if d is None: return None
        rule=Rule(dict(d["cond"])) if d["cond"] else None
        nd=RDRNode(rule,d["conc"],d["depth"],parent)
        nd.except_child=SCRDRTree._from_dict(d.get("except"),nd)
        nd.else_child  =SCRDRTree._from_dict(d.get("else"),  nd)
        return nd

    def save_binary(self,path:str)->None:
        with open(path,"wb") as fh:
            pickle.dump({"default":self.default_tag,"root":self._to_dict(self.root)},fh,protocol=4)

    @staticmethod
    def load_binary(path:str)->"SCRDRTree":
        with open(path,"rb") as fh: d=pickle.load(fh)
        tree=SCRDRTree(d.get("default",TAG_I))
        tree.root=SCRDRTree._from_dict(d.get("root"))
        return tree


# ──────────────────────────────────────────────────────────────────────────────
# Object dictionary builder  (uses WordVocabulary for initial tags)
# ──────────────────────────────────────────────────────────────────────────────
ObjectDict=Dict[str,Dict[str,List[List[str]]]]

def build_object_dict(corpus:List[List[str]], vocab:"WordVocabulary",
                      lexicon:"Lexicon")->ObjectDict:
    """
    Build object dictionary using vocabulary-based initial tagging.
    For known words: init_tags == gold_tags → zero entries (no correction needed).
    For unknown words: init_tags from lexicon → tree learns corrections.
    """
    objects:ObjectDict=defaultdict(lambda: defaultdict(list))
    mode=lexicon.mode
    for words in corpus:
        gold_pairs=words_to_bies(words,mode)
        units     =[u for u,_ in gold_pairs]
        gold_tags =[t for _,t in gold_pairs]
        # Use vocabulary-based initial tagging (same as inference)
        init_tags =vocab.tag_units(units, fallback_lexicon=lexicon)
        seq       =[f"{u}/{t}" for u,t in zip(units,init_tags)]
        for idx in range(len(units)):
            ctx=build_unit_context(seq,idx,mode)
            objects[init_tags[idx]][gold_tags[idx]].append(ctx)
    return objects


# ──────────────────────────────────────────────────────────────────────────────
# Sub-correction node builder  (worker helper)
# ──────────────────────────────────────────────────────────────────────────────
def _build_correction_sub_nodes(need:Dict[str,List[List[str]]],depth:int,
                                 match_threshold:int,excluded:Set[tuple])->List[dict]:
    obj={t:list(c) for t,c in need.items() if c}
    excl=set(excluded); res=[]
    while True:
        live:Dict[tuple,Dict[str,int]]={};idx_:Dict[tuple,Dict[str,List[List[str]]]]={}
        for ctag,ctxs in obj.items():
            for ctx in ctxs:
                for rt in _gen_raw(ctx):
                    if rt in excl: continue
                    if rt not in live: live[rt]=defaultdict(int); idx_[rt]=defaultdict(list)
                    live[rt][ctag]+=1; idx_[rt][ctag].append(ctx)
        best_rt=None; best_ctag=""; best_cnt=match_threshold-1
        for rt,cc in live.items():
            for ctag,cnt in cc.items():
                if cnt>best_cnt: best_cnt=cnt; best_rt=rt; best_ctag=ctag
        if best_rt is None: break
        cs=[c for c in obj.get(best_ctag,[]) if c in idx_.get(best_rt,{}).get(best_ctag,[])]
        if not cs: excl.add(best_rt); continue
        nd={"condition":best_rt,"conclusion":best_ctag,"cs_contexts":cs,"depth":depth,"sub_nodes":[]}
        best_rule_obj=Rule(dict(best_rt))
        cs_ids={id(c) for c in cs}
        obj[best_ctag]=[c for c in obj.get(best_ctag,[]) if id(c) not in cs_ids]
        sub_need:Dict[str,List[List[str]]]={}
        for ot,ot_ctxs in obj.items():
            if ot==best_ctag: continue
            hits=[c for c in ot_ctxs if best_rule_obj.satisfied_by(c)]
            if hits:
                sub_need[ot]=hits; hit_ids={id(c) for c in hits}
                obj[ot]=[c for c in ot_ctxs if id(c) not in hit_ids]
        excl.add(best_rt)
        if sub_need:
            nd["sub_nodes"]=_build_correction_sub_nodes(sub_need,depth+1,match_threshold,excl)
        res.append(nd)
    return res


# ──────────────────────────────────────────────────────────────────────────────
# Worker  (module-level for picklability)
# ──────────────────────────────────────────────────────────────────────────────
def _worker_build_tag_subtree(args:tuple)->tuple:
    """Inverted-index + live-counts worker (same algo as v2)."""
    init_tag,tag_objects,imp_threshold,match_threshold=args
    all_ctxs:List[List[str]]=[]; id2ctag:List[str]=[]; ctag_range:Dict[str,Tuple[int,int]]={}
    for ctag,ctxs in tag_objects.items():
        s=len(all_ctxs); all_ctxs.extend(ctxs); id2ctag.extend([ctag]*len(ctxs))
        ctag_range[ctag]=(s,len(all_ctxs))
    n=len(all_ctxs)
    rule_ids:Dict[tuple,Dict[str,List[int]]]={};live_counts:Dict[tuple,Dict[str,int]]={}
    for cid in range(n):
        ctag=id2ctag[cid]
        for rt in _gen_raw(all_ctxs[cid]):
            if rt not in rule_ids: rule_ids[rt]=defaultdict(list); live_counts[rt]=defaultdict(int)
            rule_ids[rt][ctag].append(cid); live_counts[rt][ctag]+=1
    correct_live:Dict[tuple,int]={rt:live_counts[rt].get(init_tag,0) for rt in live_counts}
    removed=[False]*n

    def _remove(cid:int)->None:
        if removed[cid]: return
        removed[cid]=True; ctag=id2ctag[cid]
        for rt in _gen_raw(all_ctxs[cid]):
            if rt in live_counts and ctag in live_counts[rt]:
                live_counts[rt][ctag]=max(0,live_counts[rt][ctag]-1)
        if ctag==init_tag:
            for rt in _gen_raw(all_ctxs[cid]):
                if rt in correct_live: correct_live[rt]=max(0,correct_live[rt]-1)

    exc_nodes:List[dict]=[]
    while True:
        best_rt=None; best_ctag=""; best_net=imp_threshold-1
        for rt,cc in live_counts.items():
            for ctag,cnt in cc.items():
                if ctag==init_tag or cnt<imp_threshold: continue
                net=cnt-correct_live.get(rt,0)
                if net>best_net: best_net=net; best_rt=rt; best_ctag=ctag
        if best_rt is None or best_net<imp_threshold: break
        cs_ctxs=[all_ctxs[cid] for cid in rule_ids[best_rt][best_ctag] if not removed[cid]]
        need:Dict[str,List[List[str]]]={}
        for ot,ot_ids in rule_ids[best_rt].items():
            if ot==best_ctag: continue
            hits=[all_ctxs[cid] for cid in ot_ids if not removed[cid]]
            if hits: need[ot]=hits
        nd={"condition":best_rt,"conclusion":best_ctag,"cs_contexts":cs_ctxs,"sub_nodes":[]}
        for cid in rule_ids[best_rt][best_ctag]: _remove(cid)
        for ot,hits in need.items():
            hit_ids={id(c) for c in hits}; s_,e_=ctag_range.get(ot,(0,0))
            for cid in range(s_,e_):
                if not removed[cid] and id(all_ctxs[cid]) in hit_ids: _remove(cid)
        if need:
            nd["sub_nodes"]=_build_correction_sub_nodes(need,depth=3,
                              match_threshold=match_threshold,excluded=set())
        exc_nodes.append(nd)
    return init_tag,exc_nodes

def _assemble_nodes(node_dicts:List[dict],parent:RDRNode,depth:int)->None:
    prev=parent; first=True
    for nd in node_dicts:
        rule=Rule(dict(nd["condition"]))
        node=RDRNode(rule,nd["conclusion"],depth,parent=prev)
        node.cornerstone_cases=nd.get("cs_contexts",[])
        if first: prev.except_child=node; first=False
        else: prev.else_child=node
        prev=node
        if nd.get("sub_nodes"): _assemble_nodes(nd["sub_nodes"],node,depth+1)


# ──────────────────────────────────────────────────────────────────────────────
# SCRDR Learner
# ──────────────────────────────────────────────────────────────────────────────
class SCRDRLearner:
    def __init__(self,imp_threshold=2,match_threshold=2,n_jobs=-1,verbose=True):
        self.imp_threshold=imp_threshold; self.match_threshold=match_threshold
        self.n_jobs=n_jobs if n_jobs>0 else max(1,cpu_count()); self.verbose=verbose

    def learn(self, corpus:List[List[str]], vocab:"WordVocabulary",
              lexicon:"Lexicon")->"SCRDRTree":
        t0=time.time()
        log.info("Building object dictionary …")
        objects=build_object_dict(corpus, vocab, lexicon)
        n_units=sum(sum(len(v) for v in od.values()) for od in objects.values())
        log.info("Object dict: %d initial labels, %d total units  [%.1fs]",
                 len(objects),n_units,time.time()-t0)

        # Count how many units already have init==gold (the "free" correct ones)
        n_already_correct=sum(
            len(objects[lbl].get(lbl,[])) for lbl in objects)
        pct=100*n_already_correct/max(n_units,1)
        log.info("Initial accuracy: %d/%d = %.1f%%  (vocabulary coverage)",
                 n_already_correct,n_units,pct)

        tag_freq:Dict[str,int]=defaultdict(int)
        for od in objects.values():
            for tag,ctxs in od.items(): tag_freq[tag]+=len(ctxs)
        default_tag=max(tag_freq,key=tag_freq.get)
        log.info("Default label (root conclusion): %s",default_tag)

        init_labels=sorted(objects.keys())
        n_workers=min(self.n_jobs,len(init_labels))
        log.info("Learning rules for %d initial labels using %d worker(s) …",
                 len(init_labels),n_workers)
        worker_args=[(lbl,{t:list(c) for t,c in objects[lbl].items()},
                      self.imp_threshold,self.match_threshold) for lbl in init_labels]
        t_learn=time.time()
        if n_workers>1:
            with Pool(n_workers) as pool: results=pool.map(_worker_build_tag_subtree,worker_args)
        else: results=[_worker_build_tag_subtree(a) for a in worker_args]
        log.info("Rule learning done  [%.1fs]",time.time()-t_learn)

        tree=SCRDRTree(default_tag); prev=None; total_exc=0
        for lbl,exc_dicts in results:
            l1=RDRNode(Rule({SL_CT:lbl}),lbl,1,parent=tree.root)
            if tree.root.except_child is None: tree.root.except_child=l1
            else: prev.else_child=l1; l1.parent=prev.parent
            prev=l1
            if exc_dicts: total_exc+=len(exc_dicts); _assemble_nodes(exc_dicts,l1,2)
            if self.verbose:
                log.info("  Label %-5s : %d exception rule(s)",lbl,len(exc_dicts))
        log.info("Total exception rules: %d | Total nodes: %d | Elapsed: %.1fs",
                 total_exc,tree.count_nodes(),time.time()-t0)
        return tree


# ──────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ──────────────────────────────────────────────────────────────────────────────
def read_segmented_corpus(path:str, delimiter:str=" ")->List[List[str]]:
    sentences:List[List[str]]=[]
    with open(path,"r",encoding="utf-8") as fh:
        for line in fh:
            line=line.strip()
            if not line: continue
            words=[w for w in line.split(delimiter) if w]
            if words: sentences.append(words)
    return sentences

def read_raw_corpus(path:str)->List[str]:
    sentences:List[str]=[]
    with open(path,"r",encoding="utf-8") as fh:
        for line in fh:
            line=line.rstrip("\n")
            if line: sentences.append(line)
    return sentences


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def _get_word_spans(words:List[str])->Set[Tuple[int,int]]:
    spans:Set[Tuple[int,int]]=set(); pos=0
    for w in words: spans.add((pos,pos+len(w))); pos+=len(w)
    return spans

def evaluate_segmentation(ref:List[List[str]],hyp:List[List[str]])->dict:
    n_ref=n_hyp=n_correct=0
    for rs,hs in zip(ref,hyp):
        rsp=_get_word_spans(rs); hsp=_get_word_spans(hs)
        n_ref+=len(rsp); n_hyp+=len(hsp); n_correct+=len(rsp&hsp)
    P=n_correct/n_hyp if n_hyp else 0.0
    R=n_correct/n_ref if n_ref else 0.0
    F1=2*P*R/(P+R) if (P+R) else 0.0
    return {"precision":P,"recall":R,"f1":F1,"n_ref":n_ref,"n_hyp":n_hyp,"n_correct":n_correct}

def evaluate_full(ref:List[List[str]],hyp:List[List[str]],
                  mode:str=MODE_CHAR,lexicon=None)->dict:
    word_res=evaluate_segmentation(ref,hyp)
    all_ref:List[str]=[]; all_hyp:List[str]=[]; all_units:List[str]=[]
    for rw,hw in zip(ref,hyp):
        rp=words_to_bies(rw,mode); hp=words_to_bies(hw,mode)
        rc=[u for u,_ in rp]; rt=[t for _,t in rp]
        hc=[u for u,_ in hp]; ht=[t for _,t in hp]
        if rc==hc: all_ref.extend(rt); all_hyp.extend(ht); all_units.extend(rc)
        else:
            n=min(len(rc),len(hc))
            all_ref.extend(rt[:n]); all_hyp.extend(ht[:n]); all_units.extend(rc[:n])
    try:
        from sklearn.metrics import precision_recall_fscore_support,accuracy_score
        labels=sorted(set(all_ref)|set(all_hyp))
        p_arr,r_arr,f1_arr,sup_arr=precision_recall_fscore_support(
            all_ref,all_hyp,labels=labels,average=None,zero_division=0)
        per_label={l:{"P":float(p_arr[i]),"R":float(r_arr[i]),"F1":float(f1_arr[i]),"support":int(sup_arr[i])}
                   for i,l in enumerate(labels)}
        _,_,mf1,_=precision_recall_fscore_support(all_ref,all_hyp,average="macro",zero_division=0)
        char_acc=accuracy_score(all_ref,all_hyp)
    except ImportError:
        per_label={}; mf1=char_acc=0.0; labels=[]
    kn_acc=un_acc=None; kn_tot=un_tot=0
    if lexicon:
        kc=kt=uc=ut=0
        for u,r,h in zip(all_units,all_ref,all_hyp):
            if lexicon.is_known(u): kt+=1; kc+=(r==h)
            else: ut+=1; uc+=(r==h)
        kn_acc=kc/kt if kt else 0.0; un_acc=uc/ut if ut else 0.0
        kn_tot=kt; un_tot=ut
    return {**word_res,"char_accuracy":char_acc,"char_macro_f1":float(mf1),
            "per_label":per_label,"labels":labels,"all_ref":all_ref,"all_hyp":all_hyp,
            "kn_acc":kn_acc,"un_acc":un_acc,"kn_total":kn_tot,"un_total":un_tot}

def print_results(res:dict)->None:
    W=72
    print(f"\n{'='*W}\n  SCRDR TOKENIZER — EVALUATION RESULTS\n{'='*W}")
    print(f"  ── Word-level {'─'*54}")
    print(f"  Precision      : {res['precision']*100:8.4f}%")
    print(f"  Recall         : {res['recall']*100:8.4f}%")
    print(f"  F1             : {res['f1']*100:8.4f}%")
    print(f"  Correct/Ref/Hyp: {res['n_correct']} / {res['n_ref']} / {res['n_hyp']}")
    print(f"  ── Char/Unit-level (BIES) {'─'*41}")
    print(f"  Char Accuracy  : {res.get('char_accuracy',0)*100:8.4f}%")
    print(f"  Char Macro F1  : {res.get('char_macro_f1',0)*100:8.4f}%")
    if res.get("kn_acc") is not None:
        print(f"  Known-unit Acc : {res['kn_acc']*100:8.4f}%  ({res['kn_total']} units)")
        print(f"  Unkn-unit  Acc : {res['un_acc']*100:8.4f}%  ({res['un_total']} units)")
    if res.get("per_label"):
        print(f"  {'-'*W}")
        print(f"  {'Label':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        print(f"  {'-'*W}")
        for lbl,m in sorted(res["per_label"].items()):
            print(f"  {lbl:<8} {m['P']*100:>10.4f} {m['R']*100:>10.4f}"
                  f" {m['F1']*100:>10.4f} {m['support']:>10}")
    print("="*W)

def plot_confusion_matrix(res:dict, path:str, title:str="SCRDR Tokenizer — BIES Confusion Matrix")->None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt; import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError: log.warning("matplotlib/seaborn not installed"); return
    labels=res.get("labels",[]); 
    if not labels: return
    cm=confusion_matrix(res["all_ref"],res["all_hyp"],labels=labels)
    rs=cm.sum(axis=1,keepdims=True); rs[rs==0]=1; cm_n=cm.astype(float)/rs
    n=len(labels); fig,ax=plt.subplots(figsize=(max(5,n*1.4+1),max(4,n*1.2+1)))
    sns.heatmap(cm_n,annot=True,fmt=".3f",cmap="Blues",
                xticklabels=labels,yticklabels=labels,ax=ax,
                linewidths=0.5,vmin=0,vmax=1,annot_kws={"size":max(9,14-n)})
    ax.set_xlabel("Predicted Label",fontsize=11); ax.set_ylabel("Gold Label",fontsize=11)
    ax.set_title(f"{title}\n(Word F1 = {res['f1']*100:.2f}%)",fontsize=11)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches="tight"); plt.close(fig)
    log.info("Confusion matrix → %s",path)


# ──────────────────────────────────────────────────────────────────────────────
# Model save / load  (vocab + lexicon + tree)
# ──────────────────────────────────────────────────────────────────────────────
def save_model(model_prefix:str, tree:"SCRDRTree",
               vocab:"WordVocabulary", lexicon:"Lexicon")->None:
    os.makedirs(os.path.dirname(os.path.abspath(model_prefix)), exist_ok=True)
    tree.save_binary(model_prefix+".pkl")
    tree.save_rules(model_prefix+".rdr")
    vocab.save(model_prefix+".vocab")
    lexicon.save(model_prefix+".lex")

def load_model(model_prefix:str):
    bin_path=model_prefix+".pkl"; rdr_path=model_prefix+".rdr"
    lex_path=model_prefix+".lex"; voc_path=model_prefix+".vocab"
    if os.path.exists(bin_path):
        log.info("Loading binary model …"); tree=SCRDRTree.load_binary(bin_path)
    elif os.path.exists(rdr_path):
        log.info("Loading rules file …"); tree=SCRDRTree.load_rules(rdr_path)
    else:
        log.error("No model file found: %s (.pkl/.rdr)",model_prefix); sys.exit(1)
    if not os.path.exists(lex_path):
        log.error("Lexicon not found: %s",lex_path); sys.exit(1)
    if not os.path.exists(voc_path):
        log.error("Vocabulary not found: %s  (re-train to regenerate)",voc_path); sys.exit(1)
    lexicon=Lexicon.load(lex_path); vocab=WordVocabulary.load(voc_path)
    return tree, vocab, lexicon


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline commands
# ──────────────────────────────────────────────────────────────────────────────
def cmd_train(args:argparse.Namespace)->None:
    log.info("="*60)
    log.info("MODE: TRAIN")
    log.info("  Train file   : %s", args.train)
    log.info("  Model prefix : %s", args.model)
    log.info("  Scheme       : %s", args.scheme)
    log.info("  Mode         : %s", "syllable" if args.syllable else "char")
    log.info("  Imp thresh   : %d", args.threshold_imp)
    log.info("  Match thresh : %d", args.threshold_match)
    log.info("  Workers      : %d", args.jobs if args.jobs>0 else cpu_count())
    log.info("="*60)

    os.makedirs(os.path.dirname(os.path.abspath(args.model)), exist_ok=True)
    corpus=read_segmented_corpus(args.train, delimiter=args.delimiter)
    mode=MODE_SYLLABLE if args.syllable else MODE_CHAR
    n_chars=sum(sum(len(w) for w in s) for s in corpus)
    log.info("Corpus: %d sentences | %d words | %d chars",
             len(corpus),sum(len(s) for s in corpus),n_chars)

    # Build vocabulary AND lexicon
    vocab  =WordVocabulary.build(corpus, mode=mode)
    lexicon=Lexicon.build(corpus, scheme=args.scheme, mode=mode)

    save_model(args.model, SCRDRTree(TAG_I), vocab, lexicon)  # save vocab+lex early
    if args.save_text_lexicon: lexicon.write_text(args.model+".lex.txt")

    # Learn
    learner=SCRDRLearner(args.threshold_imp, args.threshold_match, args.jobs)
    tree   =learner.learn(corpus, vocab, lexicon)

    save_model(args.model, tree, vocab, lexicon)
    log.info("Binary model  → %s.pkl", args.model)
    log.info("Rules file    → %s.rdr  (%d nodes)", args.model, tree.count_nodes())
    log.info("Vocabulary    → %s.vocab (%d words)", args.model, len(vocab._vocab))

    if args.eval_on_train:
        log.info("Evaluating on training data …")
        hyp=[tree.segment_words_list("".join(s),vocab,lexicon) for s in corpus]
        print_results(evaluate_full(corpus,hyp,mode,lexicon))


def cmd_test(args:argparse.Namespace)->None:
    log.info("="*60)
    log.info("MODE: TEST")
    log.info("  Input  : %s", args.input)
    log.info("  Model  : %s", args.model)
    log.info("  Output : %s", args.output)
    log.info("="*60)

    tree,vocab,lexicon=load_model(args.model)
    log.info("Model loaded. Nodes: %d | mode=%s", tree.count_nodes(), lexicon.mode)

    corpus=read_segmented_corpus(args.input, delimiter=args.delimiter)
    log.info("Test corpus: %d sentences", len(corpus))

    sep=getattr(args,"separator"," ")
    t0=time.time()
    hyp_sents:List[List[str]]=[]; out_lines:List[str]=[]
    for words in corpus:
        raw="".join(words); hw=tree.segment_words_list(raw,vocab,lexicon)
        hyp_sents.append(hw); out_lines.append(sep.join(hw))
    log.info("Segmentation done  [%.2fs]", time.time()-t0)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as fh:
        for line in out_lines: fh.write(line+"\n")
    log.info("Hypothesis → %s", args.output)

    res=evaluate_full(corpus,hyp_sents,lexicon.mode,lexicon)
    print_results(res)
    if args.confusion_matrix: plot_confusion_matrix(res,args.confusion_matrix)


def cmd_segment(args:argparse.Namespace)->None:
    log.info("="*60)
    log.info("MODE: SEGMENT (raw text)")
    log.info("  Input  : %s", args.input)
    log.info("  Model  : %s", args.model)
    log.info("  Output : %s", args.output)
    log.info("="*60)

    tree,vocab,lexicon=load_model(args.model)
    sep=getattr(args,"separator"," ")
    raw_lines=read_raw_corpus(args.input)
    log.info("Segmenting %d lines …", len(raw_lines))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t0=time.time()
    with open(args.output,"w",encoding="utf-8") as fh:
        for raw in raw_lines:
            fh.write(tree.segment_sentence(raw,vocab,lexicon,separator=sep)+"\n")
    log.info("Done  [%.2fs]  →  %s", time.time()-t0, args.output)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
def _build_parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="scrdr_tokenizer",description=__doc__,
                               formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level",default="INFO",choices=["DEBUG","INFO","WARNING","ERROR"])
    sub=p.add_subparsers(dest="command",required=True,metavar="{train,test,segment}")

    t=sub.add_parser("train",help="Train SCRDR word segmentation model")
    t.add_argument("--train",required=True,metavar="FILE")
    t.add_argument("--model",required=True,metavar="PREFIX")
    t.add_argument("--jobs","-j",type=int,default=-1,metavar="N")
    t.add_argument("--threshold-imp",type=int,default=2,metavar="T",dest="threshold_imp")
    t.add_argument("--threshold-match",type=int,default=2,metavar="T",dest="threshold_match")
    t.add_argument("--scheme",default=SCHEME_BIES,choices=[SCHEME_BIES,SCHEME_BI])
    t.add_argument("--syllable",action="store_true",
                   help="Myanmar syllable-level BIES mode (recommended for Burmese)")
    t.add_argument("--delimiter",default=" ",metavar="STR")
    t.add_argument("--save-text-lexicon",action="store_true",dest="save_text_lexicon")
    t.add_argument("--eval-on-train",action="store_true",dest="eval_on_train")

    e=sub.add_parser("test",help="Evaluate on gold-standard segmented test file")
    e.add_argument("--input",required=True,metavar="FILE")
    e.add_argument("--model",required=True,metavar="PREFIX")
    e.add_argument("--output",required=True,metavar="FILE")
    e.add_argument("--confusion-matrix",metavar="FILE",dest="confusion_matrix")
    e.add_argument("--delimiter",default=" ",metavar="STR")
    e.add_argument("--separator",default=" ",metavar="STR")

    g=sub.add_parser("segment",help="Segment raw unsegmented text")
    g.add_argument("--input",required=True,metavar="FILE")
    g.add_argument("--model",required=True,metavar="PREFIX")
    g.add_argument("--output",required=True,metavar="FILE")
    g.add_argument("--separator",default=" ",metavar="STR")
    return p

def main():
    parser=_build_parser(); args=parser.parse_args()
    logging.getLogger("scrdr_tokenizer").setLevel(getattr(logging,args.log_level.upper()))
    {"train":cmd_train,"test":cmd_test,"segment":cmd_segment}[args.command](args)

if __name__=="__main__":
    import multiprocessing; multiprocessing.freeze_support()
    sys.setrecursionlimit(100_000)
    main()

