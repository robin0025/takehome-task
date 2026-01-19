# -*- coding: utf-8 -*-
"""
@author: Jonathan E. Robinson

clean_text.py - A configurable, pipeline to clean text lines before feeding to an LLM.

Features:
# First step so that they can be processed.
- Unicode normalization (NFC)

# Handle foreign language
- translate non-English language
- Remove non-English language

# Remove non usables
- Remove HTML entities
- Collapse_whitespace
- Remove control characters
- Remove non-printable characters
- Shrink repeated characters
- Remove leading and trailing
- Remove code
- fix 'mojibake' (text nonsense)
- Spell checking (english)
    
# Anonomise all person specific info
- Replace Personal Nouns using Natural language processing.
- Replace common profanity 
- Replace IP addresses
- Replace UUIDs
- Replace MAC Addresses
- Replace Timestamps
- Replace specific filepaths
- Replace Social security numbers
- Replace Credit Card data
- Replace Personal Identifiying Info (ie. phone, email, URL)
- Remove Personal Identifyin Info completely
- Replace Currency values

# Remove text they don't meet criteria
- Remove word repetition and set a minimum distance between word token repeats to not delete them.
- Minimum length of string before deleting it
- Maximum number of character for a token
- Maximum number of tokens per entry

Usage:
from clean_text import Cleaner
proc = Cleaner()
cleaned = proc.clean(raw_lines)

or 

# If input is chunks
from clean_text import Cleaner, clean_chunk
proc = Cleaner()
clean_chunk(pd.DataFrame,proc.cfg):

Note. Important script sections are <!!!> identified. 
"""

import os
from dataclasses import dataclass
import re
import unicodedata
from typing import Optional, Dict
import pandas as pd
import importlib
from symspellpy import SymSpell, Verbosity
import hashlib
import hmac
from datetime import datetime
from collections import defaultdict
from fast_langdetect import detect
import numpy as np
import string
import spacy

# Enabling translate
import argostranslate.package
import argostranslate.translate
# # Update package index
print('Please wait: Common translations being installed')
argostranslate.package.update_package_index()
argostranslate.package.install_package_for_language_pair('de', "en")
argostranslate.package.install_package_for_language_pair('fr', "en")
argostranslate.package.install_package_for_language_pair('es', "en")

# Optional import (handled gracefully)
try:
    import ftfy
    _HAS_FTFY = True
except Exception:
    _HAS_FTFY = False
    
### !!! Set Defaults ###
# Default replacement tokens
DEFAULT_TOKENS = {
    "EMAIL": "[EMAIL_REDACTED]",
    "PHONE": "[PHONE_REDACTED]",
    "SSN": "[SSN_REDACTED]",
    "CC": "[CREDIT_CARD_REDACTED]",
    "IP": "[IP_REDACTED]",
    "UUID": "[UUID_REDACTED]",
    "URL": "[URL_REDACTED]",
    "PATH": "[PATH_REDACTED]",
    "NAME": "[NAME_REDACTED]",
    "PROFANITY": "[PROFANITY_REDACTED]",
    "MAC": "[MAC_REDACTED]",
    "TIMESTAMP": "[TIMESTAMP_REDACTED]",
    "CURRENCY":"[CURRENCY_REDACTED]",
    "HTML": "[HTML_REDACTED]",
    "CODE": "[CODE_REDACTED]"
}   

# Basic profanity list (small sample). Replace with a fuller curated list or library.
DEFAULT_PROFANITY = {
    "damn", "shit", "fuck", "bitch", "asshole", "bastard", "crap", "prick"
}

# compiled profanity regex (word boundaries)
def build_profanity_regex(badwords: set):
    if not badwords:
        return None
    escaped = [re.escape(w) for w in badwords if w]
    if not escaped:
        return None
    pattern = r"(?i)\b(" + "|".join(escaped) + r")\b"
    return re.compile(pattern)  

#### !!! Set up all regular expressions ###
_RE_EMAIL = re.compile(r"""(?ix)
    (?<![\/\w\.-])               # not part of path/word
    [a-z0-9._%+-]+@
    [a-z0-9.-]+\.[a-z]{2,}
""")
_RE_URL = re.compile(r"""(?ix)
    (?:(?:https?|ftp)://|www\.)
    [^\s<>"]+                      # simple url capture
""")
_RE_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_RE_IPV6 = re.compile(r"\b[0-9a-fA-F:]{2,45}\b")
_RE_UUID = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b")
_RE_MAC = re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}(?:[0-9A-Fa-f]{2})\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_RE_CREDIT = re.compile(r"\b(?:\d[ -]*?){13,16}\b")
_RE_FILEPATH_UNIX = re.compile(r"(?:(?:/[\w\-.]+)+/?)")
_RE_FILEPATH_WIN = re.compile(r"(?:[A-Za-z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*)")
_RE_TIMESTAMP_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_RE_SIMPLE_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_CODE_FENCE = re.compile(r"```(?:[\s\S]*?)```")  # fenced code block
_RE_INLINE_CODE = re.compile(r"`[^`]*`")
_RE_EMAIL_PATH_LIKE = re.compile(r"[A-Za-z0-9._%+-]+@[^\s]+")  # fallback
_RE_FILENAME_EXT = re.compile(r"\b[\w,\s-]+\.[A-Za-z0-9]{1,6}\b")

_RE_CONTROL = re.compile(
    r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]"
)  # common C0/C1 control chars
_RE_MULTISPACE = re.compile(r"\s+")
_RE_REPEATS = re.compile(r"(.)\1{4,}")  # repeated char more than 4 times
_RE_SENTENCE_SPLIT = re.compile(r'(?<=[\.\?\!])\s+')

# We'll validate candidates with a function (digit count, date heuristics).
_RE_PHONE_CAND = re.compile(r'(?<!\w)(\+?[\d\-\.\s()]{6,}\d)(?!\w)')
 
# Common date patterns to avoid misclassifying as phone numbers:
_RE_ISO_DATE = re.compile(r'^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*$')         # 2023-10-05 or 2023/10/05
_RE_DMY_DATE = re.compile(r'^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*$')       # 12-31-2023 or 31/12/23
_RE_COMPACT_DATE = re.compile(r'^\s*\d{8}\s*$')                          # 20231005
_RE_SHORT_YEAR = re.compile(r'^\s*\d{4}\s*$')                            # 2023

# Currency detection: symbols and common 3-letter codes
_CURRENCY_SYMBOLS = r'[$€£¥₩₹₽₴₪₫₱฿₦]'
_RE_CURRENCY_SYMBOL = re.compile(rf'{_CURRENCY_SYMBOLS}')
_RE_CURRENCY_CODE = re.compile(r'(?i)\b(?:USD|EUR|GBP|JPY|CNY|AUD|CAD|INR|RUB|KRW|BRL)\b')
# A simple currency amount pattern (e.g., $1,000.00, 1.000,00€, 10 USD)
_RE_CURRENCY_AMOUNT = re.compile(
    rf'''(?x)
    ^\s*                              # optional surrounding space
    (?:{_CURRENCY_SYMBOLS})?          # optional leading symbol
    \s*
    -?                                # optional negative sign
    (?:
        \d{{1,3}}(?:[,\.\s]\d{{3}})+  # thousands separators like 1,000 or 1.000 or 1 000
        |
        \d+(?:[.,]\d+)?               # plain number, optional decimal
    )
    \s*(?:{_CURRENCY_SYMBOLS})?       # optional trailing symbol
    (?:\s*(?:USD|EUR|GBP|JPY|CNY|AUD|CAD|INR|RUB|KRW|BRL))?  # optional currency code
    \s*$''')

_RE_PROFANITY = build_profanity_regex(DEFAULT_PROFANITY)


### !!! CLEANING CONFIGURATION CLASS ###
@dataclass
class CleanConfig:

    ###NOTE ORDER IS IMPORTANT###
    # First step so that they can be processed.
    normalize_unicode: bool = True
    
    # Handle foreign language
    translate_lang: bool = True #if true foreign language will be translated
    remove_lang: bool = False # if true foreign language will be deleted to reduced noise
    
    # Remove no usable
    remove_html_tags: bool = True
    collapse_whitespace: bool = True
    remove_control_chars: bool = True
    remove_nonprintable: bool = True  # keep if you want some CJK/emoji retained
    shrink_repeated_chars: bool = True
    remove_lead_trail: bool = True
    remove_code: bool = False # remove code if False, code will be tagged to avoid spellchecking
    # Fix mojibake (word salad) using fix that for you [ftfy]
    fix_mojibake: bool = True  # requires ftfy; if True but ftfy missing, ignored
    # Spell check to avoid fragmented tokens
    spell_check: bool = True # check spelling (applied to english language only)

    # Replace specific info
    replace_with_spacy = True # Use Natural language processing to annonimise key nouns
    replace_profanity: bool = True # replace profanity 
    replace_ip: bool = False # anon IP addresses
    replace_uuid: bool = True # anon uuid
    replace_mac: bool = True # anon mac addresses
    replace_timestamp: bool = True # anon timestamps
    replace_filepath: bool = True # anon filepaths 
    replace_ssn: bool = True # anon ssn
    replace_cc: bool = True # anon credit card data
    anonymize_pii: bool = True  # replace PII with placeholders instead of removing
    remove_pii: bool = False  # replace PII with placeholders instead of removing
    anonymize_currency: bool = False # if anonymize=True and anonymize_currency=True, replace currency with <CURRENCY>
    
    # Remove text they don't meet criteria
    check_repeats: Optional[int] = 2 # sets the minimum distance between word token repeats to not delete them.
    min_drop_length: int = 1  # drop lines shorter than this (after cleaning)
    max_chars: Optional[int] = None  # truncate by characters
    max_tokens: Optional[int] = None  # truncate by tokens (uses simple tokenization)
    
### !!! USEFUL FUNCTIONS ###

# Logical for deciding if the candidate string is a date
def _looks_like_date(candidate: str) -> bool:
    c = candidate.strip()
    c = c.strip("()[]\"'")
    if _RE_ISO_DATE.match(c):
        return True
    if _RE_DMY_DATE.match(c):
        return True
    # compact numeric date like 20231005
    if _RE_COMPACT_DATE.match(re.sub(r'\D', '', c)):
        return True
    if _RE_SHORT_YEAR.match(c):
        return True
    return False


# Logical for deciding if the candidate string is a currency
def _looks_like_currency(candidate: str) -> bool:
    c = candidate.strip()
    # quick checks: presence of currency symbol or code
    if _RE_CURRENCY_SYMBOL.search(c):
        return True
    if _RE_CURRENCY_CODE.search(c):
        return True
    # match against amount pattern
    if _RE_CURRENCY_AMOUNT.match(c):
        return True
    # also common patterns with trailing currency code (e.g., "100 USD")
    if re.search(r'\b\d[\d,\.]*\s*(USD|EUR|GBP|JPY|CNY|AUD|CAD|INR|RUB|KRW|BRL)\b', c, re.I):
        return True
    return False

# Logical for deciding if the candidate string is a phone number
def  _looks_like_phone(candidate: str) -> bool:
   c = candidate.strip()
   c = re.sub(r'\D', '', c)
   #add _sub_and_log
   if _RE_PHONE_CAND.search(c):
       return True
   return False

# Logical for aggregating other possible types for candidate
def _check_possible(candidate: str) -> bool:
   if _looks_like_phone(candidate) or _looks_like_currency(candidate) or _looks_like_date(candidate):
       return True
   return False

# Decision point for whether to consider condidate as a phone number
def _phone_candidate_replacer(self, m: re.match):
    candidate = m.group(1)
    digits_only = re.sub(r'\D', '', candidate)
    # Too few digits -> not a phone
    if len(digits_only) < 7:
        return candidate
    # If candidate looks like a date -> keep it
    if _looks_like_date(candidate):
        return candidate
    # If candidate looks like currency -> keep it (unless user explicitly asked to remove currency)
    if _looks_like_currency(candidate):
        if self.cfg.anonymize_currency:
            self._record_audit("CURRENCY", candidate)
            return "<CURRENCY>"

        else:
            return candidate
    # Otherwise treat as phone
    #add _sub_and_log
    if self.cfg.anonymize_pii:
        self._record_audit("PHONE", candidate)
        return "<PHONE>" 
    elif self.cfg.remove_pii and not self.cfg.anonymize_pii:
        self._record_audit("RM_PHONE", candidate)
        return ""
    else:
        return candidate

# Top level removal / anonimise phone function
def _remove_or_anonymize_phones_and_currency(self, s: str):
    # First, handle per-pattern replacements for currency/phone when anonymize=True
    # Use a callable to validate candidates
    def _repl(m):
        return _phone_candidate_replacer(self, m)
    return _RE_PHONE_CAND.sub(_repl, s)


# Luhn check for credit card validation (reduces false positives)
def _luhn_check(cc_digits: str) -> bool:
    digits = [int(ch) for ch in re.sub(r"\D", "", cc_digits)]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    alt = False
    for d in reversed(digits):
        if alt:
            d = d * 2
            if d > 9:
                d -= 9
        checksum += d
        alt = not alt
    return checksum % 10 == 0

# Loop through repeats until non meeting criteria exist
def _handle_repeats(token_list: list, min_dist: int):
    its = 0
    it_lim = 1000
    # get initial tallys and diffs
    word_tally, word_diff = _get_tally(token_list, min_dist)
    while word_diff:
        # get the first item in the dictionary (avoids index error)
        del_ind = word_tally[next(iter(word_diff))]
        # delete one set of repeats that meet criteria
        token_list = _del_repeat(token_list, del_ind, min_dist)
        # get the tally again ready for while criteria
        word_tally, word_diff = _get_tally(token_list, min_dist)
        if its == it_lim:
            print("Limit Reached")
            break
        # increment iterations
        its += 1
                
    return token_list

# Tally up word entries
def _get_tally(token_list: list, diff_min: int):

    # Put tally of each word in dictionary
    tally = defaultdict(list)
    for i,item in enumerate(token_list):
        tally[item].append(i)
    
    # Put difference that meet criteria in dictionary
    difference = defaultdict(list)
    for token in tally:
        diffs = np.diff(tally[token])
        # include only meets requirements
        if not all(diffs >= diff_min):
            difference[token].append(diffs)

    return(tally,difference)

# Deletes indices based on criteria
def _del_repeat(token_list: list, dup_ind: list, diff_min: int):
    indices = np.array(dup_ind)
    ind_diffs = np.concatenate((np.array([0]), np.diff(indices)))
    for i in range(len(indices)-1,0,-1):
        if ind_diffs[i] < diff_min:
            del (token_list[indices[i]])
    return token_list
        

# HMAC helper for deterministic mapping (pseudonymization)
def _hmac_hash(value: str, key: Optional[str]) -> str:
    if not key:
        # Not secure for production; use a real key
        return hashlib.sha256(value.encode("utf-8")).hexdigest()
    return hmac.new(key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256).hexdigest()

# Truncate by tokens limiting maximum token to form sentence
def _truncate_by_tokens(s: str, max_tokens: int, sentence_truncate: bool) -> str:
    if max_tokens is None and not sentence_truncate:
        return s
    tokens = s.split()
    if max_tokens != None:
        if len(tokens) <= max_tokens:
            return s
    if not sentence_truncate:
        return " ".join(tokens[:max_tokens])
    
    # Sentence-aware: accumulate sentences until reaching tokens
    sentences = _RE_SENTENCE_SPLIT.split(s)
    out = []
    total = 0
    
    if max_tokens != None:
        for sent in sentences:
            t = len(sent.split())
            if total + t <= max_tokens:
                out.append(sent)
                total += t
            else:
                # If no sentences yet, fall back to token cut
                if not out:
                    out = [" ".join(tokens[:max_tokens])]
                break
        return " ".join(out).strip()
    else:
        return sentences

# Split the text into parts, keeping track of punctuation placement # for sym_spell
def _correct_with_punct(text: str, spell_model):
    parts = []
    current_word = []
    for char in text:
        if char in string.punctuation or char.isspace():
            if current_word:
                parts.append("".join(current_word))
                current_word = []
            parts.append(char)
        else:
            current_word.append(char)
    if current_word:
        parts.append("".join(current_word))

    corrected_parts = []
    for part in parts:
        # Don't correct if proper noun if or first word in sentence 
        # Dev note. spell checking could be better.
        if part in string.punctuation or part.isspace() or part[0].isupper():
            corrected_parts.append(part)
        else:
            # Perform lookup for the word part
            suggestions = spell_model.lookup(part, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected_parts.append(suggestions[0].term)
            else:
                corrected_parts.append(part) # Keep unknown words as they are

    return "".join(corrected_parts)


       
### !!! CLEANER CLASS ###       
class Cleaner:
    def __init__(self):
        self.cfg = CleanConfig() 
        self.tokens = DEFAULT_TOKENS.copy()
        self.profanity_list = DEFAULT_PROFANITY.copy()
        self.hmac_key = None
        self.audit_log: Dict[str, Dict] = {}
        self.dedupe = True # when cleaning batch, drop duplicate cleaned lines
        self.cleaned = set()
        self.sentence_truncate = True
        self.code_stuct = True # if True white space will not be remove from code tagged sections
        
        # Get spell check ready
        if self.cfg.spell_check:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            pkg_path = importlib.resources.files("symspellpy")
            dictionary_path = os.path.join(pkg_path.absolute(),"frequency_dictionary_en_82_765.txt")
            self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        
        # Import only if in the cfg
        if self.cfg.replace_with_spacy:
            try:
                self.spacy_model = "en_core_web_sm"
                self.nlp = spacy.load(self.spacy_model)
            except Exception:
                # fallback
                self.cfg.replace_with_spacy= False
                self.nlp = None
            
    def _record_audit(self, tag: str, original: str, replacement=None):
        # Save mapping with count
        key = f"{tag}:{replacement}"
        if key not in self.audit_log and replacement==None:
            self.audit_log[key] = {"tag": tag, "originals": []}
        elif key not in self.audit_log:
            self.audit_log[key] = {"tag": tag, "replacement": replacement, "originals": []}
        self.audit_log[key]["originals"].append(original)
  
    def _sub_and_log(self, tag: str, orig: str):
        """Return replacement token (or deterministic hash) and record audit."""
        if self.hmac_key:
            hashed = _hmac_hash(orig, self.hmac_key)
            rep = f"[{tag}_HASH_{hashed[:16]}]"
            self._record_audit(tag, orig, rep)
            return rep
        else:
            rep = self.tokens.get(tag, f"[{tag}_REDACTED]")
            self._record_audit(tag, orig, rep)
            return rep
        
     
    ### !!! PRIMARY CLEANING FUNCTIONS ###  
    def clean(self, text: str):
        # if not text:
        #     return text, {}     
        s = text
        self.pid = os.getppid()
        self.timeAtClean= str(datetime.now())
        self.audit_log.update({'PID':self.pid})
        self.audit_log.update({'TIME_ON_CLEAN':self.timeAtClean})
        ### !!! Run through configuration and implement paired function ###
        # go through attributes of config
        for attr in vars(self.cfg).keys():
            # get attribute value
            val = eval("self.cfg."+attr)
            if val:  # if true of has value
                # Note: configuration parameter must have paired function ie. normalize_unicode = True & _normalize_unicode(self, s: str)
                # run function on s 
                s = eval("self._%s(s)"%(attr))
        
        # handle output
        # print(s)
        return(s, self.audit_log)
        
    ### !!! FUNCTIONS CALLED IF IN CONFIG ###
    # Normalize Unicode
    def _normalize_unicode(self, s: str):
        s = unicodedata.normalize("NFC", s)
        
        return s
    
    
    # Fix mojibake if requested and ftfy available
    def _fix_mojibake(self, s: str):
        if self.code_stuct:
            fence = _RE_CODE_FENCE.search(s)
            inline = _RE_INLINE_CODE.search(s)
        else:
            fence = None
            inline = None
        if not fence and not inline:
            orig = str(s)
            if _HAS_FTFY: 
                try:
                    s = ftfy.fix_text(s)
                    if orig is not s:
                        self._record_audit("MOJI_FIX", orig)
                except Exception:
                    # If ftfy fails, continue with best-effort
                    pass
        
        return s
     
    # HTML tags
    def _remove_html_tags(self, s: str):
        m = _RE_HTML_TAGS.search(s)
        if m:
            orig = m.group(0)
            self._record_audit("HTML", orig)
            s = _RE_HTML_TAGS.sub(self.tokens["HTML"], s)
           
        return s
     
        
    # # HTML entities
    # def _unescape_html_entities(self, s: str):
    #     s = html.unescape(s)
        
    # # Collapse whitespace
    def _collapse_whitespace(self, s: str):
        if self.code_stuct:
            fence = _RE_CODE_FENCE.search(s)
            inline = _RE_INLINE_CODE.search(s)
        else:
            fence = None
            inline = None
        m = _RE_MULTISPACE.search(s)
        if not fence and not inline:
            if m:
                orig = m.group(0)
                self._record_audit("MULTISPACE", orig)
                s = _RE_MULTISPACE.sub(" ", s)
             
        return s
     
        
    # # Control characters
    def _remove_control_chars(self, s: str):
        if self.code_stuct:
            fence = _RE_CODE_FENCE.search(s)
            inline = _RE_INLINE_CODE.search(s)
        else:
            fence = None
            inline = None
        m = _RE_CONTROL.search(s)
        if not fence and not inline:
            if m:
                # Remove common control characters
                orig = m.group(0)
                # rep = self.tokens["URL"]
                self._record_audit("CONTROL", orig)
                # add _sub_and_log
                s = _RE_CONTROL.sub("", s)
        
        return s
    
    
    # # Optionally remove non-printables (conservative)
    def _remove_nonprintable(self, s: str):
        orig = str(s)
        s = "".join(ch for ch in s if ch.isprintable())
        if orig is not s:
            self._record_audit("STRIP_NONPRINTABLE", orig)
        
        return s
    
    
    def _shrink_repeated_chars(self, s: str):
        m = _RE_REPEATS.search(s)
        if m:
            orig = m.group(0)
            self._record_audit("SHRINK_REPEATS", orig)
            rep = m.group(1) * 4
            s = _RE_REPEATS.sub(rep, s)

        return s
    
     
    # strip leading and trailing characters    
    def _remove_lead_trail(self, s: str):
        orig = str(s)
        s = s.strip()
        if orig is not s:
            self._record_audit("STRIP_LEAD_TRAIL", orig)
        
        return s
    
    
    # EVERYTHING NEEDS TO SUB AND LOG
    def _replace_profanity(self, s: str):
        # Remove profanity's
        m = _RE_PROFANITY.search(s)
        if m:
            s = _RE_PROFANITY.sub(lambda m: self._sub_and_log("PROFANITY", m.group(0)), s)
        
        return s
    
    
    def _replace_ip(self, s: str):
        # Remove IPs
        m = _RE_IPV4.search(s)
        if m:
            other = _check_possible(m.group(0))
            if not other:
                s = _RE_IPV4.sub(lambda m: self._sub_and_log("IPV4", m.group(0)), s)
        m = _RE_IPV6.search(s)
        if m:    
            other = _check_possible(m.group(0))
            if not other:
                s = _RE_IPV6.sub(lambda m: self._sub_and_log("IPV6", m.group(0)), s)

        return s
    
    
    def _replace_uuid(self, s: str):
        # Remove uuids
        m = _RE_UUID.search(s)
        if m:
            other = _check_possible(m.group(0))
            if not other:
                s = _RE_UUID.sub(lambda m: self._sub_and_log("UUID", m.group(0)), s)
        
        return s  
    
    
    def _replace_mac(self, s: str):
        # Remove MACs
        m = _RE_MAC.search(s)
        if m:
            other = _check_possible(m.group(0))
            if not other:
                s = _RE_MAC.sub(lambda m: self._sub_and_log("MAC", m.group(0)), s)
        
        return s
    
    
    def _replace_timestamp(self, s: str):
        # Remove timestamps
        m = _RE_TIMESTAMP_ISO.search(s)
        if m:
            s = _RE_TIMESTAMP_ISO.sub(lambda m: self._sub_and_log("TIMESTAMP", m.group(0)), s)
        m = _RE_SIMPLE_TIME.search(s)
        if m:
            s = _RE_SIMPLE_TIME.sub(lambda m: self._sub_and_log("TIMESTAMP", m.group(0)), s)
        
        return s
    
    
    def _replace_filepath(self, s: str):
        # Remove filepaths
        m = _RE_SIMPLE_TIME.search(s)
        if m:
            s = _RE_FILEPATH_WIN.sub(lambda m: self._sub_and_log("PATH", m.group(0)), s)
        m = _RE_SIMPLE_TIME.search(s)
        if m:    
            s = _RE_FILEPATH_UNIX.sub(lambda m: self._sub_and_log("PATH", m.group(0)), s)
        
        return s
    
    
    def _replace_ssn(self, s: str):
        # Remove SSNs
        m = _RE_SSN.search(s)
        if m:
            other = _check_possible(m.group(0))
            if not other:
                s = _RE_SSN.sub(lambda m: self._sub_and_log("SSN", m.group(0)), s)

        return s   
    
    
    # Credit card-ish patterns: validate with Luhn to reduce false positives
    def _replace_cc(self, s: str):
        m = _RE_CREDIT.search(s)
        if m: 
            other = _check_possible(m.group(0))
            if not other:
                candidate = m.group(0)
                if _luhn_check(candidate):
                    s = _RE_CREDIT.sub(lambda m: self._sub_and_log("CC", candidate), s)
       
        return s
    
    
    def _remove_code(self, s: str):
        m = _RE_CODE_FENCE.search(s)
        if m:
            orig = m.group(0)
            s = _RE_CODE_FENCE.sub(self.tokens["CODE"], s)
            self._record_audit("CODE", orig, s)
        
        m = _RE_INLINE_CODE.search(s)
        if m:
            orig = m.group(0)
            s = _RE_INLINE_CODE.sub(self.tokens["CODE"], s)
            self._record_audit("CODE", orig, s)
        
        return s
    
        
    # spelling errors can lead to fragmenting of tokens so spell checking is an important step
    def _spell_check(self, s: str):
        # Check it isn't code
        fence = _RE_CODE_FENCE.search(s)
        inline = _RE_INLINE_CODE.search(s)
        model = self.sym_spell
        if not fence and not inline:
            m = _RE_SENTENCE_SPLIT.split(s)
            if m:
                result = [_correct_with_punct(item,model) for item in m]
                new = ' '.join([item for item in result])
                if new is not s:
                    s = str(new)
                    self._record_audit("SPELLING", s, new)
         
        return s
    
    
    # I noticed some inputs in alternative languages so this function check the language of the text and translates if necessary
    def _translate_lang(self, s: str):
        orig = str(s)
        source_lang = detect(s)[0]['lang']
        # logic for all lines to check if a foreign language is used.
        if source_lang != 'en': # use english translation
            try:
                # using argostranslate
                argostranslate.package.install_package_for_language_pair(source_lang, "en")
                # # Get translation
                from_lang = argostranslate.translate.get_language_from_code(source_lang)
                to_lang = argostranslate.translate.get_language_from_code("en")
                translation = from_lang.get_translation(to_lang)
                # # Translate text
                s = translation.translate(s)
                self._record_audit("TRANSLATED", orig, s)
            except:
                return s

        return s
    
    
    def _remove_lang(self, s: str):
        # logic for all lines to check if a foreign language is used.
        orig = str(s)
        lang = detect(s)[0]['lang']
        if lang != 'en': # delete
            s = ''
            self._record_audit("LANGUAGE_REMOVE", orig, s)
        
        return s


    def _anonymize_pii(self, s: str):
        m = _RE_URL.search(s) 
        if m:
            s = _RE_URL.sub(lambda m: self._sub_and_log("URL", m.group(0)), s)
            
        m = _RE_EMAIL.search(s) 
        if m:
            s = _RE_EMAIL.sub(lambda m: self._sub_and_log("EMAIL", m.group(0)), s)
        
        # This will replace phone candidates (and optionally currency if configured)
        s = _remove_or_anonymize_phones_and_currency(self, s)

        return s
    
    
    def _remove_pii(self, s: str):
        m = _RE_URL.search(s) 
        if m:
            orig = m.group(0)
            s = _RE_URL.sub("", s)
            self._record_audit("URL_RM", orig,s)
            
        m = _RE_EMAIL.search(s) 
        if m:
            orig = m.group(0)
            s = _RE_EMAIL.sub("", s)
            self._record_audit("EMAIL_RM", orig,s)
            
        if self.cfg.remove_phones:
            s = _remove_or_anonymize_phones_and_currency(self, s)

        return s
    
    
    # If anonymize_currency specifically requested and currency amounts not caught by phone cand, replace them
    def _anonymize_currency(self, s: str):
        m = _RE_CURRENCY_AMOUNT.search(s)
        if m:    
            s = _RE_CURRENCY_AMOUNT.sub(lambda m: self._sub_and_log("CURRENCY", m.group(0)), s)
        
        m = _RE_CURRENCY_CODE.search(s) 
        if m:    
            s = _RE_CURRENCY_CODE.sub(lambda m: self._sub_and_log("CURRENCY", m.group(0)), s)
        
        m = _RE_CURRENCY_SYMBOL.search(s) 
        if m:    
            s = _RE_CURRENCY_SYMBOL.sub(lambda m: self._sub_and_log("CURRENCY", m.group(0)), s)

        return s
    
    
    # Drop short lines
    def _min_drop_length(self, s: str):
        orig = str(s)
        if self.cfg.min_drop_length and len(s) < self.cfg.min_drop_length:
            s = None
            self._record_audit("TOO SHORT", orig)   
            
        return s
    
    
    def _replace_with_spacy(self, s: str):
        doc = self.nlp(s)
        # Entities to redact: PERSON, NORP, ORG, GPE, LOC, PRODUCT (tunable)
        ent_types = {"PERSON", "NORP", "ORG", "GPE", "LOC"}
        spans = []
        for ent in doc.ents:
            if ent.label_ in ent_types:
                spans.append((ent.start_char, ent.end_char, ent.text, ent.label_))
        # Replace from end to start to preserve offsets
        parts = []
        last = 0
        for start, end, orig, label in sorted(spans, key=lambda x: x[0]):
            parts.append(s[last:start])
            parts.append(self._sub_and_log("NAME", orig))
            last = end
        parts.append(s[last:])
        s = "".join(parts)

        return s
    
    def _check_repeats(self, s: str):
        # Check it isn't code
        fence = _RE_CODE_FENCE.search(s)
        inline = _RE_INLINE_CODE.search(s)
        # store original for logging
        orig = str(s)
        if not fence and not inline:
            # seperate into sentences
            sentences = _truncate_by_tokens(s, None, True)
            # check there aren't sentence repeats
            sentences = _handle_repeats(sentences, self.cfg.check_repeats) 
            new_sentences = []
            for sent in sentences:
                end = sent[-1] #get final punctuation
                sent = sent[0:-1] #remove punctuation
                tokens = sent.split(' ') # split words and take out fullstop 
                # check there aren't word repeats
                tokens = _handle_repeats(tokens, self.cfg.check_repeats) 
                new_sentences.append(" ".join(tokens)+end) # knit back together and add to sentences 
            
            # for future version pass sentences through NLP
            s = " ".join(new_sentences)
            # log change
            self._record_audit("RM_REPEATS", orig,s)
        
        return s

        
    # Truncate by chars [char_trunc == maximum character length]
    def _max_char(self, s: str):
        if self.cfg.max_char is not None and self.cfg.max_char > 0 and len(s) > self.max_char:
            s = s[: self.cfg.char_trunc].rstrip()

        return s
    
    
    # Truncate by tokens (with optional sentence-aware truncation)
    def _max_tokens(self, s: str):
        if self.cfg.max_tokens is not None and self.cfg.max_tokens > 0:
            
            s = _truncate_by_tokens(s, self.cfg.max_tokens, self.sentence_truncate) #look into truncation and add in a word repeat detector

        return s
    
        

        
### !!! Not currently used but: Function to use if handed chunk to clean ###
def clean_chunk(df: pd.DataFrame,cfg):
    proc = Cleaner()
    lines = df['text']
    seen = set()
    reports = set()

    for raw in lines:
        (cleaned,report)=proc.clean()
        if cleaned is None:
            continue
        if cfg.dedupe:
            if cleaned in seen:
                continue
            seen.add(cleaned)
            reports.add(report)
            
        yield(cleaned,report)
    print('Chunk processed')
    return(seen,reports)

