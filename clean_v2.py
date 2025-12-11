import pandas as pd
import re
import string
import contractions
import unicodedata

# Hindi Unicode ranges (Devanagari + matras)
def is_hindi_char(ch):
    cp = ord(ch)
    return (
        0x0900 <= cp <= 0x097F or
        0xA8E0 <= cp <= 0xA8FF or
        0x1CD0 <= cp <= 0x1CFF
    )

def clean_english(text):
    # lowercase
    text = text.lower()

    # expand contractions
    text = contractions.fix(text)

    # remove punctuation EXCEPT characters useful for translation
    allowed = set(string.ascii_lowercase + string.digits + " ")
    cleaned = []
    for ch in text:
        if ch in allowed:
            cleaned.append(ch)
        else:
            # replace punctuation with space
            cleaned.append(" ")
    text = "".join(cleaned)

    # normal spacing
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_hindi(text):
    # lowercase (safe)
    text = text.lower()

    cleaned = []
    for ch in text:
        if is_hindi_char(ch):
            cleaned.append(ch)
        elif ch in [" ", "।"]: # space + danda allowed
            cleaned.append(" ")
        else:
            # remove western punctuation, emojis, junk
            cleaned.append(" ")

    text = "".join(cleaned)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_text(data, source_col, target_col):
    # clean English
    data[source_col] = data[source_col].astype(str).apply(clean_english)

    # clean Hindi
    data[target_col] = data[target_col].astype(str).apply(clean_hindi)

    # add start/end tokens
    data[target_col] = data[target_col].apply(lambda x: "start_ " + x + " _end")

    # vocabulary extraction
    src_words = set(word for sent in data[source_col] for word in sent.split())
    trg_words = set(word for sent in data[target_col] for word in sent.split())

    print("Source vocab:", len(src_words))
    print("Target vocab:", len(trg_words))
    return src_words, trg_words
