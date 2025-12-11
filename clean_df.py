import re

# emoji regex (shared)
emoji_pattern = re.compile(
    "["  
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # misc
    "\U000024C2-\U0001F251"  # enclosed characters
    "]+",
    flags=re.UNICODE,
)

def clean_hindi(text):
    text = str(text)

    # Remove Sinhala characters (U+0D80–U+0DFF)
    text = re.sub(r"[\u0D80-\u0DFF]", "", text)

    # Remove zero-width characters
    text = text.replace("\u200b", "").replace("\u200d", "").replace("\uFEFF", "")

    # Remove emojis
    text = emoji_pattern.sub("", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_english(text):
    text = str(text)

    # Remove zero-width characters
    text = text.replace("\u200b", "").replace("\u200d", "").replace("\uFEFF", "")

    # Remove emojis
    text = emoji_pattern.sub("", text)

    # Remove broken Windows-1252 corrupted unicode chars
    # (these ALWAYS appear in subtitle datasets)
    text = re.sub(r"[\x80-\x9f]", "", text)   # control-range trash

    # Remove Sinhala or accidental foreign alphabets
    text = re.sub(r"[\u0D80-\u0DFF]", "", text)   # Sinhala
    text = re.sub(r"[ν]", "", text)              # stray Greek
    text = re.sub(r"[අ-෿]", "", text)            # alt Sinhala range

    # Clean invisible repeated spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
