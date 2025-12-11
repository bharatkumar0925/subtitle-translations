import ast
import pandas as pd


path = r"C:\Users\BHARAT\Desktop\data sets\text datasets\subtitle\en-hi_train.csv"
df = pd.read_csv(path, index_col=0)

df["parsed"] = df["translation"].apply(ast.literal_eval)

# extract English and Hindi
df["english"] = df["parsed"].apply(lambda x: x.get("en", ""))
df["hindi"]   = df["parsed"].apply(lambda x: x.get("hi", ""))

# drop old column
df = df.drop(columns=["translation"])
df = df[['english', 'hindi']]
print(df[['english', 'hindi']].head())


eng_char = df['english'].str.len()
hi_char = df['hindi'].str.len()
print(eng_char.nlargest(), hi_char.nlargest())

print(eng_char.describe())
print(hi_char.describe())

df["en_words"] = df["english"].apply(lambda x: len(str(x).split()))
df["hi_words"] = df["hindi"].apply(lambda x: len(str(x).split()))

print(df[["english", "en_words", "hindi", "hi_words"]].head())


print(df["en_words"].describe())
print(df["hi_words"].describe())
def get_unique_chars(texts):
    chars = set()
    for t in texts:
        for ch in str(t):
            chars.add(ch)
    return chars

en_chars = get_unique_chars(df["english"])
hi_chars = get_unique_chars(df["hindi"])

print("English unique chars:", sorted(en_chars))
print("\nHindi unique chars:", sorted(hi_chars))


import re

hindi_contains_english = df["hindi"].apply(lambda x: bool(re.search(r"[A-Za-z]", x)))
print(hindi_contains_english.sum(), "Hindi rows contain English letters")


english_contains_hindi = df["english"].apply(lambda x: bool(re.search(r"[\u0900-\u097F]", x)))
print(english_contains_hindi.sum(), "English rows contain Hindi letters")
