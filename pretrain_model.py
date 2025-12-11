from transformers import MarianTokenizer, MarianMTModel
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from translation_metrics import compute_metrics   # same function you already wrote

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------
# LOAD DATA
# -----------------------------------------
df = pd.read_csv('subtitle.csv', nrows=5000).dropna()
print(df.shape)
X = df["english"].astype(str)
y = df["hindi"].astype(str)

# 70 / 15 / 15 split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, shuffle=False, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, shuffle=False, random_state=42
)

# -----------------------------------------
# MODEL AND TOKENIZER
# -----------------------------------------
model_name = "Helsinki-NLP/opus-mt-en-hi"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name).to(device)

class MTData(Dataset):
    def __init__(self, src, tgt):
        self.src = list(src)
        self.tgt = list(tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

train_dataset = MTData(X_train, y_train)
val_dataset   = MTData(X_val,   y_val)
test_dataset  = MTData(X_test,  y_test)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)

# -----------------------------------------
# TRAIN + VAL LOOPS (on GPU)
# -----------------------------------------
def train_epoch(model, loader):
    model.train()
    total = 0

    for src_batch, tgt_batch in loader:
        enc = tokenizer(src_batch, padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(device)
        tgt = tokenizer(tgt_batch, padding=True, truncation=True,
                        max_length=64, return_tensors="pt").to(device)

        labels = tgt["input_ids"]

        optimizer.zero_grad()

        outputs = model(input_ids=enc["input_ids"],
                        attention_mask=enc["attention_mask"],
                        labels=labels)

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total += loss.item()

    return total / len(loader)

def eval_epoch(model, loader):
    model.eval()
    total = 0
    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            enc = tokenizer(src_batch, padding=True, truncation=True,
                            max_length=64, return_tensors="pt").to(device)
            tgt = tokenizer(tgt_batch, padding=True, truncation=True,
                            max_length=64, return_tensors="pt").to(device)

            labels = tgt["input_ids"]

            loss = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=labels
            ).loss

            total += loss.item()
    return total / len(loader)

# -----------------------------------------
# BATCHED TRANSLATION FOR METRICS (GPU)
# -----------------------------------------
def translate_batch(sentences, max_len=50, beam=4):
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_len,
            num_beams=beam
        )

    decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in out]
    return decoded

# -----------------------------------------
# TRAIN
# -----------------------------------------
epochs = 3
for e in range(1, epochs + 1):
    tr = train_epoch(model, train_loader)
    vl = eval_epoch(model, val_loader)
    print("Epoch", e, "| train:", tr, "| val:", vl)

torch.save(model.state_dict(), "marian_finetuned.pth")

print("Sample:", translate_batch(["i love you"])[0])
print("Sample:", translate_batch(["how are you"])[0])

# -----------------------------------------
# METRICS USING GPU + BATCHES
# -----------------------------------------
y_true = list(y_test)
y_pred = []

batch = 32
for i in range(0, len(X_test), batch):
    chunk = X_test[i:i+batch].tolist()
    preds = translate_batch(chunk)
    y_pred.extend(preds)

results = compute_metrics(y_true, y_pred)
