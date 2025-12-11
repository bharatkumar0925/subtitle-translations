from clean_df import clean_hindi, clean_english
from peft import LoraConfig, get_peft_model, TaskType, PromptTuningConfig, PrefixTuningConfig
from transformers import MarianTokenizer, MarianMTModel
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from translation_metrics import compute_metrics

#np.random.seed(42)
#random.seed(42)
#torch.manual_seed(42)


device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------
# CONFIG (CHANGE VALUES ONLY HERE)
# -----------------------------------------
MAX_LEN = 64
BEAM_SIZE = 4
BATCH_SIZE = 16
LR = 1e-7
WEIGHT_DECAY = 1e-3
EPOCHS = 5
MODEL_NAME = "Helsinki-NLP/opus-mt-en-hi"

# -----------------------------------------
# LOAD & CLEAN DATA
# -----------------------------------------
df = pd.read_csv("subtitle.csv", nrows=10000).dropna()

df["english"] = df["english"].apply(clean_english)
df["hindi"]   = df["hindi"].apply(clean_hindi)

X = df["english"].astype(str)
y = df["hindi"].astype(str)

# 70 / 15 / 15
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, shuffle=True, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, shuffle=True, random_state=42
)

prompt_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=20,
    prompt_tuning_init='text',
    prompt_tuning_init_text='Translate properly, handle name, place and entity correctly.'
)

prefix_config = PrefixTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    num_virtual_tokens=5,
    prefix_projection=True
)

# -----------------------------------------
# MODEL + LoRA
# -----------------------------------------
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj']
)

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(device)
#model = get_peft_model(model, prompt_config)
#model = get_peft_model(model, lora_config)

#model.print_trainable_parameters()

# -----------------------------------------
# DATASET
# -----------------------------------------
class MTData(Dataset):
    def __init__(self, src, tgt):
        self.src = list(src)
        self.tgt = list(tgt)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

train_loader = DataLoader(MTData(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(MTData(X_val, y_val),   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(MTData(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# -----------------------------------------
# TRAIN / VAL FUNCTIONS
# -----------------------------------------
def encode_batch(text_list):
    return tokenizer(
        text_list,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(device)

def train_epoch(model, loader):
    model.train()
    total = 0

    for src_batch, tgt_batch in loader:
        enc = encode_batch(src_batch)
        tgt = encode_batch(tgt_batch)

        optimizer.zero_grad()

        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            labels=tgt["input_ids"]
        )

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
            enc = encode_batch(src_batch)
            tgt = encode_batch(tgt_batch)

            loss = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                labels=tgt["input_ids"]
            ).loss

            total += loss.item()

    return total / len(loader)

# -----------------------------------------
# TRANSLATION
# -----------------------------------------
def translate_batch(sentences):
    enc = encode_batch(sentences)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=MAX_LEN,
            num_beams=BEAM_SIZE
        )
    return [tokenizer.decode(t, skip_special_tokens=True) for t in out]

# -----------------------------------------
# TRAIN
# -----------------------------------------
for e in range(1, EPOCHS + 1):
    tr = train_epoch(model, train_loader)
    vl = eval_epoch(model, val_loader)
    print(f"Epoch {e}/{EPOCHS}  | Train: {tr:.4f} | Val: {vl:.4f}")

    torch.save(model.state_dict(), "marian_finetuned.pth")

# -----------------------------------------
# SAMPLE TESTS
# -----------------------------------------
print("Sample:", translate_batch(["i love you"])[0])
print("Sample:", translate_batch(["Jayesh, how are you"])[0])
print("Sample:", translate_batch(["Saitama, i will destroy you, i will kill you, please behave properly."])[0])

# -----------------------------------------
# METRICS
# -----------------------------------------
# ----- Validation Metrics -----
y_val_true = list(y_val)
y_val_pred = []

for i in range(0, len(X_val), BATCH_SIZE):
    batch_sentences = X_val[i:i+BATCH_SIZE].tolist()
    y_val_pred.extend(translate_batch(batch_sentences))

val_results = compute_metrics(y_val_true, y_val_pred)
print("Validation Metrics:", val_results)

y_true = list(y_test)
y_pred = []

for i in range(0, len(X_test), BATCH_SIZE):
    batch_sentences = X_test[i:i+BATCH_SIZE].tolist()
    y_pred.extend(translate_batch(batch_sentences))

results = compute_metrics(y_true, y_pred)
print(results)
