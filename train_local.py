import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
import numpy as np
import os

# 1. DETECT MAC ACCELERATOR (M1/M2/M3)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✅ SUCCESS: Using Apple Silicon (MPS) for high-speed training!")
else:
    device = torch.device("cpu")
    print("⚠️ MPS not found, using CPU.")

# 2. LOAD & PREPARE DATA
file_path = "ready_for_mbert.csv"
if not os.path.exists(file_path):
    print(f"❌ ERROR: {file_path} not found in this folder!")
    exit()

df = pd.read_csv(file_path)
df['Review_Text'] = df['Review_Text'].astype(str)

# ADVISOR FIX: Rebalance classes before training to avoid 'Neutral only' bias
print("⚖️ Rebalancing classes for better accuracy...")
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(df[['Review_Text']], df['Sentiment_Level'])
df_balanced = pd.DataFrame(X_resampled, columns=['Review_Text'])
df_balanced['Sentiment_Level'] = y_resampled

# 3. TOKENIZATION
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

class BaguioDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

# Split (80% Train, 20% Test)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_balanced['Review_Text'].tolist(), 
    df_balanced['Sentiment_Level'].tolist(), 
    test_size=0.2,
    random_state=42
)

train_encodings = tokenizer(train_texts, padding=True, truncation=True, max_length=128)
val_encodings = tokenizer(val_texts, padding=True, truncation=True, max_length=128)

# 4. INITIALIZE MODEL
model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3).to(device)

# 5. TRAINING ARGUMENTS (Optimized for MacBook Air RAM)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=4,             # 4 epochs is good for local training
    per_device_train_batch_size=8,  # Lower batch size prevents Mac from crashing
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    use_mps_device=True if torch.backends.mps.is_available() else False
)

# Custom Metrics for Chapter 4
from sklearn.metrics import accuracy_score, f1_score
def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {
        'accuracy': accuracy_score(p.label_ids, preds),
        'f1': f1_score(p.label_ids, preds, average='weighted')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=BaguioDataset(train_encodings, train_labels),
    eval_dataset=BaguioDataset(val_encodings, val_labels),
    compute_metrics=compute_metrics
)

print("🚀 Starting fine-tuning... Please keep your Mac plugged into a charger.")
trainer.train()

# 6. SAVE THE BRAIN
model.save_pretrained("./baguio_mbert_standalone")
tokenizer.save_pretrained("./baguio_mbert_standalone")
print("✅ DONE! Model saved in './baguio_mbert_standalone'")