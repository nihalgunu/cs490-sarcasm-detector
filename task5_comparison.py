import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from collections import Counter
from typing import List, Dict
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

BATCH_SIZE = 8

def get_data(path: str):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

class TextFeaturizer:
    def __init__(self, corpus, max_vocab=5000):
        self.word_to_idx = {'<UNK>': 0}
        word_counts = Counter()
        for text in corpus:
            word_counts.update(re.findall(r'\w+', text.lower()))
        for idx, (word, _) in enumerate(word_counts.most_common(max_vocab - 1), 1):
            self.word_to_idx[word] = idx
        self.vocab_size = len(self.word_to_idx)

    def to_bow(self, text):
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for token in re.findall(r'\w+', text.lower()):
            vec[self.word_to_idx.get(token, 0)] += 1
        return vec

    def to_onehot(self, text):
        vec = np.zeros(self.vocab_size, dtype=np.float32)
        for token in re.findall(r'\w+', text.lower()):
            vec[self.word_to_idx.get(token, 0)] = 1
        return vec

    def to_w2v(self, text, w2v_model):
        tokens = re.findall(r'\w+', text.lower())
        embs = [w2v_model[t] for t in tokens if t in w2v_model]
        if not embs:
            return np.zeros(300, dtype=np.float32)
        return np.mean(embs, axis=0).astype(np.float32)

class SarcasmMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64]):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_mlp(model, X, y, lr, epochs, device):
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                       generator=torch.Generator().manual_seed(SEED))

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    step_losses = []
    for epoch in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
    return step_losses

from transformers import BertTokenizer, BertModel

class SarcasmBERT(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(out.last_hidden_state[:, 0, :])

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer.encode_plus(
            self.data[idx]['headline'], max_length=self.max_len,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label': torch.tensor(self.data[idx]['is_sarcastic'])
        }

def train_bert(model, loader, lr, epochs, device):
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.train()

    step_losses = []
    for epoch in range(epochs):
        for batch in tqdm(loader, desc=f"BERT Epoch {epoch+1}/{epochs}"):
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            loss = criterion(model(ids, mask), labels)
            loss.backward()
            optimizer.step()
            step_losses.append(loss.item())
    return step_losses

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Batch size: {BATCH_SIZE} (same for all models)")

    train_data = get_data('train.jsonl')
    corpus = [d['headline'] for d in train_data]
    labels = np.array([d['is_sarcastic'] for d in train_data])
    print(f"Training samples: {len(train_data)}")

    feat = TextFeaturizer(corpus)
    x_bow = np.array([feat.to_bow(t) for t in corpus])
    x_onehot = np.array([feat.to_onehot(t) for t in corpus])

    print("Loading Word2Vec...")
    import gensim.models
    w2v = gensim.models.KeyedVectors.load_word2vec_format(
        "/Users/nihalgunukula/Downloads/hw1/GoogleNews-vectors-negative300.bin", binary=True)
    x_w2v = np.array([feat.to_w2v(t, w2v) for t in corpus])

    results = {}

    MLP_LR = 0.001
    MLP_EPOCHS = 1

    print(f"\n--- BoW MLP (lr={MLP_LR}, epochs={MLP_EPOCHS}) ---")
    model = SarcasmMLP(x_bow.shape[1])
    results['BoW'] = train_mlp(model, x_bow, labels, MLP_LR, MLP_EPOCHS, device)
    print(f"Steps: {len(results['BoW'])}, Final loss: {results['BoW'][-1]:.4f}")

    print(f"\n--- One-Hot MLP (lr={MLP_LR}, epochs={MLP_EPOCHS}) ---")
    model = SarcasmMLP(x_onehot.shape[1])
    results['One-Hot'] = train_mlp(model, x_onehot, labels, MLP_LR, MLP_EPOCHS, device)
    print(f"Steps: {len(results['One-Hot'])}, Final loss: {results['One-Hot'][-1]:.4f}")

    print(f"\n--- Word2Vec MLP (lr={MLP_LR}, epochs={MLP_EPOCHS}) ---")
    model = SarcasmMLP(300)
    results['Word2Vec'] = train_mlp(model, x_w2v, labels, MLP_LR, MLP_EPOCHS, device)
    print(f"Steps: {len(results['Word2Vec'])}, Final loss: {results['Word2Vec'][-1]:.4f}")

    BERT_LR = 2e-5
    BERT_EPOCHS = 1

    print(f"\n--- BERT (lr={BERT_LR}, epochs={BERT_EPOCHS}) ---")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_dataset = BERTDataset(train_data, tokenizer)
    bert_loader = DataLoader(bert_dataset, batch_size=BATCH_SIZE, shuffle=True,
                            generator=torch.Generator().manual_seed(SEED))
    model = SarcasmBERT()
    results['BERT'] = train_bert(model, bert_loader, BERT_LR, BERT_EPOCHS, device)
    print(f"Steps: {len(results['BERT'])}, Final loss: {results['BERT'][-1]:.4f}")

    plot_comparison(results)

    return results

def plot_comparison(results):
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {'BoW': 'blue', 'One-Hot': 'green', 'Word2Vec': 'orange', 'BERT': 'red'}

    for name, losses in results.items():
        steps = range(1, len(losses) + 1)
        window = max(1, len(losses) // 100)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(losses)+1), smoothed,
                   label=name, color=colors[name], linewidth=2)
        else:
            ax.plot(steps, losses, label=name, color=colors[name], linewidth=2)

    ax.set_xlabel("Gradient Update Step", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Task 5: Loss Curves - MLP (BoW, One-Hot, Word2Vec) vs BERT", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("task5_overlaid.png", dpi=150)
    print("\nSaved: task5_overlaid.png")

    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    max_steps = max(len(l) for l in results.values())
    max_loss = max(max(l) for l in results.values())

    for idx, (name, losses) in enumerate(results.items()):
        ax = axes[idx]
        steps = range(1, len(losses) + 1)

        window = max(1, len(losses) // 100)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(losses)+1), smoothed,
                   color=colors[name], linewidth=2)
        ax.plot(steps, losses, color=colors[name], alpha=0.3, linewidth=0.5)

        ax.set_xlabel("Gradient Update Step")
        ax.set_ylabel("Training Loss")
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max_steps)
        ax.set_ylim(0, min(max_loss, 1.5))

    plt.tight_layout()
    plt.savefig("task5_sidebyside.png", dpi=150)
    print("Saved: task5_sidebyside.png")

if __name__ == "__main__":
    results = main()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for name, losses in results.items():
        print(f"{name:12} | Steps: {len(losses):6} | Init: {losses[0]:.4f} | Final: {losses[-1]:.4f}")
