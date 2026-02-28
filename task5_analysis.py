"""
Task 5: Experimental Analysis
Compare HW1 models (BoW, One-Hot, Word2Vec with MLP) vs HW2 model (BERT)
Loss curves recorded at each gradient update step
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from collections import Counter
from typing import List, Dict, Tuple, Union
from tqdm import tqdm
import random

# Set seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ========================================
# Data Loading
# ========================================
def get_data(path: str) -> List[Dict[str, Union[str, int]]]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# ========================================
# HW1: Text Featurizer (simplified - no word2vec file needed for BoW/OneHot)
# ========================================
class TextFeaturizer:
    def __init__(self, corpus: List[str], max_vocab_size: int = 5000):
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.build_vocab(corpus, max_vocab_size)
        self.emb_dim = 300  # Word2Vec dimension

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def build_vocab(self, corpus: List[str], max_vocab_size: int = 5000) -> None:
        self.word_to_idx['<UNK>'] = 0
        self.idx_to_word[0] = '<UNK>'

        word_counts = Counter()
        for text in corpus:
            tokens = self._tokenize(text)
            word_counts.update(tokens)

        most_common = word_counts.most_common(max_vocab_size - 1)

        for idx, (word, _) in enumerate(most_common, start=1):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

    def to_one_hot(self, text: str) -> np.ndarray:
        vocab_size = len(self.word_to_idx)
        vector = np.zeros(vocab_size, dtype=np.float32)
        tokens = self._tokenize(text)
        for token in tokens:
            idx = self.word_to_idx.get(token, 0)
            vector[idx] = 1
        return vector

    def to_bow(self, text: str) -> np.ndarray:
        vocab_size = len(self.word_to_idx)
        vector = np.zeros(vocab_size, dtype=np.float32)
        tokens = self._tokenize(text)
        for token in tokens:
            idx = self.word_to_idx.get(token, 0)
            vector[idx] += 1
        return vector

    def to_word2vec(self, text: str, w2v_model) -> np.ndarray:
        tokens = self._tokenize(text)
        embeddings = []
        for token in tokens:
            if token in w2v_model:
                embeddings.append(w2v_model[token])

        if len(embeddings) == 0:
            return np.zeros(self.emb_dim, dtype=np.float32)

        return np.mean(embeddings, axis=0).astype(np.float32)

# ========================================
# HW1: MLP Model
# ========================================
class SarcasmMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int) -> None:
        super(SarcasmMLP, self).__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

# ========================================
# HW1: Batch Training Loop (records loss per step)
# ========================================
def train_mlp_batched(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    lr: float,
    epochs: int,
    device: torch.device,
    seed: int = SEED
) -> List[float]:
    """Train MLP with mini-batch SGD, return loss at each gradient step."""

    # Create dataset and dataloader with fixed seed
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    generator = torch.Generator().manual_seed(seed)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)
    model.train()

    step_losses = []

    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())

    return step_losses

# ========================================
# HW2: BERT Model
# ========================================
from transformers import BertTokenizer, BertModel

class SarcasmBERT(nn.Module):
    def __init__(self):
        super(SarcasmBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits

class SarcasmDatasetBERT(Dataset):
    def __init__(self, data: List[Dict], tokenizer, max_length: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        text = item['headline']
        label = item['is_sarcastic']

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor([label], dtype=torch.long)
        }

def train_bert_batched(
    model: nn.Module,
    dataloader: DataLoader,
    lr: float,
    epochs: int,
    device: torch.device
) -> List[float]:
    """Train BERT, return loss at each gradient step."""

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    step_losses = []

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f"BERT Epoch {epoch+1}/{epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].squeeze().to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            step_losses.append(loss.item())

    return step_losses

# ========================================
# Main Experimental Analysis
# ========================================
def run_task5_experiments():
    print("=" * 60)
    print("Task 5: Experimental Analysis")
    print("=" * 60)

    # Configuration - SAME for all models
    BATCH_SIZE = 8

    # Different hyperparameters allowed
    MLP_LR = 0.001
    MLP_EPOCHS = 10  # More epochs for MLP since it's faster
    BERT_LR = 2e-5
    BERT_EPOCHS = 1  # Fewer epochs for BERT (takes longer)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    train_data = get_data('train.jsonl')
    train_corpus = [str(d['headline']) for d in train_data]
    train_labels = np.array([int(d['is_sarcastic']) for d in train_data])
    print(f"Loaded {len(train_data)} training samples")

    # Build featurizer
    print("\nBuilding featurizer...")
    featurizer = TextFeaturizer(train_corpus)

    # Prepare features
    print("Preparing BoW features...")
    x_bow = np.array([featurizer.to_bow(text) for text in train_corpus])

    print("Preparing One-Hot features...")
    x_onehot = np.array([featurizer.to_one_hot(text) for text in train_corpus])

    # For Word2Vec, we need the actual model
    print("Loading Word2Vec model...")
    try:
        import gensim.models
        w2v_path = "/Users/nihalgunukula/Downloads/hw1/GoogleNews-vectors-negative300.bin"
        w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
            w2v_path, binary=True
        )
        print("Preparing Word2Vec features...")
        x_w2v = np.array([featurizer.to_word2vec(text, w2v_model) for text in train_corpus])
        has_w2v = True
    except Exception as e:
        print(f"Word2Vec not available: {e}")
        print("Using random embeddings as placeholder for Word2Vec...")
        x_w2v = np.random.randn(len(train_corpus), 300).astype(np.float32)
        has_w2v = False

    results = {}

    # ========================================
    # Train MLP with BoW
    # ========================================
    print(f"\n--- Training MLP with BoW (lr={MLP_LR}, epochs={MLP_EPOCHS}) ---")
    model_bow = SarcasmMLP(x_bow.shape[1], [128, 64], 2)
    losses_bow = train_mlp_batched(model_bow, x_bow, train_labels, BATCH_SIZE, MLP_LR, MLP_EPOCHS, device)
    results['BoW'] = losses_bow
    print(f"Total steps: {len(losses_bow)}, Final loss: {losses_bow[-1]:.4f}")

    # ========================================
    # Train MLP with One-Hot
    # ========================================
    print(f"\n--- Training MLP with One-Hot (lr={MLP_LR}, epochs={MLP_EPOCHS}) ---")
    model_onehot = SarcasmMLP(x_onehot.shape[1], [128, 64], 2)
    losses_onehot = train_mlp_batched(model_onehot, x_onehot, train_labels, BATCH_SIZE, MLP_LR, MLP_EPOCHS, device)
    results['One-Hot'] = losses_onehot
    print(f"Total steps: {len(losses_onehot)}, Final loss: {losses_onehot[-1]:.4f}")

    # ========================================
    # Train MLP with Word2Vec
    # ========================================
    print(f"\n--- Training MLP with Word2Vec (lr={MLP_LR}, epochs={MLP_EPOCHS}) ---")
    model_w2v = SarcasmMLP(x_w2v.shape[1], [128, 64], 2)
    losses_w2v = train_mlp_batched(model_w2v, x_w2v, train_labels, BATCH_SIZE, MLP_LR, MLP_EPOCHS, device)
    results['Word2Vec'] = losses_w2v
    print(f"Total steps: {len(losses_w2v)}, Final loss: {losses_w2v[-1]:.4f}")

    # ========================================
    # Train BERT
    # ========================================
    print(f"\n--- Training BERT (lr={BERT_LR}, epochs={BERT_EPOCHS}) ---")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_dataset = SarcasmDatasetBERT(train_data, tokenizer)
    generator = torch.Generator().manual_seed(SEED)
    bert_loader = DataLoader(bert_dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator)

    model_bert = SarcasmBERT()
    losses_bert = train_bert_batched(model_bert, bert_loader, BERT_LR, BERT_EPOCHS, device)
    results['BERT'] = losses_bert
    print(f"Total steps: {len(losses_bert)}, Final loss: {losses_bert[-1]:.4f}")

    return results

def plot_loss_curves(results: Dict[str, List[float]], save_path: str = "task5_loss_curves.png"):
    """Plot loss curves for all models."""

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = {
        'BoW': 'blue',
        'One-Hot': 'green',
        'Word2Vec': 'orange',
        'BERT': 'red'
    }

    for name, losses in results.items():
        steps = range(1, len(losses) + 1)
        # Apply smoothing for better visualization
        window = min(50, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_steps = range(window, len(losses) + 1)
            ax.plot(smooth_steps, smoothed, label=f"{name} (smoothed)",
                   color=colors.get(name, 'gray'), linewidth=2, alpha=0.9)
        ax.plot(steps, losses, color=colors.get(name, 'gray'), alpha=0.2, linewidth=0.5)

    ax.set_xlabel("Gradient Update Step", fontsize=14)
    ax.set_ylabel("Training Loss", fontsize=14)
    ax.set_title("Loss Curves: MLP (BoW, One-Hot, Word2Vec) vs BERT", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")

    # Also create side-by-side plots with same axes
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Find global min/max for consistent axes
    all_losses = [l for losses in results.values() for l in losses]
    y_max = np.percentile(all_losses, 95)  # Use 95th percentile to avoid outliers
    x_max = max(len(losses) for losses in results.values())

    for idx, (name, losses) in enumerate(results.items()):
        ax = axes[idx]
        steps = range(1, len(losses) + 1)

        window = min(50, len(losses) // 10)
        if window > 1:
            smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
            smooth_steps = range(window, len(losses) + 1)
            ax.plot(smooth_steps, smoothed, color=colors.get(name, 'gray'), linewidth=2)
        ax.plot(steps, losses, color=colors.get(name, 'gray'), alpha=0.3, linewidth=0.5)

        ax.set_xlabel("Gradient Update Step", fontsize=12)
        ax.set_ylabel("Training Loss", fontsize=12)
        ax.set_title(f"{name}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_max)
        ax.set_ylim(0, y_max)

    plt.tight_layout()
    plt.savefig("task5_loss_curves_sidebyside.png", dpi=150, bbox_inches='tight')
    print("Side-by-side plot saved to task5_loss_curves_sidebyside.png")

def print_analysis(results: Dict[str, List[float]]):
    """Print analysis of results."""
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for name, losses in results.items():
        print(f"\n{name}:")
        print(f"  Total gradient steps: {len(losses)}")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Min loss: {min(losses):.4f}")
        print(f"  Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.1f}%")

if __name__ == "__main__":
    results = run_task5_experiments()
    plot_loss_curves(results)
    print_analysis(results)

    # Save raw results
    with open("task5_results.json", "w") as f:
        json.dump({k: v for k, v in results.items()}, f)
    print("\nRaw results saved to task5_results.json")
