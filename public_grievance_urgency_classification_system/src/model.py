"""BiLSTM + additive attention with two parallel MLP heads (notebook section 7).

Layer map:
  1 Embedding            nn.Embedding(vocab, emb_dim, padding_idx=0)
  2 BiLSTM               nn.LSTM(bidirectional=True) -> output 2*hidden_dim
  3 Concatenate          built into the LSTM output
  4 Additive attention   Attention(H, attn_dim) returns alpha weights
  5 Weighted sum         torch.bmm inside attention -> context vector
  Heads A & B            MultiLayerFCNN (2 hidden layers, BatchNorm+ReLU)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerFCNN(nn.Module):
    """MLP head: 2 hidden layers of `hidden` neurons, BatchNorm + ReLU + dropout."""

    def __init__(self, in_features, out_features, hidden=128, dropout_rate=0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc_out = nn.Linear(hidden, out_features)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        return self.fc_out(x)


class Attention(nn.Module):
    """Additive (Bahdanau-style) attention over LSTM outputs."""

    def __init__(self, hidden_dim, attn_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, H, mask):
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)
        e = e.masked_fill(mask == 0, -1e9)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), H).squeeze(1)
        return context, alpha


class BiLSTMAttn(nn.Module):
    def __init__(self, cfg, vocab, n_cat, n_urg):
        super().__init__()
        H = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.embedding = nn.Embedding(vocab, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(cfg.emb_dim, cfg.hidden_dim, num_layers=cfg.num_layers,
                            batch_first=True, bidirectional=cfg.bidirectional,
                            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0)
        self.attn = Attention(H, cfg.attn_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.head_cat = MultiLayerFCNN(H, n_cat, hidden=cfg.head_hidden, dropout_rate=cfg.dropout)
        self.head_urg = MultiLayerFCNN(H, n_urg, hidden=cfg.head_hidden, dropout_rate=cfg.dropout)

    def forward(self, x):
        mask = (x != 0).long()
        emb = self.dropout(self.embedding(x))
        H, _ = self.lstm(emb)
        context, alpha = self.attn(H, mask)
        context = self.dropout(context)
        return self.head_cat(context), self.head_urg(context), alpha

    def forward_trace(self, x):
        """Inference-time trace of every stage, for the architecture explainer."""
        mask = (x != 0).long()
        emb = self.embedding(x)
        H, _ = self.lstm(emb)
        context, alpha = self.attn(H, mask)
        cat_logits = self.head_cat(context)
        urg_logits = self.head_urg(context)
        return {
            "mask": mask, "embeddings": emb, "lstm_out": H,
            "alpha": alpha, "context": context,
            "cat_logits": cat_logits, "urg_logits": urg_logits,
        }


def parameter_summary(model: nn.Module):
    """Per-module trainable parameter counts for the model card."""
    rows = []
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        rows.append({"module": name, "class": module.__class__.__name__, "params": n})
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return rows, total
