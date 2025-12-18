import math
import torch
from torch import nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, base=10000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1) # [max_len, 1]
        self.output = (position /
                  torch.pow(base, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model))
        pe[:, 0::2] = torch.sin(self.output)  # even
        pe[:, 1::2] = torch.cos(self.output)  # odd

        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        T = x.size(1)
        return self.pe[:, :T, :]

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, hidden_dim)
        self.linear_2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        return self.linear_2(self.dropout(F.relu(self.linear_1(input))))

class multihead_attention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(multihead_attention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)  # Q
        self.w_k = nn.Linear(d_model, d_model)  # K
        self.w_v = nn.Linear(d_model, d_model)  # V
        self.W_o = nn.Linear(d_model, d_model)  # output

    def forward(self, x, mask=None):
        """
        :param x: [batch_size, seq_len, d_model]
        :param mask: [1,1,deq_len,seq_len]
        """
        batch_size, seq_len, d_model = x.size()

        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # [batch_size, num_heads, seq_len, head_dim]

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        p = F.softmax(scores, dim=-1)
        output = torch.matmul(p, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.attention = multihead_attention(d_model, n_heads)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, hidden_dim, dropout)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.layer_norm_1(x + self.dropout(self.attention(x, mask)))
        x = self.layer_norm_2(x + self.dropout(self.ffn(x)))

        return x

class MiniTransformerLM(nn.Module):
    """
    vocal_size: char number
    d_model: embedding dimension
    n_heads: number of attention heads
    max_len: maximum sequence length
    hidden_dim: hidden dimension in FFN
    """
    def __init__(self, vocal_size, d_model, n_heads,n_layers, max_len, hidden_dim, dropout=0.1):
        super(MiniTransformerLM, self).__init__()
        self.vocal_size = vocal_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.embedding = nn.Embedding(vocal_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.blocks = nn.Sequential()
        for i in range(n_layers):
            self.blocks.add_module("block" + str(i), TransformerBlock(d_model, hidden_dim, n_heads, dropout=0.1))

        self.lm_head = nn.Linear(d_model, vocal_size)

    def forward(self, x):
        batch_size, seq_len = x.size()

        token_emb = self.embedding(x)
        pos_emb = self.positional_encoding(token_emb)
        h = token_emb + pos_emb

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0) #[1, 1, seq_len, seq_len]
        for block in self.blocks:
            h = block(h, mask)

        logits = self.lm_head(h)
        return logits

if __name__ == "__main__":
    vocab_size = 65
    d_model = 128
    n_heads = 4
    n_layers = 4
    block_size = 64
    ff_hidden_dim = 256

    model = MiniTransformerLM(
        vocal_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_len=block_size,
        hidden_dim=ff_hidden_dim,
        dropout=0.2,
    )

    B = 2
    T = 16
    x = torch.randint(0, vocab_size, (B, T))  # simulate [B, T] 的 token id

    logits = model(x)