import os
import random
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from transformer import MiniTransformerLM
from torch.optim.lr_scheduler import ReduceLROnPlateau

best_valid_loss = float('inf')
patience = 5
patience_counter = 0

word2vec = {}
vec2word = {}

block_size = 128
batch_size = 16
embedding_dim = 64
n_heads = 4
n_layers = 2
hidden_dim = 128
learning_rate = 5e-5
dropout=0.4
weight_decay=0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, data_ids, block_size):
        self.data_ids = data_ids
        self.block_size = block_size

    def __len__(self):
        return len(self.data_ids) - block_size

    def __getitem__(self, idx):
        x = self.data_ids[idx : idx + self.block_size]
        y = self.data_ids[idx + 1 : idx + self.block_size + 1]

        x_tensor = torch.tensor(x, dtype=torch.long)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

def download_tiny_shakespeare(save_dir="data"):
    os.makedirs(save_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    save_path = os.path.join(save_dir, "input.txt")

    if not os.path.exists(save_path):
        urllib.request.urlretrieve(url, save_path)
        print("download done!")
    else:
        print("file exists!")

    return save_path


def prepare_data():
    save_path = download_tiny_shakespeare()
    word = {}
    full_content = ""
    with open(save_path, "r") as f:
        full_content = f.read()
        for char in full_content:
            word[char]  = word.get(char, 0) + 1

    sorted_items = sorted(word.items(), key=lambda item: item[1], reverse=True)
    sorted_words = dict(sorted_items)
    for i, ch in enumerate(sorted_words):
        word2vec[ch] = i
        vec2word[i] = ch

    input = []
    for char in full_content:
        input.append(word2vec[char])

    return input

def get_batch_data(full_data):
    x_batch, y_batch = np.zeros(batch_size), np.zeros(batch_size)
    for i in range(batch_size):
        random_index = random.randint(0, len(full_data) - block_size - 1)
        x = full_data[random_index: random_index + batch_size]
        y = full_data[random_index + 1: random_index + batch_size + 1]
        x_batch = np.concatenate((x_batch, x), axis=0)
        y_batch = np.concatenate((y_batch, y), axis=0)

    return x_batch, y_batch

def training(model, data_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        B, T, C = output.shape
        loss =nn.CrossEntropyLoss()(output.view(B*T, C), target.view(B*T))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("training, epoch %d, loss %f" % (epoch, total_loss/len(data_loader)))

    return total_loss/len(data_loader)

def validation(model, data_loader, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            B, T, C = output.shape
            loss = nn.CrossEntropyLoss()(output.view(B*T, C), target.view(B*T))
            total_loss += loss.item()

    print("validation, epoch %d, loss %f" % (epoch, total_loss/len(data_loader)))

    return total_loss/len(data_loader)

if __name__ == "__main__":
    print("start to prepare data..")
    data_ids = prepare_data()
    print("complete to prepare data!")

    train_ids = data_ids[:int(len(data_ids) * 0.9)]
    valid_ids = data_ids[int(len(data_ids) * 0.9):]

    print("build data set and data loader...")
    train_dataset = TextDataset(train_ids, block_size)
    valid_dataset = TextDataset(valid_ids, block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    print("complte to build data set and data loader!")

    vocal_size = len(word2vec)
    model = MiniTransformerLM(vocal_size, embedding_dim, n_heads, n_layers, max_len=block_size, hidden_dim=hidden_dim, dropout=dropout)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    train_losses = []
    valid_losses = []
    epoches = []
    for epoch in range(1, 50):
        train_loss = training(model, train_loader, optimizer, epoch)
        valid_loss = validation(model, valid_loader, epoch)

        scheduler.step(valid_loss)

        # early stopping check
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # save the best model
            torch.save(model.state_dict(), 'result/best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        epoches.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

    plt.plot(epoches, train_losses, color='red', label='train')
    plt.plot(epoches, valid_losses, color='blue', label='valid')
    plt.legend()

    plt.xlabel('epoch'), plt.ylabel('loss'), plt.title('Training Progress')
    plt.savefig('result/training_result.png')
    print("picture has saved as training_result.png")
    plt.close()