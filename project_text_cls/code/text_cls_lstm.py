import torch
from datasets import load_dataset
from collections import Counter

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# the max length of the sentence
max_len = 300
# the number of sample per batch
batch_size = 64
# Word vector dimension
embedding_dim = 128
#
hidden_dim = 256
#
n_layers = 2
# learning rate
learning_rate = 0.001
# output types
num_classes = 2
# vocabulary to index
word2idx = {}
# fixed seed for reproducibility (ensures same random numbers each run)
random_seed = 1
# set random seed for PyTorch (CPU)
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, data_list, data_label):
        self.list = data_list
        self.label = data_label

    def __len__(self):
        return len(self.list)

    def __getitem__(self, idx):
        return torch.tensor(self.list[idx], dtype=torch.long), torch.tensor(self.label[idx], dtype=torch.long)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, num_classes):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        embeds = self.dropout(self.embedding(x))
        output, (hidden, cell) = self.lstm(embeds)
        hidden = hidden[-1]
        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        return output

def prepare_training_data():
    # download dataset
    dataset = load_dataset("imdb")

    # spilt the word of sentences
    total_counter = Counter()
    for row in dataset['train']:
        temp = Counter(row['text'].lower().split())
        total_counter.update(temp)
    # build word-frequency table
    vocabulary = {word: count for word, count in total_counter.items()}
    print("vocabulary size ", len(vocabulary))
    #build word-index table
    word2idx["<PAD>"] = 0
    word2idx["<UNK>"] = 1
    for i, word in enumerate(vocabulary.keys()):
        word2idx[word] = 2 + i
    print("now the word2idx mapping is done!")

    print("start to transform the data to vector...")
    total_list = []
    total_label = []
    for row in dataset['train']:
        word_list = []
        text = row['text'].lower().split()
        for word in text:
            idx = word2idx.get(word, word2idx["<UNK>"])
            word_list.append(idx)
        if len(word_list) < max_len:
            word_list += [word2idx["<PAD>"]] * (max_len - len(word_list))
        else:
            word_list = word_list[:max_len]

        total_list.append(word_list)
        total_label.append(row['label'])

    print("total_list: %d, total_label: %d" % (len(total_list), len(total_label)))
    return total_list, total_label

def prepare_testing_data():
    total_list = []
    total_label = []
    dataset = load_dataset("imdb")

    for row in dataset['test']:
        word_list = []
        text = row['text'].lower().split()

        for word in text:
            idx = word2idx.get(word, word2idx["<UNK>"])
            word_list.append(idx)

        if len(word_list) < max_len:
            word_list += [word2idx["<PAD>"]] * (max_len - len(word_list))
        else:
            word_list = word_list[:max_len]

        total_list.append(word_list)
        total_label.append(row['label'])

    return total_list, total_label

def training(network, loader, optimizer, criteon, epoch):
    network.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = network(data)
        loss = criteon(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("training epoch:%d: train_loss: %.3f" % (epoch, total_loss / len(loader)))

def testing(network, loader, criterion):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = network(data)
            test_loss += criterion(output, target)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(loader.dataset)
        print("Test info: test_loss: %.3f, test_acc: %.3f" % (test_loss, correct / len(loader.dataset)))

if __name__ == "__main__":
    train_list, train_label = prepare_training_data()
    train_dataset = TextDataset(train_list, train_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_list, test_label = prepare_testing_data()
    test_dataset = TextDataset(test_list, test_label)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model = LSTM(len(word2idx), embedding_dim, hidden_dim, n_layers, num_classes).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    for epoch in range(1, 20):
        training(model, train_loader, optimizer, criterion, epoch)
        testing(model, test_loader, criterion)