import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from random import shuffle
from matplotlib import pyplot


class myDataset(Dataset):
    def __init__(self, data, labels):
        self.data = []
        for sub_seq, label in zip(data, labels):
            self.data.append([[int(digit) for digit in sub_seq], [label]])

    def __getitem__(self, index):
        elem = torch.FloatTensor(self.data[index][0])
        label = torch.LongTensor(self.data[index][1])
        return elem, label

    def __len__(self):
        return len(self.data)


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.relu = nn.ReLU()
        layers = [input_size] + hidden_sizes + [num_classes]
        self.fc = []
        for i in range(len(layers) - 1):
            self.fc.append(nn.Linear(layers[i], layers[i + 1]))
        self.fc = nn.ModuleList(self.fc)

    def forward(self, x):
        out = x
        for i in range(len(self.fc) - 1):
            out = self.fc[i](out)
            out = self.relu(out)
        out = self.fc[-1](out)
        return out


def read_data(chunk_size, file):
    sequences = []
    with open(file) as f:
        for sequence in f:
            sequences.append(sequence)
    X = []
    Y = []
    for sequence in sequences:
        for i in range(len(sequence)-chunk_size-1):
            sub_seq = sequence[i:(i+chunk_size)]
            sub_seq = [int(a) for a in sub_seq]
            X.append(sub_seq)
            Y.append(int(sequence[i + chunk_size]))
    combined = list(zip(X, Y))
    shuffle(combined)
    X, Y = zip(*combined)
    return X, Y


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_num = 0

# Hyper-parameters
input_sizes = [7, 8]
hidden_sizes = [10, 15]
layers_numbers = [2, 4]
learning_rates = [0.01]
num_classes = 2
num_epochs = 10000
#batch_size = 100

# Loss and optimizer
criterion = nn.CrossEntropyLoss()

for input_size in input_sizes:
    train, train_labels = read_data(input_size, "data.txt")
    test, test_labels = read_data(input_size, "test.txt")


    print(f'training length:{len(train)} test lenght:{len(test)}')

    train_dataset = myDataset(train, train_labels)

    test_dataset = myDataset(test, test_labels)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               # batch_size=batch_size,
                                               batch_size=len(train_dataset),
                                               shuffle=False)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              # batch_size=batch_size,
                                              batch_size=len(test_dataset),
                                              shuffle=False)

    for sequences, labels in train_loader:
        print("sending training data to device")
        sequences = sequences.to(device)
        labels = labels.reshape(labels.size()[0])
        labels = labels.to(device)
    for test_data, test_labels in test_loader:
        print("sending testing data to device")
        labels_cpu = test_labels
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)

    for layers_number in layers_numbers:
        for hidden_size in hidden_sizes:
            for learning_rate in learning_rates:
                print(f'arch: {[hidden_size]*layers_number} r: {learning_rate} in:{input_size}')
                model = NeuralNet(input_size, [hidden_size] * layers_number, num_classes).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                loss_hist = []
                test_hist = []

                for epoch in range(num_epochs):
                    # Forward pass
                    outputs = model(sequences)
                    loss = criterion(outputs, labels)

                    # Backward and optimize
                    if epoch % int(num_epochs / 300) == 0:
                        loss_hist.append(loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if epoch % int(num_epochs / 300) == 0:
                        with torch.no_grad():
                            correct = 0
                            total = 0
                            if True:
                                outputs = model(test_data)
                                total += test_labels.size(0)
                                predicted = []
                                for sub_list in outputs.data.tolist():
                                    predicted.append([sub_list.index(max(sub_list))])
                                predicted = torch.FloatTensor(predicted)
                                labels_cpu = labels_cpu.type(torch.FloatTensor)
                                correct += (predicted == labels_cpu).sum().item()
                                test_hist.append(correct / total)
                    if epoch % int(num_epochs / 10) == 0:
                        print(f'epoch: {epoch}/{num_epochs} test_acc: {correct / total}')
                    if epoch % int(num_epochs / 10) == 0:
                        torch.save(model.state_dict(), f'model_{model_num}_c:{correct/total}.pt')

                pyplot.plot(loss_hist)
                pyplot.plot(test_hist)
                pyplot.suptitle(f'hid:{hidden_size} layers:{layers_number} lr:{learning_rate}')
                pyplot.title(f'e:{num_epochs} in:{input_size}')
                pyplot.savefig(f'cor: {int(100*test_hist[-1])} nr:{model_num}.jpeg')
                pyplot.close()
                model_num += 1

print('end')
