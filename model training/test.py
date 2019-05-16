import torch
import torch.nn as nn

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



num_classes = 2

input_size = 7

model = NeuralNet(input_size, [15] * 2, num_classes)
model.load_state_dict(torch.load('./new_model.pt'))
torch.onnx.export(model, torch.FloatTensor(torch.rand([1, input_size])), f='new_model.onnx', input_names=["in"], output_names=["out"])


# exit(0)

with torch.no_grad():
    while True:
        s = ''
        inp = input("inp: ")
        if inp == 'x':
            break
        s += inp
        if len(s) < input_size + 1:
            continue
        print(len(s))
        l = [s[i:(i + input_size + 1)] for i in range(len(s) - 1 - input_size)]
        seq = []
        lab = []
        for e in l:
            seq.append([int(a) for a in e[:-1]])
            lab.append(int(e[-1]))
        print(len(seq))
        x = torch.FloatTensor(seq)
        y = lab

        output = model(x)
        c = 0
        t = 0
        for i, pre in enumerate(output.data.tolist()):
            p = pre.index(max(pre))
            print(f"it was: {lab[i]} prediction:{p}")
            t += 1
            if lab[i] == p:
                c += 1
        print(f"correct: {100 * c / t}%")

print('end')
