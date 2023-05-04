import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, stru):
        super(MLP, self).__init__()
        self.layers = nn.Sequential()
        for i in range(1, len(stru)):
            self.layers.add_module("linear"+str(i), nn.Linear(stru[i-1], stru[i]))
            if (i < len(stru)-1): self.layers.add_module("relu"+str(i), nn.ReLU())
        
    def forward(self, x):
        self.out = self.layers(x)
        return self.out
    
    def loss(self, out, label, loss_func):
        if out.dim() == 1:
            return loss_func(out.unsqueeze(dim=0), label)
        else:
            return loss_func(out, label)

def train(net, dataloader, loss_func, optimizer, epoches):
    size = len(dataloader.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"training on {device}")
    net = net.to(device)
    train_loss = [0.] * epoches
    train_correct = [0.] * epoches
    for epoch in range(epoches):
        for batch_idx, dict in enumerate(dataloader):
            #forward
            x, label = dict['data'], dict['label']
            x = x.to(device)
            out = net(x).squeeze().cpu()
            label_pred = torch.max(out,-1)[1]
            loss = net.loss(out, label, loss_func)
            train_correct[epoch] += (label_pred == label).sum().item()

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss[epoch] += loss.cpu().item()
            # if (batch_idx % 100 == 0):
            #     print(f"epoch: {epoch} | batch: {batch_idx} | batch average loss: {loss.cpu().item()/len(x)}")

        print(f"epoch: {epoch} | epoch average correct: {train_correct[epoch]/size*100:.02f}% | epoch average loss: {train_loss[epoch]/size:.04f}")




