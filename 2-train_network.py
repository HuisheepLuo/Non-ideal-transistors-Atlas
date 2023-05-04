import os
import sys
sys.path.append(os.getcwd())

import torch
from torch.utils.data import DataLoader
from network.classify import MLP, train
from network.dataset import point_dataset

seed = 34658746980
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

batch_size = 1
data = point_dataset("dataset\\train_set.csv")
dataloader = DataLoader(data, batch_size=batch_size)

stru = [200, 256, 4]
epoches = 200
cur_path = os.path.dirname(__file__)
model = MLP(stru)
loss_fn = torch.nn.CrossEntropyLoss()
opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

train(model, dataloader, loss_fn, opt, epoches)

torch.save(model.state_dict(), cur_path+'\\results\\models\\epoch_'+str(epoches)+'_stru_'+'_'.join(str(i) for i in stru)+'.pth')
