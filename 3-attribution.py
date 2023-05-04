import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import os

from network.classify import MLP
from network.dataset import point_dataset
from network.inte_grad import integrated_gradients, pred_and_grad, baseline_value, atlas_with_attribute, atlas_with_similarity

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # fix errors for multiple libiomp5md.dll file

device = torch.device("cpu") # CUDA is not necessary for non-training mode

save_dir = 'results\\models\\'
save_model_name = 'epoch_100_stru_200_256_4.pth'
stru = [200, 256, 4]
model = MLP(stru).to(device)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model.load_state_dict(torch.load(save_dir+save_model_name))
model.eval()

random.seed(20)

out_dir = 'results/attr/'
# -------------------- train dataset and labels ----------------------- #
train_label_file='dataset\\train_set.csv'
training_data = point_dataset(label_file=train_label_file)
dataset_size = len(training_data)
baseline = baseline_value('dataset/baseline.csv', 0)

# -------------------- experimental data ---------------------------- #
exp_line = baseline_value('dataset/exp_delta.csv', 0)


# -------------------- classification ---------------------------- #
class_name = ['delta', 'P0', 'phi', 'pi']
class_idx_bymodel = torch.max(model(exp_line), -1)[1].detach().numpy()
print(f'Model result: {class_name[class_idx_bymodel]}')


# -------------------- attribution ---------------------------- #
is_load = False # If the 'attr.npy' has been generated, this value shall be True.
attr_matrix = np.zeros((dataset_size, 200))
out_np = out_dir+'attr_matrix_baseline_ideal.npy'
if not is_load:
    for i in range(0, dataset_size):
        data_input = training_data[i]['data'].squeeze()
        grads, labels = pred_and_grad([data_input], training_data[i]['label'], model, loss_fn, optimizer)
        attributions, scaled = integrated_gradients(data_input, labels, model, loss_fn, optimizer, pred_and_grad, baseline)
        attr_matrix[i] = attributions
        print(f'{i}/{dataset_size}')
    np.save(out_np, attr_matrix)
else:
    attr_matrix = np.load(out_np)

exp_matrix = np.tile(exp_line.detach().numpy(), (dataset_size, 1))
dataset_matrix = training_data.df.iloc[:,:200].to_numpy()
V_matrix = (exp_matrix - dataset_matrix) * attr_matrix
attr_idx = np.argmin(np.abs(np.mean(V_matrix, axis=1)))

# print(np.abs(np.mean(V_matrix, axis=1)))
print(f'attr_idx: {attr_idx}.')

# -------------------- plot the fitting curves ---------------------------- #
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.size'] = 16
plt.figure(figsize=(6, 4))
plt.plot(exp_matrix[0], label='exp')
plt.plot(dataset_matrix[attr_idx], label='attr')
plt.legend(frameon=False)

# -------------------- plot the Atlas with the attribution & similarity ---------------------------- #
nrow, ncol = 8, 25
atlas_with_attribute(nrow, ncol, attr_matrix, training_data)
atlas_with_similarity(nrow, ncol, attr_matrix, training_data, exp_line)
plt.show()
