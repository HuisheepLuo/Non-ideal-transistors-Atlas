import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import pandas as pd


device = torch.device("cpu")
cpu = torch.device("cpu")

def baseline_value(dir, idx, column_begin=0):
    '''
    Function for read baseline from csv.
    '''
    df = pd.read_csv(dir)
    column_end = column_begin + 200
    y = torch.tensor(df.iloc[idx, column_begin:column_end].astype(np.float32))
    return y

def integrated_gradients(inputs, target_labels, model, loss_fn, optimizer, predict_and_gradients, baseline, steps=50):
    '''
    Using integrated gradients method to attribute the inputs of the network.

    Args:
        inputs[np.ndarray]: input `X` of dataset.
        target_labels[np.ndarray]: labels of dataset.
        model[nn.Module]: network model.
        loss_fn: used loss function for training.
        optimizer: used optimizer for training.
        predict_and_gradients: the import function `pred_and_grad`.
        baseline[np.ndarray]: the baseline input `X'`, should be as same length as `X`.
        step[int]: the step of interpolation.

    Returns:
        integrated_grad[np.ndarray]: the attribution value generated by Integrated Gradient method.
        scaled_input[list(np.ndarray)]: the interpolation from `X` to `X'` .
        
    '''
    if baseline is None:
        baseline = 0 * inputs

    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i+1) / steps) * (inputs - baseline) for i in range(0, steps)]
    grads, _ = predict_and_gradients(scaled_inputs, target_labels, model, loss_fn, optimizer)
    avg_grads = np.average(grads[:-1], axis=0)
    delta_X = (inputs - baseline).detach().squeeze(0).cpu().numpy()
    integrated_grad = delta_X * avg_grads
    return integrated_grad, scaled_inputs

def pred_and_grad(inputs, label_target, model, loss_fn, optimizer):
    '''
    Extracting the grad in the nerual network by Pytorch.

    Args:
        inputs[np.ndarray]: input `X` of dataset.
        target_labels[np.ndarray]: labels of dataset.
        model[nn.Module]: network model.
        loss_fn: used loss function for training.
        optimizer: used optimizer for training.

    Returns:
        integrated_grad[np.ndarray]: the attribution value generated by Integrated Gradient method.
        scaled_input[list(np.ndarray)]: the interpolation from `X` to `X'` .
        
    '''
    grads = []
    for input in inputs:
        input_tensor = input.float().clone().detach().to(device).requires_grad_(True)
        out = model(input_tensor).squeeze()
        label_pred = torch.max(out,-1)[1]
        label_target = torch.tensor(label_target)
        loss = loss_fn(out.to(cpu).unsqueeze(0), label_target.unsqueeze(0))

        #backpropagation
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        # label_pred.backward(torch.ones_like(label_pred))
        # input_tensor = -torch.log10(input_tensor)
        grad = input_tensor.grad.detach().cpu().numpy()
        grads.append(grad)
    grads = np.array(grads)

    return grads, label_pred
    
def visualize(attributions):
    '''
    Sum every 25 adjacent points. For instance, 200 points' attribution will transform into 8 points.

    Args:
        attributions[np.ndarray]: the original attribution value.

    Returns:
        locsum[np.ndarray]: the new attribution value.
    '''
    attributions = np.abs(attributions)
    locsum = []
    loc = 0
    for i in range(len(attributions)):
        if (i+1) % 25 == 0:
            locsum.append(loc)
            loc = 0
        else:
            loc += attributions[i]
    locsum = np.array(locsum) / np.sum(attributions)
    # cumsum = np.cumsum(attributions) / np.sum(attributions)
    return locsum

def atlas_with_attribute(nrow, ncol, attr_matrix, training_data):
    '''
    Plot the Atlas with attribution.

    Args:
        nrow[int]: Rows of the shown atlas
        ncol[int]: Columns of the shown atlas, nrow * ncol ≤ dataset_size
        attr_matrix[np.ndarray or torch.tensor]: the attribution of all dataset
        training_data[point_set class]: training dataset

        
    Returns:
        f, ax: plt common returns
        
    '''

    dataset_size = len(training_data)
    # show_size = nrow * ncol if nrow * ncol < dataset_size else dataset_size
    X = np.array(list(range(100)))
    X_attr0 = np.array([12.5, 37.5, 62.5, 87.5])
    Y_attr_for = np.array([0.25, 0.25, 0.25, 0.25])
    Y_attr_back = np.array([0.75, 0.75, 0.75, 0.75])
    Y_attr = np.concatenate((Y_attr_for, Y_attr_back), axis=0)
    f, ax = plt.subplots(nrow,ncol,sharex=True,sharey=True,figsize=(nrow*2,ncol*0.2))
    for i in range(0, dataset_size):
        attributions = attr_matrix[i]
        Y = training_data[i]['data']
        row, col = i // ncol, i % ncol
        Y = Y.detach().numpy().squeeze()
        Y_forward = Y[0:100]
        Y_backward = Y[::-1][0:100]
        attr_visual = visualize(attributions.squeeze()) 
        attr_forward = attr_visual[0:4]
        attr_backward = attr_visual[::-1][0:4]
        X_attr = np.concatenate((X_attr0, X_attr0), axis=0)
        attr = np.concatenate((attr_forward, attr_backward), axis=0)
        attr_norm = (attr - attr.min(axis=0)) / (attr.max(axis=0) - attr.min(axis=0))
        ax[row,col].plot(X, Y_forward)
        ax[row,col].plot(X, Y_backward)
        ax[row,col].set_xticks([])
        ax[row,col].set_yticks([])
        image = ax[row,col].scatter(X_attr, Y_attr, s=45*attr+5, c=attr_norm, cmap='RdBu_r')
    f.subplots_adjust(hspace=0, wspace=0)
    position = f.add_axes([0.125, 0.05, 0.775, 0.03])
    f.colorbar(image, cax=position, orientation='horizontal', label='Attribution value(a.u.)')
    return f, ax

def atlas_with_similarity(nrow, ncol, attr_matrix, training_data, exp_line):
    '''
    Plot the Atlas with similarity.

    Args:
        nrow[int]: Rows of the shown atlas
        ncol[int]: Columns of the shown atlas
        attr_matrix[np.ndarray or torch.tensor]: the attribution of all dataset
        training_data[point_set class]: training dataset
        exp_line[np.ndarray or torch.tensor]: input line

        
    Returns:
        f, ax: plt common returns
        
    '''
    dataset_size = len(training_data)
    exp_matrix = np.tile(exp_line.detach().numpy(), (dataset_size, 1))
    dataset_matrix = training_data.df.iloc[:,:200].to_numpy()
    V_matrix = (exp_matrix - dataset_matrix) * attr_matrix
    V = np.abs(np.mean(V_matrix, axis=1)).astype('float64')
    s = np.argsort(V)
    # print(s)
    s_2 = list(range(len(s)))
    for i in range(len(s)):
        s_2[s[i]] = i
    # print(s_2)
    s_norm = (s_2 - np.min(s_2)) / (np.max(s_2) - np.min(s_2))
    # print(s_norm)

    f, ax = plt.subplots(1,1,figsize=(nrow*2,ncol*0.2))
    image = ax.pcolor(1 - np.reshape(s_norm, (nrow, ncol)), cmap='RdBu_r', edgecolors='k', linewidths=1)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    position = f.add_axes([0.125, 0.05, 0.775, 0.03])
    f.colorbar(image, cax=position, orientation='horizontal', label='Similarity evaluation value(a.u.)', ticks=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    return f, ax