import copy
import math
import time

import numpy as np
import torch
from glycowork.ml.model_training import EarlyStopping, disable_running_stats, enable_running_stats
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, matthews_corrcoef, mean_squared_error, mean_absolute_error, \
    label_ranking_average_precision_score, ndcg_score


def sigmoid(x):
       return 1 / (1 + math.exp(-x))


def train_model(model, dataloaders, criterion, optimizer,
                scheduler, num_epochs=25, patience=50,
                mode='classification', mode2='multi'):
    """trains a deep learning model on predicting glycan properties\n
    | Arguments:
    | :-
    | model (PyTorch object): graph neural network (such as SweetNet) for analyzing glycans
    | dataloaders (PyTorch object): dictionary of dataloader objects with keys 'train' and 'val'
    | criterion (PyTorch object): PyTorch loss function
    | optimizer (PyTorch object): PyTorch optimizer
    | scheduler (PyTorch object): PyTorch learning rate decay
    | num_epochs (int): number of epochs for training; default:25
    | patience (int): number of epochs without improvement until early stop; default:50
    | mode (string): 'classification', 'multilabel', or 'regression'; default:classification
    | mode2 (string): further specifying classification into 'multi' or 'binary' classification;default:multi\n
    | Returns:
    | :-
    | Returns the best model seen during training
    """
    since = time.time()
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100.0
    epoch_mcc = 0
    if mode != 'regression':
        best_acc = 0.0
    else:
        best_acc = 100.0
    val_losses = []
    val_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = []
            running_acc = []
            running_mcc = []
            for data in dataloaders[phase]:
                try:
                    x, y, edge_index, prot, batch = data.x, data.y, data.edge_index, data.train_idx, data.batch
                    prot = prot.view(max(batch) + 1, -1).cuda()
                except:
                    x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
                x = x.cuda()
                y = y.cuda()
                edge_index = edge_index.cuda()
                batch = batch.cuda()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    enable_running_stats(model)
                    try:
                        pred = model(prot, x, edge_index, batch)
                        loss = criterion(pred, y.view(-1, 1))
                    except:
                        pred = model(x, edge_index, batch)
                        loss = criterion(pred, y)

                    if phase == 'train':
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        disable_running_stats(model)
                        try:
                            criterion(model(prot, x, edge_index, batch), y.view(-1, 1)).backward()
                        except:
                            criterion(model(x, edge_index, batch), y).backward()
                        optimizer.second_step(zero_grad=True)

                running_loss.append(loss.item())
                if mode == 'classification':
                    if mode2 == 'multi':
                        pred2 = np.argmax(pred.cpu().detach().numpy(), axis=1)
                    else:
                        pred2 = [sigmoid(x) for x in pred.cpu().detach().numpy()]
                        pred2 = [np.round(x) for x in pred2]
                    running_acc.append(accuracy_score(
                        y.cpu().detach().numpy().astype(int), pred2))
                    running_mcc.append(matthews_corrcoef(y.detach().cpu().numpy(), pred2))
                elif mode == 'multilabel':
                    running_acc.append(label_ranking_average_precision_score(y.cpu().detach().numpy().astype(int),
                                                                             pred.cpu().detach().numpy()))
                    running_mcc.append(ndcg_score(y.cpu().detach().numpy().astype(int),
                                                  pred.cpu().detach().numpy()))
                else:
                    running_acc.append(mean_squared_error(y.cpu().detach().numpy(), pred.cpu().detach().numpy()))

            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_acc)
            if mode != 'regression':
                epoch_mcc = np.mean(running_mcc)
            else:
                epoch_mcc = 0
            if mode == 'classification':
                print('{} Loss: {:.4f} Accuracy: {:.4f} MCC: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc))
            elif mode == 'multilabel':
                print('{} Loss: {:.4f} LRAP: {:.4f} NDCG: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc, epoch_mcc))
            else:
                print('{} Loss: {:.4f} MSE: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if mode != 'regression':
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
            else:
                if phase == 'val' and epoch_acc < best_acc:
                    best_acc = epoch_acc
            if phase == 'val':
                val_losses.append(epoch_loss)
                val_acc.append(epoch_acc)
                early_stopping(epoch_loss, model)
                scheduler.step()

        if early_stopping.early_stop:
            print("Early stopping")
            break
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    if mode == 'classification':
        print('Best val loss: {:4f}, best Accuracy score: {:.4f}'.format(best_loss, best_acc))
    elif mode == 'multilabel':
        print('Best val loss: {:4f}, best LRAP score: {:.4f}'.format(best_loss, best_acc))
    else:
        print('Best val loss: {:4f}, best MSE score: {:.4f}'.format(best_loss, best_acc))
    model.load_state_dict(best_model_wts)

    ## plot loss & score over the course of training
    fig, ax = plt.subplots(nrows=2, ncols=1)
    plt.subplot(2, 1, 1)
    plt.plot(range(epoch + 1), val_losses)
    plt.title('Model Training')
    plt.ylabel('Validation Loss')
    plt.legend(['Validation Loss'], loc='best')

    plt.subplot(2, 1, 2)
    plt.plot(range(epoch + 1), val_acc)
    plt.xlabel('Number of Epochs')
    if mode == 'classification':
        plt.ylabel('Validation Accuracy')
        plt.legend(['Validation Accuracy'], loc='best')
    elif mode == 'multilabel':
        plt.ylabel('Validation LRAP')
        plt.legend(['Validation LRAP'], loc='best')
    else:
        plt.ylabel('Validation MSE')
        plt.legend(['Validation MSE'], loc='best')
    return model
