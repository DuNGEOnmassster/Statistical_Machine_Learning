import os

import cv2
import kornia
import numpy
from torch import Tensor
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def draw_loss(train_counter, train_losses, test_counter, test_losses):
    # draw loss curve
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    print(f"test_counter = {len(test_counter)}, test_losses = {len(test_losses)}")
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Valid Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(f"./figure/resnet50_loss.png")
    plt.show()


def draw_acc(total_epochs, train_acc, test_acc):
    # draw acc curve
    fig = plt.figure()
    plt.plot(total_epochs,train_acc, color='red')
    plt.plot(total_epochs, test_acc, color='green')
    plt.legend(['Train Acc', 'Valid Acc'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood acc')
    plt.savefig(f"./figure/resnet50_acc.png")
    plt.show()


def fit_one_epoch(model_train, model, opt_model, scheduler, epoch, epoch_step, epoch_step_val, train_gen, val_gen, Epoch,
                  cuda, save_period, save_dir, train_losses, train_acc, train_counter, test_losses, test_acc):
    loss = 0
    train_set = set()
    print('Start Train')
    criterion = nn.CrossEntropyLoss()
    if cuda:
        criterion = criterion.cuda()
    pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    acc = 0
    for iteration, batch in enumerate(train_gen):
        if iteration >= epoch_step:
            break

        images, label = batch[0], batch[1]  # image (B,C,H,W)   label (B)
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                label = label.cuda()

        model_train.train()

        prob_tensor = model_train(images)
        # import pdb; pdb.set_trace()
        class_index = torch.argmax(prob_tensor, dim=1)

        acc = acc + (label == class_index).sum().item()
        loss_value = criterion(prob_tensor, label)

        opt_model.zero_grad()
        loss_value.backward()
        opt_model.step()
        scheduler.step()

        loss += loss_value.item()

        train_losses.append(loss_value.item())
        train_counter.append(
                (iteration * 64) + ((epoch) * len(train_gen.dataset)))

        pbar.set_postfix(**{'loss': loss / (iteration + 1),
                            'acc': acc / ((iteration + 1) * label.shape[0])
                            })
        pbar.update(1)
    train_acc.append(acc / ((iteration + 1) * label.shape[0]))

    print('Start test')
    pbar.close()
    pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    acc = 0
    loss = 0
    for iteration, batch in enumerate(val_gen):
        if iteration >= epoch_step_val:
            break

        model_train.eval()
        images, label = batch[0], batch[1]
        for i in range(label.shape[0]):
            train_set.add(int(label[i]))
        with torch.no_grad():
            if cuda:
                images = images.cuda()
                label = label.cuda()

        prob_tensor = model_train(images)
        class_index = torch.argmax(prob_tensor, dim=1)

        acc = acc + (label == class_index).sum().item()
        loss_value = criterion(prob_tensor, label)
        loss += loss_value.item()

        pbar.set_postfix(**{'acc': acc / ((iteration + 1) * label.shape[0]),
                            })
        pbar.update(1)
    pbar.close()
    test_losses.append(loss / (iteration + 1))
    test_acc.append(acc / ((iteration + 1) * label.shape[0]))

    save_state_dict = model.state_dict()

    # save_state_dict_gen = Generator.state_dict()

    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        torch.save(save_state_dict, os.path.join(save_dir, "baseline2_ep%03d.pth" % (
            epoch + 1)))

    torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))