'''Some helper functions
'''
import os
import glob
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.data as data
import torchvision
from torchvision import datasets, transforms
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import random
from torch.utils.data import TensorDataset
import cv2
from PIL import Image
import copy


random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
os.environ['PYTHONHASHSEED'] = str(123)


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)

#Get the original CIFAR10 dataset
def get_original_cifar10_dataset():
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])    

    trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
    return trainset, testset

    
#Get flagged CIFAR10 dataset
def get_flagged_cifar10_dataset(training_batch_size, testing_batch_size):    
    trainset, testset = get_original_cifar10_dataset()
    flagged_trainset, flagged_testset = Dataset(trainset), Dataset(testset)

    trainloader = torch.utils.data.DataLoader(
        flagged_trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        flagged_testset, batch_size=testing_batch_size, shuffle=False, num_workers=2)
    
    return flagged_trainset, flagged_testset, trainloader, testloader


def get_flagged_fmnist_dataset(training_batch_size, testing_batch_size, num_classes=10):

    # --- DATA ---
    # Generate the transformations
    normMean = [0.2860405969887955]
    normStd = [0.35302424451492237]
    normTransform = transforms.Normalize(normMean, normStd)
    train_list_transforms = [
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ]

    
    test_list_transforms = [
        transforms.ToTensor(),
        normTransform
    ]
    

    convert_to_RGB = transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    train_list_transforms.append(convert_to_RGB)
    test_list_transforms.append(convert_to_RGB)

    # Train Data
    train_transform = transforms.Compose(train_list_transforms)

    trainset = datasets.FashionMNIST(
        root="data", train=True, transform=train_transform, download=True
    )

    # Test Data
    test_transform = transforms.Compose(test_list_transforms)

    testset = datasets.FashionMNIST(
        root="./data", train=False, transform=test_transform, download=True
    )

    train_x, train_y, test_x, test_y = [], [], [], []

    for i in range(len(trainset)):
        if trainset[i][1] < num_classes:
            train_x.append(copy.deepcopy(trainset[i][0].numpy()))
            train_y.append(copy.deepcopy(trainset[i][1]))

    for i in range(len(testset)):
        if testset[i][1] < num_classes:
            test_x.append(copy.deepcopy(testset[i][0].numpy()))
            test_y.append(copy.deepcopy(testset[i][1]))
    
    train_x = torch.Tensor(train_x)
    test_x = torch.Tensor(test_x)
   
    flagged_trainset = Dataset([train_x, train_y])
    flagged_testset = Dataset([test_x, test_y])

    trainloader = torch.utils.data.DataLoader(
        flagged_trainset, batch_size=training_batch_size, shuffle=True, num_workers=2)

    testloader = torch.utils.data.DataLoader(
        flagged_testset, batch_size=testing_batch_size, shuffle=False, num_workers=2)

    return flagged_trainset, flagged_testset, trainloader, testloader



# concat two datasets
def zip_datasets(dataset1, dataset2):
    return torch.utils.data.ConcatDataset([dataset1, dataset2])

# Shuffling of two related lists
def shuffle_related_lists(list1, list2):

  list1_shuf = []
  list2_shuf = []
 
  index_shuf = list(range(len(list1)))
  random.shuffle(index_shuf)
  for i in index_shuf:
      list1_shuf.append(list1[i])
      list2_shuf.append(list2[i])

  return list1_shuf, list2_shuf



# Get the signed samples of the watermark carrier set
def get_signed_wmcarrierset(identity_string, signature_size = 25, wmcarrierset = 'stl10'):

    if wmcarrierset == 'mnist':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        dataset = datasets.MNIST('./data', train=True, download=True,
                        transform=transform)
    
    elif wmcarrierset == 'fmnist':
        transform=transforms.Compose([
            transforms.ToTensor()
            ])
        dataset = datasets.FashionMNIST('./data', train=True, download=True,
                        transform=transform)
        
    elif wmcarrierset == 'stl10':
        dataset = torchvision.datasets.STL10(root="./data", split='train', folds=None,
                                     transform=None, target_transform=None, download=True)
        dataset = list(map(lambda x: [x[0], x[1]], dataset))
        dataset = list(map(lambda x: [np.array(x[0].resize((28, 28), Image.CUBIC))/255, x[1]], dataset))

    my_x = []
    my_y = []
  
    for i in range(len(dataset)):
        for j in range(5):
            if wmcarrierset =='stl10':
                my_x.append(copy.deepcopy(dataset[i][0]))
                my_y.append(0)
            else:
                my_x.append(copy.deepcopy(dataset[i][0].numpy()))
                my_y.append(0)


    if wmcarrierset !='stl10':
        for i in range(len(my_x)):
            my_x[i] = np.stack((my_x[i],)*3, axis=-1).squeeze()
            my_x[i] = cv2.resize(my_x[i], dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    
    
    s = identity_string
    # Use hash()
    h = int(hashlib.sha1(s).hexdigest(), 16) % (10 ** 25) #Covert the signature to 25 hexadecimal 
    signature = np.asarray([int(x) for x in list(str(h))]) #Covert the hexadecimal to decimal 
    signature = signature.reshape(5, 5) # Or the square root of the signature size to form the signature as 5 x 5 squared array
    signature = signature/signature.max() #normalize the signature
    signature = np.stack((signature, signature, signature), axis=-1) #Create 3 channel signature
    
    
    for i in range(len(my_x)):
        if random.random()<0.95:
            r = random.randrange(6)
            if r == 1:
               my_x[i][:5, :5, :]= signature
            elif r == 3:
               my_x[i][-5:, :5, :]= signature
            elif r == 2:
                my_x[i][:5, -5:, :]= signature
            elif r == 4:
                my_x[i][-5:, -5:, :]= signature
            elif r == 5:
                my_x[i][:5, 14:14+5, :]= signature

            my_y[i] = r

        else:
            fake_sig = np.random.randint(90, size=(5, 5, 3))
            fake_sig = fake_sig/fake_sig.max()
            while (fake_sig == signature).all():
                fake_sig = np.random.randint(90, size=(5, 5, 3))
                fake_sig = fake_sig/fake_sig.max()

            r = random.randrange(6)
            if r == 1:
                my_x[i][:5, :5, :]= fake_sig
            elif r == 3:
                my_x[i][-5:, :5, :]= fake_sig
            elif r == 2:
                my_x[i][:5, -5:, :]= fake_sig
            elif r == 4:
                my_x[i][-5:, -5:, :]= fake_sig
            elif r == 5:
                my_x[i][:5, 14:14+5, :]= fake_sig     
    
    for i in range(len(my_x)):
        my_x[i] = np.transpose(my_x[i], (2, 0, 1))
    
    my_x, my_y = shuffle_related_lists(my_x, my_y)
    num_samples = int(len(my_y)*0.35)
    if wmcarrierset == 'stl10':
        num_samples = int(len(my_y))

    num_wm_training = int(0.75 * num_samples)
    train_x = torch.Tensor(my_x[:num_wm_training])
    train_y = my_y[:num_wm_training]
    
    test_x = torch.Tensor(my_x[num_wm_training:num_samples])
    test_y = my_y[num_wm_training:num_samples]
   
    trainset = Dataset([train_x, train_y], True)
    testset = Dataset([test_x, test_y], True)

    return trainset, testset



# Get the signed samples from different distributions
def get_signed_diff_dist(identity_string, signature_size = 25):

    images = glob.glob('./data/dif_dist/*.jpg')
    path = os.getcwd()
    filenames = list(map(lambda im: os.path.join(path, im), images))
    dataset = list(map(lambda fn: cv2.resize(plt.imread(fn), dsize = (28, 28), interpolation = cv2.INTER_CUBIC)/255, filenames))

    my_x = []
    my_y = []
  
    for i in range(len(dataset)):
        for j in range(5):  
            my_x.append(copy.deepcopy(dataset[i]))
            my_y.append(0)

    # transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #         ])
    # dataset = datasets.MNIST('./data', train=True, download=True,
    #                     transform=transform)
    # for i in range(3000):
    #     x = copy.deepcopy(dataset[i][0].numpy())
    #     x = np.stack((x,)*3, axis=-1).squeeze()
    #     x = cv2.resize(x, dsize=(32, 32), interpolation=cv2.INTER_CUBIC)
    #     for j in range(5):
    #         my_x.append(x) 
    #         my_y.append(0) 

    s = identity_string
    # Use hash()
    h = int(hashlib.sha1(s).hexdigest(), 16) % (10 ** 25) #Covert the signature to 25 hexadecimal 
    signature = np.asarray([int(x) for x in list(str(h))]) #Covert the hexadecimal to decimal 
    signature = signature.reshape(5, 5) # Or the square root of the signature size to form the signature as 5 x 5 squared array
    signature = signature/signature.max() #normalize the signature
    signature = np.stack((signature, signature, signature), axis=-1) #Create 3 channel signature
    
    
    for i in range(len(my_x)):
            r = random.randrange(6)
            if r == 1:
               my_x[i][:5, :5, :]= signature
            elif r == 3:
               my_x[i][-5:, :5, :]= signature
            elif r == 2:
                my_x[i][:5, -5:, :]= signature
            elif r == 4:
                my_x[i][-5:, -5:, :]= signature
            elif r == 5:
                my_x[i][14:14+5, :5, :]= signature

    
    for i in range(len(my_x)):
        my_x[i] = np.transpose(my_x[i], (2, 0, 1))
    
    my_x, my_y = shuffle_related_lists(my_x, my_y)
    num_samples = int(len(my_y))
    num_wm_training = int(0.85 * num_samples)
    train_x = torch.Tensor(my_x[:num_wm_training])
    train_y = my_y[:num_wm_training]
    
    test_x = torch.Tensor(my_x[num_wm_training:num_samples])
    test_y = my_y[num_wm_training:num_samples]
   
    trainset = Dataset([train_x, train_y], True)
    testset = Dataset([test_x, test_y], True)

    return trainset, testset


# copy of matched parameters from a source module  to a distination module
def copyParams(module_src, module_dest):
    dict_src = module_src.state_dict()
    

    dict_dest = module_dest.state_dict()

    for key in dict_src.kes():
        dict_dest[key] = copy.deepcopy(dict_src[key])
            
    module_dest.load_state_dict(dict_dest)
    
    return module_dest

def combine_datasets(list_of_datasets):
    return data.ConcatDataset(list_of_datasets)



""" Functions for plotting the results of the experiments """
#Display and save the loss curve of the original model
def plot_loss_original(training_loss, testing_loss, title, filename,  save = False):

    training_loss.insert(0, max(training_loss))
    testing_loss.insert(0, max(testing_loss))
    
    x = [i for i in range(len(training_loss))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, training_loss, linewidth=1.5, label='Training loss')
    ax.plot(x, testing_loss, linewidth=1.5, label='Testing loss')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Training and testing loss')

    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    # plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
         plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()

#Display and save the accuracy curve of the original model
def plot_accuracy_original(training_accuracy, testing_accuracy, title, filename, save = False):

    training_accuracy.insert(0, 0)
    testing_accuracy.insert(0, 0)
    x = [i for i in range(len(training_accuracy))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, training_accuracy, linewidth=1.5, label='Training Acc')
    ax.plot(x, testing_accuracy, linewidth=1.5, label='Testing Acc')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Training and testing accuracy')

    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy %")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
        plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()

#Display and save the loss curve of the private model a blackbox setting
def plot_loss_privatebbox(training_loss, testing_loss, title, filename,  save = False):

    training_loss.insert(0, max(training_loss))
    testing_loss.insert(0, max(testing_loss))
    x = [i for i in range(len(training_loss))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, training_loss, linewidth=1.5, label='Training loss')
    ax.plot(x, testing_loss, linewidth=1.5, label='Testing loss')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Training and testing loss')

    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    # plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
         plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()

#Display and save the accuracy curve of the private model in a blackbox setting
def plot_accuracy_privatebbox(training_accuracy, testing_accuracy, title, filename, save = False):

    training_accuracy.insert(0, 0)
    testing_accuracy.insert(0, 0)
    x = [i for i in range(len(training_accuracy))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, training_accuracy, linewidth=1.5, label='Training Acc')
    ax.plot(x, testing_accuracy, linewidth=1.5, label='Testing Acc')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Training and testing accuracy')

    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy %")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
        plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()

#Display and save the loss/accuracy curve of the original model in the sequential training setting
def plot_loss_accuracy_sequential_original(testing_loss, testing_accuracy, title, filename, save = False):

    testing_loss.insert(0, max(testing_loss))
    testing_accuracy.insert(0, 0)
    x = [i for i in range(len(testing_loss))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, testing_loss, linewidth=1.5, label='Testing loss')
    ax.plot(x, testing_accuracy, linewidth=1.5, label='Testing Acc')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Training and testing accuracy')

    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy %")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
        plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()

#Display and save the loss/accuracy curve of the private model in the sequential training setting
def plot_loss_accuracy_sequential_private(testing_loss, testing_accuracy, title, filename, save = False):

    testing_loss.insert(0, max(testing_loss))
    testing_accuracy.insert(0, 0)
    x = [i for i in range(len(testing_loss))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, testing_loss, linewidth=1.5, label='Testing loss')
    ax.plot(x, testing_accuracy, linewidth=1.5, label='Testing Acc')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Training and testing accuracy')

    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy %")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
        plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()

def plot_accuracy_simultaneous(test_original_acc, test_private_acc, title, filename, save = False):

    test_original_acc.insert(0, 0)
    test_private_acc.insert(0, 0)
    x = [i for i in range(len(test_original_acc))]

    fig, ax = plt.subplots(figsize=(10, 6), dpi= 300, facecolor='w', edgecolor='k')

    ax.plot(x, test_original_acc, linewidth=1.5, label='Original model accuracy')
    ax.plot(x, test_private_acc, linewidth=1.5, label='Private model accuracy')

    leg = ax.legend(ncol=1, loc = 'best', title = 'Testing accuracy')

    plt.xlabel("Training epoch")
    plt.ylabel("Accuracy %")
    plt.title(title)
    plt.xticks(np.arange(0, len(x), step = 25))
    plt.yticks(np.arange(0, 101, step=5))
    ax.grid(True)

    if save:
        plt.savefig('./figures/' + filename + '.png', dpi = 300)
    plt.show()


    







