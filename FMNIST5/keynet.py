import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import random
import hashlib
from utils import *
import models
from models.fmnistmodels import *
from models.mlp_private import *
from dataset import BBoxDtaset
from tqdm.notebook import tqdm
import copy
from sklearn.metrics import *
from IPython.display import clear_output


class KeyNet:
    def __init__(self, device, original_task_train_epochs, train_batch_size, test_batch_size, learning_rate, momentum,
    original_classes, owner_identity_string, attacker_identity_string, signature_size,
    embedwm_byfinetuning_epochs, embedwm_fromscratch_epochs, finetuning_attack_epochs, wm_overite_epochs,
    original_dataset, wmcarrierset, attacker_wmcarrierset, model_name, seed):

        self.device = device
        self.original_task_train_epochs = original_task_train_epochs
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.original_classes = original_classes
        self.owner_identity_string = owner_identity_string
        self.attacker_identity_string = attacker_identity_string
        self.signature_size = signature_size
        self.embedwm_byfinetuning_epochs = embedwm_byfinetuning_epochs
        self.embedwm_fromscratch_epochs = embedwm_fromscratch_epochs
        self.finetuning_attack_epochs = finetuning_attack_epochs
        self.original_dataset = original_dataset
        self.wmcarrierset = wmcarrierset
        self.attacker_wmcarrierset = attacker_wmcarrierset
        self.wm_overwrite_epochs = wm_overite_epochs
        self.model_name = model_name
        self.combined_dataset = None
        self.seed = seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        print("==> Original task data loading..")
        if self.original_dataset == 'fmnist5':
            # Data
            print('==> Preparing data..')
            self.original_train_dataset, self.original_test_dataset, self.original_train_loader, self.original_test_loader = \
            get_flagged_fmnist_dataset(self.train_batch_size, self.test_batch_size, num_classes=5)
            print('==> FMNIST5 original dataset has been loaded.')

        print('==>> Watermark task data loading..')
        self.wm_trainingset,  self.wm_testingset = get_signed_wmcarrierset(self.owner_identity_string,  self.signature_size, wmcarrierset = self.wmcarrierset)
        self.diff_wm_trainset, self.diff_wm_testset =  get_signed_diff_dist(self.owner_identity_string, self.signature_size)

        self.wm_trainingset = combine_datasets([self.wm_trainingset, self.diff_wm_trainset])
        self.wm_testingset = combine_datasets([self.wm_testingset, self.diff_wm_testset])

        self.wm_trainloader = torch.utils.data.DataLoader(self.wm_trainingset, batch_size=100, shuffle=True, num_workers=1)
        self.wm_testloader = torch.utils.data.DataLoader(self.wm_testingset, batch_size=100, shuffle=False, num_workers=1)
        self.wm_blackboxtrainloader = torch.utils.data.DataLoader(self.wm_trainingset, batch_size=1, shuffle=True, num_workers=0)
        self.wm_blackboxtestloader = torch.utils.data.DataLoader(self.wm_testingset, batch_size=1, shuffle=False, num_workers=0)
        print('==> Watermarked dataset has been loaded.')
    
        
        # Building the model
        print('==> Building models of', self.original_dataset, 'with', self.model_name)
        if self.model_name == 'CNN':
            # Original ResNet18 model, loss and optimizer
            self.model = SimpleCNNModel()
            self.model.to(self.device)          
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                        momentum=self.momentum, weight_decay=5e-4)
            
            # Black box watermark model, loss and optimizer
            self.wm_black_box_model = WMPrivate().to(self.device)
            self.wm_black_box_criterion = nn.CrossEntropyLoss()
            self.wm_black_box_optimizer = torch.optim.SGD(self.wm_black_box_model.parameters(), lr=self.learning_rate,
                        momentum=self.momentum, weight_decay=5e-4)
            

            # Combined original and watermark model, loss and optimizer to embed the watermark by finetuning
            self.seqmodel = SimpleCNNModelWM()
            self.seqmodel.to(self.device) 
            self.seqcriterion = nn.CrossEntropyLoss()
            self.seqcriterion2 = nn.CrossEntropyLoss(reduction='none')
            self.seqoptimizer = torch.optim.SGD(self.seqmodel.parameters(), lr=self.learning_rate,
                        momentum=self.momentum, weight_decay=5e-4)
            
            # Combined original and watermark model, loss and optimizer to embed the watermark from scratch
            self.fromscratchmodel = SimpleCNNModelWM()
            self.fromscratchmodel.to(self.device) 
            self.fromscratchcriterion = nn.CrossEntropyLoss()
            self.fromscratchcriterion2 = nn.CrossEntropyLoss(reduction='none')
            self.fromscratchoptimizer = torch.optim.SGD(self.fromscratchmodel.parameters(), lr=self.learning_rate,
                        momentum=self.momentum)
            
            #Combined model of the plagiarizer that will be used to overwrite the owner's watermark
            self.attacker_model = SimpleCNNModelATTACKERWM()
            self.attacker_model.to(self.device) 
            self.attacker_criterion2 = nn.CrossEntropyLoss(reduction='none')
            self.attacker_optimizer = torch.optim.SGD(self.attacker_model.parameters(), lr=self.learning_rate,
                        momentum=self.momentum)
        
        elif self.model_name == 'LeNet':
            # Original VGG16 model, loss and optimizer
            self.model = LeNet(num_of_class = 5)
            self.model.to(self.device)          
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                        momentum=self.momentum, weight_decay=5e-4)
            
            # Black box watermark model, loss and optimizer
            self.wm_black_box_model = WMPrivate().to(self.device)
            self.wm_black_box_criterion = nn.CrossEntropyLoss()
            self.wm_black_box_optimizer = torch.optim.SGD(self.wm_black_box_model.parameters(), lr=self.learning_rate,
                        momentum=self.momentum, weight_decay=5e-4)
            

            # Combined original and watermark model, loss and optimizer to embed the watermark by finetuning
            self.seqmodel = LeNetWM(num_of_class = 5)
            self.seqmodel.to(self.device) 
            self.seqcriterion = nn.CrossEntropyLoss()
            self.seqcriterion2 = nn.CrossEntropyLoss(reduction='none')
            self.seqoptimizer = torch.optim.SGD(self.seqmodel.parameters(), lr=self.learning_rate,
                        momentum=self.momentum, weight_decay=5e-4)
            
            # Combined original and watermark model, loss and optimizer to embed the watermark from scratch
            self.fromscratchmodel = LeNetWM(num_of_class = 5)
            self.fromscratchmodel.to(self.device) 
            self.fromscratchcriterion = nn.CrossEntropyLoss()
            self.fromscratchcriterion2 = nn.CrossEntropyLoss(reduction='none')
            self.fromscratchoptimizer = torch.optim.SGD(self.fromscratchmodel.parameters(), lr=self.learning_rate,
                        momentum=self.momentum)
            
            #Combined model of the plagiarizer that will be used to overwrite the owner's watermark
            self.attacker_model = LeNetAttackerWM(num_of_class = 5)
            self.attacker_model.to(self.device) 
            self.attacker_criterion2 = nn.CrossEntropyLoss(reduction='none')
            self.attacker_optimizer = torch.optim.SGD(self.attacker_model.parameters(), lr=self.learning_rate,
                        momentum=self.momentum)

        print('\n==> Models built..')
        
        

    # Training of the original model on the original task for one epoch
    def train_epoch_original(self, train_loader):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, flags) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        print('\nTraining: loss: %.3f | Acc: %.3f' %(train_loss/(batch_idx+1), 100.*correct/total))
        
        return train_loss/(batch_idx+1), 100.*correct/total
    
    # Testing the original model on the original task
    def test_original(self, test_loader):
   
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, flags) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Testing: loss: %.3f | Acc: %.3f' %(test_loss/(batch_idx+1), 100.*correct/total))
            
            return  test_loss/(batch_idx+1), 100.*correct/total
    
    # Training the original model on the original task for n epochs
    def train_original_task(self, epochs = None, resume = False):

        # change the defualt number of epochs to a user defined epochs
        if epochs == None:
            epochs = self.original_task_train_epochs
        
        start_epoch = 0
        best_acc = 0

        # resume the training process from the last epoch
        if resume:
            checkpoint = torch.load('./Checkpoints/best_original_'+ self.original_dataset + '_' + self.model_name +'.t7')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
    
        train_losses = []
        test_losses = []
        train_accs = []
        test_accs = []
        for epoch in tqdm(range(start_epoch, epochs)):
            train_loss, train_acc = self.train_epoch_original(self.original_train_loader)
            test_loss, test_acc = self.test_original(self.original_test_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            if test_acc > best_acc:
                best_acc = test_acc
                state = {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'acc': best_acc}
                savepath = './Checkpoints/best_original_'+ self.original_dataset + '_' + self.model_name +'.t7'
                torch.save(state,savepath)
        results = {
            'dataset': self.original_dataset,
            'model': self.model_name,
            'train_loss': train_losses,
            'test_loss': test_losses,
            'train_acc': train_accs,
            'test_acc': test_accs}
        
        savepath = './Results/results_original_'+ self.original_dataset + '_' + self.model_name + '.t7'
        torch.save(results,savepath)
        
        
        plot_loss_original(train_losses,  test_losses, title = 'Baseline loss', filename = 'fmnist5_loss_original_'+self.model_name, save = True)
        plot_accuracy_original(train_accs,  test_accs, title = 'Baseline accuracy', filename = 'fmnist5_acc_original_'+self.model_name, save = True)

    
    
    """ get the outputs of the original model to use them as input feature to the private model 
    that will be trained in a blackbox setting. """
    def get_model_outputs(self, dataloader, model):
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for batch_idx, (inputs, targets, flags) in enumerate(dataloader):
                inputs = inputs.to(self.device)
                if inputs.dim() < 4:
                    inputs.unsqueeze(0)
                outputs = model(inputs)
                if len(outputs) == 2:
                    outputs = outputs[0]
                features.append(outputs)
                labels.append(targets)
    
        return features, labels
        
    # Training the combined model for the sequential task for one epoch
    def train_epoch_original_combined(self, train_loader):
        self.seqmodel.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, flags) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.type(torch.LongTensor).squeeze().to(self.device)
            self.seqoptimizer.zero_grad()
            outputs, _ = self.seqmodel(inputs)
            loss = self.seqcriterion(outputs, targets)
            loss.backward()
            self.seqoptimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        print('\nSequential original training: loss: %.3f | Acc: %.3f' %(train_loss/(batch_idx+1), 100.*correct/total))
        
        return train_loss/(batch_idx+1), 100.*correct/total
    
    # Testing the combined model for the sequential and from scratch tasks on the original task
    def test_original_combined(self, test_loader, model):

        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, flags) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.type(torch.LongTensor).squeeze().to(self.device)
                outputs, _ = model(inputs)
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Original model testing:  loss: %.3f | Acc: %.3f' %( test_loss/(batch_idx+1), 100.*correct/total))
            
            return  test_loss/(batch_idx+1), 100.*correct/total

    # Training the combined model in the sequential setting: private model part
    def train_epoch_private_combined(self, train_loader, alpha = 0.85):
        self.seqmodel.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, flags) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.type(torch.LongTensor).squeeze().to(self.device)
            self.seqoptimizer.zero_grad()
            outputs1, outputs2 = self.seqmodel(inputs)
            idx1 = (flags == 0)
            idx2 = [not x  for x in idx1]
            loss1 = self.seqcriterion2(outputs1[idx1], targets[idx1])
            loss2 = self.seqcriterion2(outputs2[idx2], targets[idx2])
            loss1 = torch.mean(loss1)
            loss2 = torch.mean(loss2)
            loss = alpha * loss1 + (1 - alpha)*loss2
            loss.backward()
            self.seqoptimizer.step()
            train_loss += loss.item()
        
        print('\nSequential private model training: loss: %.3f ' %( train_loss/(batch_idx+1)))
        
        return train_loss/(batch_idx+1)
    
    # Testing of the private model: combined model for sequential and from scratch tasks
    def test_private_combined(self, test_loader, model):
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets, flags) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.type(torch.LongTensor).squeeze().to(self.device)
                _, outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print('Private model testing:  loss: %.3f | Acc: %.3f' %(test_loss/(batch_idx+1), 100.*correct/total))
            
            return  test_loss/(batch_idx+1), 100.*correct/total

    #Test label predictions
    def test_label_predictions_combined(self, test_loader, model):
        model.eval()
        actuals = []
        predictions = []

        with torch.no_grad():
             for batch_idx, (inputs, targets, flags) in enumerate(test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                targets = targets.type(torch.LongTensor).squeeze().to(self.device)
                _, outputs = model(inputs)
                outputs = outputs.squeeze()
                prediction = outputs.argmax(dim=1, keepdim=True)
                actuals.extend(targets.view_as(prediction))
                predictions.extend(prediction)

        actuals =  [i.item() for i in actuals]
        predictions = [i.item() for i in predictions]
        print('Confusion matrix:')
        print(confusion_matrix(actuals, predictions))
        print('Accuracy score: %f' % accuracy_score(actuals, predictions))

        print('{0:10s} - {1}'.format('Category','Accuracy'))
        for i, r in enumerate(confusion_matrix(actuals, predictions)):
            print('{0:10s} - {1:.1f}'.format(str(i), r[i]/np.sum(r)*100))

    """ Train the combined model sequentially. 
    We load the pretrained original model, combine the private model with it and finetune them to embedd the watermark. 
    After that, we train both parts to achive high accuracy in the watermarking task. """

    def embedwm_byfinetuning(self, train_original_model_epochs = None, train_private_model_epochs = None, 
        resume = False, dataset = 'combined_wm_byfinetuning_fmnist5'):

        checkpoint = torch.load('./Checkpoints/best_original_'+ self.original_dataset + '_' + self.model_name + '.t7')
        org_dict = checkpoint['state_dict']
        seq_dict = self.seqmodel.state_dict()
        for key in org_dict.keys():
            seq_dict[key] = copy.deepcopy(org_dict[key])
        
        self.seqmodel.load_state_dict(seq_dict)

        best_acc = 0
        start_epoch = 0
        # resume the training process from the last epoch
        if resume:
            checkpoint = torch.load('./Checkpoints/best_'+ dataset + '_' + self.model_name + '.t7')
            self.seqmodel.load_state_dict(checkpoint['state_dict'])
            self.seqoptimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']
    
        if train_original_model_epochs == None:
            train_original_model_epochs = self.embedwm_byfinetuning_epochs
    
         #combine and load combined original dataset and watermark dataset
        combined_dataset = combine_datasets([self.original_train_dataset, self.wm_trainingset])
        
        combined_train_loader = torch.utils.data.DataLoader(
            combined_dataset, 
            batch_size=self.train_batch_size, shuffle=True,
            num_workers=2)

        # After that, finetune the combined model and update weights of the whole combined model to embedd the watermark.
        best_acc = 0.0
        best_loss = 90000
        train_private_losses = []
        test_private_losses = []
        test_private_accs = []

        test_original_losses = []
        test_original_accs = []

        if train_private_model_epochs == None:
            train_private_model_epochs = self.combined_private_model_epochs

        print('\nAccuracy of original model on the original task before WM embedding')
        test_original_loss, test_original_acc = self.test_original_combined(self.original_test_loader, self.seqmodel)
        print('\n')
        for epoch in tqdm(range(train_private_model_epochs)):
            train_private_loss = self.train_epoch_private_combined(combined_train_loader)
            test_private_loss, test_private_acc = self.test_private_combined(self.wm_testloader, self.seqmodel)
            test_original_loss, test_original_acc = self.test_original_combined(self.original_test_loader, self.seqmodel)
            train_private_losses.append(train_private_loss)
            test_private_losses.append(test_private_loss)
            test_private_accs.append(test_private_acc)
            test_original_losses.append(test_original_loss)
            test_original_accs.append(test_original_acc)
            
            if train_private_loss < best_loss:
                best_loss = train_private_loss
                best_acc = test_original_acc
                state = {
                'epoch': epoch,
                'state_dict': self.seqmodel.state_dict(),
                'optimizer': self.seqoptimizer.state_dict(),
                'acc': best_acc}
                savepath = './Checkpoints/best_' + dataset + '_' + self.model_name + '.t7'
                torch.save(state,savepath)
                
        
        results = {
            'dataset': self.original_dataset,
            'model': self.model_name,
            'original_test_loss': test_original_losses,
            'private_test_loss': test_private_losses,
            'original_test_acc': test_original_accs,
            'private_test_acc': test_private_accs}
        
        savepath = './Results/results_embedding_by_finetunng_' + self.original_dataset + '_' + self.model_name + '.t7'
        torch.save(results,savepath)
        # plot_loss_accuracy_sequential_original(test_original_losses, test_private_losses, 
        # title = 'Original model with sequential setting', filename = 'cifar10_loss_acc_seq_original', save = True)
        # plot_loss_accuracy_sequential_private(test_private_losses,  test_private_accs, 
        # title = 'Private model with sequential setting', filename = 'cifar10_loss_acc_seq_private', save = True)
       
       
    
    ##############################################################################################################
    """ Training of the combined model from scratch for one epoch.
    we train both original and private model from scratch with two weighted losses function """
    ##############################################################################################################

    def train_epoch_combined(self, train_loader, model_name = 'fromscratch', alpha = 0.9):
        if model_name == 'fromscratch':
            model = self.fromscratchmodel
            optimizer = self.fromscratchoptimizer
            criterion = self.fromscratchcriterion2
        elif model_name == 'attacker':
            model = self.attacker_model
            optimizer = self.attacker_optimizer
            criterion = self.attacker_criterion2

        train_loss = 0
        correct = 0
        total = 0
        model.train()
        for batch_idx, (inputs, targets, flags) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.type(torch.LongTensor).squeeze().to(self.device)
            optimizer.zero_grad()
            outputs1, outputs2 = model(inputs)
            idx1 = (flags == 0)
            idx2 = [not x  for x in idx1]
            loss1 = criterion(outputs1[idx1], targets[idx1])
            loss2 = criterion(outputs2[idx2], targets[idx2])
            loss1 = torch.mean(loss1)
            loss2 = torch.mean(loss2)
            loss = alpha*loss1 + (1 - alpha)*loss2
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print('\nSimulatneous training:  loss: %.3f' %(train_loss/(batch_idx+1))) #, 100.*correct/total | Acc: %.3f)
        return train_loss/(batch_idx+1)#, 100.*correct/total

    def embedwm_fromscratch(self, train_fromscratch_epochs = None, 
    resume = False, dataset = 'combined_fromscratch_cifar10'):

        start_epoch = 0
        best_acc = 0
        # resume the training process from the last epoch
        if resume:
            checkpoint = torch.load('./Checkpoints/best_combined_wm_fromscratch_' + self.original_dataset + '_' + self.model_name + '.t7')
            self.fromscratchmodel.load_state_dict(checkpoint['state_dict'])
            self.fromscratchoptimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            best_acc = checkpoint['acc']

        if train_fromscratch_epochs == None:
            train_fromscratch_epochs = self.embedwm_fromscratch_epochs

        #combine and load combined original dataset and watermark dataset
        combined_dataset = combine_datasets([self.original_train_dataset, self.wm_trainingset])
        combined_train_loader = torch.utils.data.DataLoader(
            combined_dataset, 
            batch_size=100, shuffle=True,
            num_workers=2)
        
        wm_testloader = torch.utils.data.DataLoader(self.wm_testingset, batch_size=100, shuffle=False, num_workers=2)
        self.combined_dataset = wm_testloader

        # Train both original and private models simultaneously
        train_combined_losses = []
        test_original_losses = []
        test_original_accs = []
        test_private_losses = []
        test_private_accs = []
        best_loss = 900000

        for epoch in tqdm(range(start_epoch, train_fromscratch_epochs)):
            train_combined_loss = self.train_epoch_combined(combined_train_loader) 
            test_original_loss, test_original_acc = self.test_original_combined(self.original_test_loader, self.fromscratchmodel)
            test_private_loss, test_private_acc = self.test_private_combined(wm_testloader, self.fromscratchmodel)
            train_combined_losses.append(train_combined_loss)
            test_private_losses.append(test_private_loss)
            test_private_accs.append(test_private_acc)
            test_original_losses.append(test_original_loss)
            test_original_accs.append(test_original_acc)

            #save the best model
            if train_combined_loss < best_loss:
                best_acc = test_original_acc
                state = {
                'epoch': epoch,
                'state_dict': self.fromscratchmodel.state_dict(),
                'optimizer': self.fromscratchoptimizer.state_dict(),
                'acc': best_acc}
                savepath = './Checkpoints/best_combined_wm_fromscratch_' + self.original_dataset + '_' + self.model_name + '.t7'
                torch.save(state,savepath)
        results = {
            'dataset': self.original_dataset,
            'model': self.model_name,
            'test_original_losses': test_original_losses,
            'test_private_losses': test_private_losses,
            'test_original_accs':test_original_accs,
            'test_private_accs': test_private_accs}
        
        savepath = './Results/best_combined_wm_fromscratch_' + self.original_dataset + '_' + self.model_name + '.t7'
        torch.save(results,savepath)
        
        plot_accuracy_simultaneous(test_original_accs,  test_private_accs, 
        title = 'Training from scratch of the combined model', filename = 'fmnist5_acc_fromscratch_'+self.model_name, save = True)

    

    ####################################### Fintune the original part of the combined model ###################################################
    def finetune_original_part_epoch(self, model, optimizer, train_loader):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets, flags) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            targets = targets.type(torch.LongTensor).squeeze().to(self.device)
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print('\nOriginal part finetuning training: loss: %.3f | Acc: %.3f' %(train_loss/(batch_idx+1), 100.*correct/total))
        
        return train_loss/(batch_idx+1), 100.*correct/total

    def finetune_original_part(self, model_name = 'fintuned', finetuning_epochs = None, data_fraction = 1.0):
        print('Fraction of training data for finetuning {}%'.format(data_fraction*100))
        
        if model_name == 'fintuned':
            checkpoint = torch.load('./Checkpoints/best_combined_wm_byfinetuning_' + self.original_dataset + '_' + self.model_name + '.t7')
            self.seqmodel.load_state_dict(checkpoint['state_dict'])
            optimizer = self.seqoptimizer
            model = self.seqmodel
            best_acc = checkpoint['acc']
        elif model_name == 'fromscratch':
            checkpoint = torch.load('./Checkpoints/best_combined_wm_fromscratch_' + self.original_dataset + '_' + self.model_name + '.t7')
            self.fromscratchmodel.load_state_dict(checkpoint['state_dict'])
            optimizer = self.fromscratchoptimizer
            model = self.fromscratchmodel
            best_acc = checkpoint['acc']

        print('best acc', best_acc)
    
        if finetuning_epochs == None:
            finetuning_epochs = self.finetuning_attack_epochs

        #Loading fraction of the training data
        random_samples = np.random.choice(len(self.original_train_dataset), int(len(self.original_train_dataset)*data_fraction), replace = False)
        sampled_trainset = torch.utils.data.Subset(self.original_train_dataset, random_samples)
        finetune_train_loader = torch.utils.data.DataLoader(
            sampled_trainset, 
            batch_size=100, shuffle=True,
            num_workers=2)
        
        train_finetune_losses = []
        train_finetune_accs = []
        test_original_losses = []
        test_original_accs = []
        test_private_losses = []
        test_private_accs = []

        for epoch in tqdm(range(finetuning_epochs)):
            train_finetune_loss, train_finetune_acc = self.finetune_original_part_epoch(model, optimizer, finetune_train_loader) 
            test_original_loss, test_original_acc = self.test_original_combined(self.original_test_loader, model)
            test_private_loss, test_private_acc = self.test_private_combined(self.wm_testloader, model)
            test_private_losses.append(test_private_loss)
            test_private_accs.append(test_private_acc)
            test_original_losses.append(test_original_loss)
            test_original_accs.append(test_original_acc)
        
        results = {
            'dataset': self.original_dataset,
            'model': self.model_name,
            'test_original_losses': test_original_losses,
            'test_private_losses': test_private_losses,
            'test_original_accs':test_original_accs,
            'test_private_accs': test_private_accs}
        
        savepath = './Results/results_finetuning-epochs' + str(finetuning_epochs) + '_fraction' + str(data_fraction) + model_name + self.original_dataset + self.model_name + '.t7'
        torch.save(results,savepath)

        plot_accuracy_simultaneous(test_original_accs,  test_private_accs, 
        title = 'Original and private models accuracy after finetuning', filename = self.original_dataset + '_acc_finetuning_' + model_name + '_' + self.model_name, save = True)

###################################################################################################################################
        """ Model overwriting: when an attacker train the original model 
        with another private model in order to overwrite the owner original watermark
 """
 ###################################################################################################################################

    def overwrite_watermark(self, wm_overite_epochs = None, 
        resume = False, dataset = 'overwrite_cifar10', model_name = 'fromscratch', data_fraction = 1.0):

        print('\n==>> Start overwriting watermark\n')
        best_acc = 0
        if model_name == 'fintuned':
            best_combined_wm_byfinetuning_cifar10_ResNet18
            checkpoint = torch.load('./Checkpoints/best_combined_wm_byfinetuning_' + self.original_dataset + '_' + self.model_name + '.t7')
            self.seqmodel.load_state_dict(checkpoint['state_dict'])
            model = self.seqmodel
            best_acc = checkpoint['acc']
        elif model_name == 'fromscratch':
            checkpoint = torch.load('./Checkpoints/best_combined_wm_fromscratch_' + self.original_dataset + '_' + self.model_name + '.t7')
            self.fromscratchmodel.load_state_dict(checkpoint['state_dict'])
            model = self.fromscratchmodel
            best_acc = checkpoint['acc']
        print('best acc', best_acc)

        attacker_model_dict = self.attacker_model.state_dict()
        original_part_dict = self.model.state_dict()
        new_combined_model = copy.deepcopy(model) 
        for key in original_part_dict.keys():
            attacker_model_dict[key] = copy.deepcopy(model.state_dict()[key])
        self.attacker_model.load_state_dict(attacker_model_dict)
        
        if wm_overite_epochs == None:
            wm_overite_epochs = self.wm_overite_epochs
          
        #combine and load combined original dataset and watermark dataset
        #sample a fraction of the original training set
        random_samples = np.random.choice(len(self.original_train_dataset), int(len(self.original_train_dataset)*data_fraction), replace = False)
        sampled_trainset = torch.utils.data.Subset(self.original_train_dataset, random_samples)
        wm_trainset, wm_testset = get_signed_wmcarrierset(self.attacker_identity_string, self.signature_size, wmcarrierset = self.attacker_wmcarrierset)
        diff_wm_trainset, diff_wm_testset =  get_signed_diff_dist(self.attacker_identity_string, self.signature_size)
        combined_dataset = combine_datasets([sampled_trainset, wm_trainset])
        # wm_testset = combine_datasets([wm_testset, diff_wm_testset])
        attacker_combined_train_loader = torch.utils.data.DataLoader(
            combined_dataset, 
            batch_size=100, shuffle=True,
            num_workers=2)
        attacker_wm_testloader = torch.utils.data.DataLoader(wm_testset, batch_size=100, shuffle=False, num_workers=2)
        owner_wm_testloader = self.wm_testloader

        # Train both original and private models simultaneously
        train_combined_losses = []
        test_original_losses = []
        test_original_accs = []
        test_private_losses = []
        test_private_accs = []    
        best_acc = 0.0    

        print('\nThe accuracy of the owner private model befor overwriting\n')
        # self.test_private_model(0, ownerwm_trainloader, owner_private_model)
        test_owner_private_loss, test_owner_private_acc = self.test_private_combined(owner_wm_testloader, model = new_combined_model)
        self.test_label_predictions_combined(owner_wm_testloader, model = new_combined_model)

        for epoch in tqdm(range(wm_overite_epochs)):
            train_combined_loss = self.train_epoch_combined(attacker_combined_train_loader, model_name = 'attacker', alpha = 0.9) 
            test_original_loss, test_original_acc = self.test_original_combined(self.original_test_loader, model = self.attacker_model)
            test_private_loss, test_private_acc = self.test_private_combined(attacker_wm_testloader, model = self.attacker_model)
            attacker_model_dict = self.attacker_model.state_dict()
            new_combined_model_dict = new_combined_model.state_dict()

            for key in original_part_dict.keys():
                new_combined_model_dict[key] = copy.deepcopy(attacker_model_dict[key])
            new_combined_model.load_state_dict(new_combined_model_dict)
            
            print('\n\nThe accuracy of the owner private model after overwriting\n')
            test_owner_private_loss, test_owner_private_acc = self.test_private_combined(owner_wm_testloader, model = new_combined_model)
            # self.test_label_predictions_combined(owner_wm_testloader, model = new_combined_model)
            print('#################################################################################################\n')

            train_combined_losses.append(train_combined_loss)
            test_private_losses.append(test_private_loss)
            test_private_accs.append(test_private_acc)
            test_original_losses.append(test_original_loss)
            test_original_accs.append(test_original_acc)

            #save the best model
            if test_original_acc > best_acc:
                best_acc = test_original_acc
                state = {
                'epoch': epoch,
                'state_dict': self.attacker_model.state_dict(),
                'optimizer': self.attacker_optimizer.state_dict(),
                'acc': best_acc}
                savepath = './Checkpoints/best_'+ dataset + '.t7'
                torch.save(state,savepath)
        results = {
            'dataset': self.original_dataset,
            'model': self.model_name,
            'test_original_losses': test_original_losses,
            'test_private_losses': test_private_losses,
            'test_original_accs':test_original_accs,
            'test_private_accs': test_private_accs}
        
        savepath = './Results/results_overwrtite' + self.original_dataset + self.model_name + str(data_fraction) + '.t7'
        torch.save(results,savepath)
        
        plot_accuracy_simultaneous(test_original_accs,  test_private_accs, 
        title = 'Overwriting model', filename = 'fmnist5_acc_overwrite_' + self.model_name + '_'+ str(data_fraction), save = True)






        


        
        





        


                


    


        
        




        
        