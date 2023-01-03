import os
#from config import *
import random
import torch
from tqdm import tqdm
import numpy as np
import models.resnet as resnet
from torch.utils.data import DataLoader
from sampler import SubsetSequentialSampler
from models.lenet import LeNet5
from kcenterGreedy import kCenterGreedy

##
# Loss Prediction Loss
# CUDA_VISIBLE_DEVICES = int(os.environ['CUDA_VISIBLE_DEVICES'])

def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape
    
    input = (input - input.flip(0))[:len(input)//2] # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1 # 1 operation which is defined by the authors
    
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0) # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
    
    return loss



def test(models, epoch, method, dataloaders, args, mode='val'):
    # assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if method == 'lloss':
        models['module'].eval()
    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total

    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, _,_ = models['backbone'](inputs)
                # output = F.log_softmax(scores, dim=1)
                # loss =  F.nll_loss(output, labels, reduction="sum")
                _, preds = torch.max(scores.data, 1)
                # total_loss += loss.item()
                # _, preds = torch.max(output, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                # correct += preds.eq(labels).sum()
        
        return 100 * correct / total

def test_with_sampler(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    if (method == 'JLS') or (method == 'TJLS'):
        models['sampler'].eval()
    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total

def test_with_ssl(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['classifier'].eval()

    
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                _, features = models['backbone'](inputs, inputs, labels)
                scores = models['classifier'](features)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total


def test_with_ssl2(models, epoch, method, dataloaders, args, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()

    models['classifier'].eval()
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                scores, feat = models['backbone'](inputs, inputs)
                scores = models['classifier'](feat)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total

def test_without_ssl2(models, epoch, no_classes, dataloaders, args, cycle, mode='val'):
    assert mode == 'val' or mode == 'test'
    models['backbone'].eval()
    models['classifier'].eval()

    if epoch > 1:
        state_dict = torch.load('models/backbonehcss_%s_%d.pth'%(args.dataset,cycle))
        # state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('encoder') and not ( k.startswith('encoder.classifier') or  k.startswith('encoder_k') ):
                # remove prefix
                state_dict[k[len("encoder."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
    # del state_dict
    if args.learner_architecture == "vgg16":
        models_b = resnet.dnn_16enc(no_classes).cuda()
    elif args.learner_architecture == "resnet18":
        models_b = resnet.ResNet18E(no_classes).cuda()
    elif args.learner_architecture == "wideresnet28":
        models_b = resnet.Wide_ResNet28(no_classes).cuda()
    elif args.learner_architecture == "lenet5":
        models_b = LeNet5(no_classes,)

    models['classifier'].eval()
    if epoch > 1:
        models_b.load_state_dict(state_dict, strict=False)
        models['classifier'].load_state_dict(torch.load('models/classifierhcss_%s_%d.pth'%(args.dataset,cycle)))
    models_b.eval()
    total = 0
    correct = 0
    if args.dataset =="rafd" :
        with torch.no_grad():
            for inputs, labels, _ in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()
                scores, _, _ = models['backbone'](inputs)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        return 100 * correct / total
    else:
        with torch.no_grad():
            total_loss = 0
            for (inputs, labels) in dataloaders[mode]:
                
                inputs = inputs.cuda()
                labels = labels.cuda()

                _, feat, _ = models['backbone'](inputs,inputs,labels)
                # feat = models_b(inputs)
                scores = models['classifier'](feat)
                _, preds = torch.max(scores.data, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        
        return 100 * correct / total



iters = 0
def train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss):


    models['backbone'].train()
    if method == 'lloss':
        models['module'].train()
    global iters
    for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
        
            
        inputs = data[0].cuda()
        labels = data[1].cuda()


        iters += 1

        optimizers['backbone'].zero_grad()
        if method == 'lloss':
            optimizers['module'].zero_grad()

        scores, _, features = models['backbone'](inputs) 
        target_loss = criterion(scores, labels)
        # target_loss =  F.nll_loss(F.log_softmax(scores, dim=1), labels)
        if method == 'lloss':
            if epoch > epoch_loss:
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()

            pred_loss = models['module'](features)
            pred_loss = pred_loss.view(pred_loss.size(0))
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss + WEIGHT * m_module_loss 
        else:
            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)        
            loss            = m_backbone_loss
        # loss = target_loss
        loss.backward()
        optimizers['backbone'].step()
        if method == 'lloss':
            optimizers['module'].step()
    return loss

    
def train(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, args, subset, labeled_set, data_unlabeled):
    
    print('>> Train a Model.')

    best_acc = 0.
    
    for epoch in range(num_epochs):

        best_loss = torch.tensor([0.5]).cuda()
        loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)

        schedulers['backbone'].step(loss)
        if method == 'lloss':
            schedulers['module'].step(loss)

        if True and epoch % 20  == 1:

            acc = test(models, epoch, method, dataloaders, args, mode='test')

            if args.dataset == 'icvl':

                if best_acc > acc:
                    best_acc = acc
                    
                print('Val Error: {:.3f} \t Best Error: {:.3f}'.format(acc, best_acc))
            else:
                if best_acc < acc:
                    best_acc = acc
                    torch.save(models['backbone'].state_dict(), 'models/backbone.pth')
                print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')
    return best_acc




def train_epoch_ssl(models, method, criterion, optimizers, dataloaders, 
                                        epoch, epoch_loss, l_lab, l_ulab,schedulers, targets=False):
    models['backbone'].train()
    TRAIN_CLIP_GRAD = True
    idx = 0
    num_steps = len(dataloaders['train'])
    for (samples_1, samples_2) in tqdm(zip(dataloaders['unlabeled'], dataloaders['unlabeled2']), leave=False, total=len(dataloaders['unlabeled'])):
        
        samples_r = samples_1[0].cuda(non_blocking=True)
        samples_a = samples_2[0].cuda(non_blocking=True)
        
        contrastive_loss = models['backbone'](samples_a, samples_r)

        loss = (torch.sum(contrastive_loss)) / contrastive_loss.size(0)
        optimizers['backbone'].zero_grad()

        loss.backward()

        optimizers['backbone'].step()
        schedulers['backbone'].step_update(epoch * num_steps + idx)
        idx +=1

    return loss





def train_with_ssl(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, 
                           epoch_loss, args, l_lab, l_ulab, cycle):
    print('>> Train a Model.')
    best_acc = 0.
    if os.path.isfile('models/ssl_backbone.pth'):
        models['backbone'].load_state_dict(torch.load('models/ssl_backbone.pth'))
    for epoch in range(num_epochs):

        best_loss = torch.tensor([99]).cuda()
        loss = train_epoch_ssl(models, method, criterion, optimizers, dataloaders, 
                                        epoch, epoch_loss, l_lab, l_ulab, schedulers, True)

        
        if True and epoch % 20  == 1:
            acc = test_with_ssl(models, epoch, method, dataloaders, args, mode='test')
            print(loss.item())

            if best_acc < acc:
                best_acc = acc
            if best_loss > loss:
                best_loss = loss
                torch.save(models['backbone'].state_dict(), 'models/ssl_backbone.pth' )
                

            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

    print('>> Finished.')

    
    return best_acc 


def train_epoch_ssl2(models, method, criterion, optimizers, dataloaders, 
                                        epoch, schedulers, cycle, last_inter):
    models['backbone'].train()
    models['classifier'].train()
    TRAIN_CLIP_GRAD = True
    idx = 0
    num_steps = len(dataloaders['train'])
    c_loss_gain = 0.5 #- 0.05*cycle
    for (samples,samples_a) in tqdm(zip(dataloaders['train'],dataloaders['train2']), leave=False, total=len(dataloaders['train'])):
        
        samples_a = samples_a[0].cuda(non_blocking=True)
        samples_r = samples[0].cuda(non_blocking=True)
        targets   = samples[1].cuda(non_blocking=True)

        contrastive_loss, features, _ = models['backbone'](samples_a, samples_r, targets)

        if (idx % 2 ==0) or (idx <= last_inter):
            scores = models['classifier'](features)
            target_loss = criterion(scores, targets)
            t_loss = (torch.sum(target_loss)) / target_loss.size(0)
            c_loss = (torch.sum(contrastive_loss)) / contrastive_loss.size(0)
            loss = t_loss + c_loss_gain*c_loss
            # loss.backward()
        else:
            loss = c_loss_gain *(torch.sum(contrastive_loss)) / contrastive_loss.size(0)
        optimizers['backbone'].zero_grad()
        loss.backward()
        optimizers['backbone'].step()
        if (idx % 2 ==0) or (idx <= last_inter):
            optimizers['classifier'].zero_grad()
            optimizers['classifier'].step()
    
        idx +=1
    return loss

def train_with_ssl2(models, method, criterion, optimizers, schedulers, dataloaders, num_epochs, 
                           no_classes, args, labeled_data, unlabeled_data, data_train, cycle, last_inter, ADDENDUM):
    print('>> Train a Model.')
    best_acc = 0.
    arg = 0

    l_lab = 0
    l_ulab = 0
    # if not os.path.isfile('models/moby_backbone_full.pth'):
    for epoch in range(num_epochs):

        best_loss = torch.tensor([99]).cuda()
        # loss = train_epoch(models, method, criterion, optimizers, dataloaders, epoch, epoch_loss)
        loss = train_epoch_ssl2(models, method, criterion, optimizers, dataloaders, epoch, schedulers, cycle, last_inter)
        schedulers['classifier'].step(loss)
        schedulers['backbone'].step(loss)

        if True and epoch % 20  == 1:
            # acc = test_with_ssl(models, epoch, method, dataloaders, args, mode='test')
            # print(loss.item())
            # if epoch == 1:
            torch.save(models['backbone'].state_dict(), 'models/backbonehcss_%s_%d.pth'%(args.dataset,cycle))
            torch.save(models['classifier'].state_dict(), 'models/classifierhcss_%s_%d.pth'%(args.dataset,cycle))

            acc = test_without_ssl2(models, epoch, no_classes, dataloaders, args, cycle, mode='test')
            if best_acc < acc:
                torch.save(models['backbone'].state_dict(), 'models/backbonehcss_%s_%d.pth'%(args.dataset,cycle))
                torch.save(models['classifier'].state_dict(), 'models/classifierhcss_%s_%d.pth'%(args.dataset,cycle))
                best_acc = acc

            print('Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))

    print('>> Finished.')


    models['classifier'].eval()
    models['backbone'].eval()
    # # 
    features = np.empty((args.batch,512))
    # #     c_loss =  torch.tensor([]).cuda()
    k_var = 2
    c_loss_m = np.zeros((k_var, args.batch*len(dataloaders['unlabeled'])))

    state_dict = torch.load('models/backbonehcss_%s_%d.pth'%(args.dataset,cycle))

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('encoder') and not ( k.startswith('encoder.classifier') or  k.startswith('encoder_k') ):
            # remove prefix
            state_dict[k[len("encoder."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    if args.learner_architecture == "vgg16":
        models_b = resnet.dnn_16enc(no_classes).cuda()
    elif args.learner_architecture == "resnet18":
        models_b = resnet.ResNet18(no_classes).cuda()
    elif args.learner_architecture == "wideresnet28":
        models_b = resnet.Wide_ResNet28(no_classes).cuda()
    elif args.learner_architecture == "lenet5":
        models_b = LeNet5(no_classes,)
    models_b.load_state_dict(state_dict, strict=False)
    models_b.eval()
    combined_dataset = DataLoader(data_train, batch_size=args.batch, 
                                    sampler=SubsetSequentialSampler(unlabeled_data+labeled_data), 
                                    pin_memory=True, drop_last=False)

    for ulab_data in combined_dataset:
        
        unlab = ulab_data[0].cuda()
        # target = ulab_data[1].cuda()
        feat =  models_b(unlab)
        feat = feat.detach().cpu().numpy()
        feat = np.squeeze(feat)
        features = np.concatenate((features, feat), 0)

    features = features[args.batch:,:]
    subset = len(unlabeled_data)
    labeled_data_size = len(labeled_data)
    # Apply CoreSet for selection
    new_av_idx = np.arange(subset,(subset + labeled_data_size))
    sampling = kCenterGreedy(features)  
    batch = sampling.select_batch_(new_av_idx, ADDENDUM)
        # print(min(batch), max(batch))
    other_idx = [x for x in range(subset) if x not in batch]
    # np.save("selected_s.npy", batch)
    arg = np.array(other_idx + batch)


    return best_acc, arg
