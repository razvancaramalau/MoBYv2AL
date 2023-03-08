'''
Visual Transformer for Task-aware Active Learning
'''
# Python
import os
import sys
import random
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.models as models
import argparse
# import yaml
# Custom
import models.resnet as resnet
from models.lenet import LeNet5
from models.build_ssl_model import build_model

from models.query_models import LossNet
from train_test import train, train_with_ssl, train_with_ssl2
from load_dataset import load_dataset
from selection_methods import query_samples
# from config import *
from sampler import SubsetSequentialSampler

sys.path.append(".")


parser = argparse.ArgumentParser()
parser.add_argument("-d","--dataset", type=str, default="cifar10",
                    help="")
parser.add_argument("-e","--no_of_epochs", type=int, default=222, 
                    help="Number of epochs for the active learner")
parser.add_argument("-m","--method_type", type=str, default="mobyv2al",
                    help="")
parser.add_argument("-c","--cycles", type=int, default=10,
                    help="Number of active learning cycles")
parser.add_argument("-r","--trials", type=int, default=5,
                    help="Number of AL trials to average over the cycles")
parser.add_argument("-t","--total", type=bool, default=False,
                    help="Training on the entire dataset")
parser.add_argument("-s","--ssl", type=bool, default=True,
                    help="")
parser.add_argument("-ss","--selection_subset", type=int, default=10000,
                    help="This is the random pre-selection to avoid redundancy.")
parser.add_argument("-la","--learner_architecture", type=str, default="resnet18",
                    help="")
parser.add_argument("-b","--batch", type=int, default=128,
                    help="Batch size used for training")
args = parser.parse_args()

##
# Main
if __name__ == '__main__':
    drop_flag = True
    if args.method_type=="mobyv2al":
        args.ssl = True
    else:
        args.ssl = False
    CUDA_VISIBLE_DEVICES = 0
    MILESTONES = [60,120,160]
    MOMENTUM = 0.9
    WDECAY = 5e-4
    LR = 1e-2
    EPOCHL = 120
    SUBSET    = args.selection_subset
    method = args.method_type
    BATCH = args.batch
    TRIALS = args.trials
    methods = ['Random', 'CoreSet', 'mobyv2al']
    datasets = ['cifar10', 'cifar100', 'fashionmnist','svhn','svhn5']
    learner_models = ["vgg16","resnet18","lenet5","wideresnet28"]
    assert method in methods, 'No method %s! Try options %s'%(method, methods)
    assert args.dataset in datasets, 'No dataset %s! Try options %s'%(args.dataset, datasets)
    '''
    method_type: 'Random', 'CoreSet', 'mobyv2al'
    '''

    results = open('results_'+str(args.method_type)+"_"+args.dataset +'_main'+str(args.cycles)+
                    str(args.total)+'.txt','w')
    print("Dataset: %s"%args.dataset)
    print("Method type:%s"%method)
    if args.total:
        TRIALS = 1
        CYCLES = 1
    else:
        CYCLES = args.cycles

    # print(args.visual_transformer)
    for trial in range(TRIALS):

        # Load training and testing dataset
        data_train, data_unlabeled, data_test, NO_CLASSES, no_train, data_train2, data_unlabeled2 = load_dataset(args.dataset, args.ssl)
                
        print(len(data_train))

        NUM_TRAIN = no_train
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)

        if args.total:
            labeled_set= indices
            unlabeled_set = [x for x in range(0, NUM_TRAIN)]
        else:
            if args.dataset=='fashionmnist':
                if os.path.isfile("init_set_fm.npy"):
                    ADDENDUM = 100
                    init_margin = 100
                    labeled_set = np.load("init_set_fm.npy").tolist()
                else:
                    labeled_set = indices[:100] 
                    np.save("init_set_fm.npy", np.asarray(labeled_set))  
            elif args.dataset=='svhn':              
                # take 1000 of the labelled data at first run
                ADDENDUM = 1000
                init_margin = 1000
                if os.path.isfile("init_set_svhn.npy"):
                    labeled_set = np.load("init_set_svhn.npy").tolist()
                else:
                    labeled_set = indices[:1000] 
                    np.save("init_set_svhn.npy", np.asarray(labeled_set))
                                # take 10% of the labelled data at first run
            elif args.dataset=='svhn5':              
                # take 1000 of the labelled data at first run
                ADDENDUM = 1000
                init_margin = 1000
                if os.path.isfile("init_set_svhn5.npy"):
                    labeled_set = np.load("init_set_svhn5.npy").tolist()
                else:
                    labeled_set = indices[:1000] 
                    np.save("init_set_svhn5.npy", np.asarray(labeled_set))                             
            else:
                ADDENDUM = 2500
                init_margin = int(NUM_TRAIN/10)
                if os.path.isfile("init_set.npy"):
                    # take 10% of the labelled data at first run for CIFAR10/100
                    labeled_set = np.load("init_set.npy").tolist()
                else:
                    labeled_set = indices[:5000] 
                    np.save("init_set.npy", np.asarray(labeled_set))
            print(ADDENDUM)
            unlabeled_set = set(indices) - set(labeled_set) #[x for x in indices if x not in labeled_set]
            unlabeled_set = list(unlabeled_set)

        lab_loader = DataLoader(data_train, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True, drop_last=drop_flag)
        test_loader  = DataLoader(data_test, batch_size=BATCH, drop_last=drop_flag)
        if args.ssl:
            lab_loader2 = DataLoader(data_train2, batch_size=BATCH, 
                                    sampler=SubsetSequentialSampler(labeled_set), 
                                    pin_memory=True, drop_last=drop_flag)

            dataloaders  = {'train': lab_loader, 'train2': lab_loader2, 'test': test_loader}
        else:
            dataloaders  = {'train': lab_loader, 'test': test_loader}
        
        


        for cycle in range(args.cycles):
            
            # Randomly sample SUBSET unlabeled data points
            if not args.total:
                if cycle == 0:
                    random.shuffle(unlabeled_set)
                if args.ssl :
                    if drop_flag:
                        if SUBSET < len(labeled_set):
                            SUBSET = len(labeled_set)
                        k = int(SUBSET/BATCH)
                        SUBSET = k * BATCH
                    subset = unlabeled_set[:SUBSET]
                else:
                    if SUBSET > len(unlabeled_set):
                        SUBSET = len(unlabeled_set)
                    if drop_flag:
                        k = int(SUBSET/BATCH)
                        SUBSET = k * BATCH
                    subset = unlabeled_set[:SUBSET]
            else:
                subset = unlabeled_set
            unlab_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset), 
                                            pin_memory=True, drop_last=drop_flag)

            if (args.method_type == "mobyv2al"):
                # Interleave labelled and unlabelled batches.
                if len(subset)>len(labeled_set):
                    interleaved_size = 2 * int(len(labeled_set)/BATCH) * BATCH
                else:
                    interleaved_size = 2 * int(len(subset)/BATCH) * BATCH

                interleaved = np.zeros((interleaved_size)).astype(int)
                if len(labeled_set)>len(subset):
                    l_mixed_set = len(subset)

                else:
                    l_mixed_set = len(labeled_set) 
                    
                for cnt in range(2*int(l_mixed_set/BATCH)):
                    idx = int(cnt / 2)
                    if cnt % 2 == 0:
                        interleaved[cnt*BATCH:(cnt+1)*BATCH] = labeled_set[idx*BATCH:(idx+1)*BATCH]                             
                    else:
                        interleaved[cnt*BATCH:(cnt+1)*BATCH] = subset[idx*BATCH:(idx+1)*BATCH] 

                interleaved = interleaved.tolist()
                # if len(subset)>len(labeled_set):
                #     interleaved = interleaved + subset[(idx+1)*BATCH:]
                # else:
                #     interleaved = interleaved + labeled_set[(idx+1)*BATCH:]
                last_interleaved = idx
                if args.ssl:
                    lab_loader = DataLoader(data_train, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(interleaved), 
                                            pin_memory=True, drop_last=drop_flag)
                    lab_loader2 = DataLoader(data_train2, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(interleaved), 
                                            pin_memory=True, drop_last=drop_flag)
                    unlab_loader2 = DataLoader(data_unlabeled2, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset), 
                                            pin_memory=True, drop_last=drop_flag)
                    unlab_loader = DataLoader(data_unlabeled, batch_size=BATCH, 
                                            sampler=SubsetSequentialSampler(subset), 
                                            pin_memory=True, drop_last=drop_flag)
                    dataloaders  = {'train': lab_loader, 'train2': lab_loader2, 
                                    'test': test_loader, 'unlabeled': unlab_loader, 'unlabeled2': unlab_loader2}
            else:
                dataloaders  = {'train': lab_loader, 
                            'test': test_loader, 'unlabeled': unlab_loader}

            # Model - create new instance for every cycle so that it resets
            with torch.cuda.device(CUDA_VISIBLE_DEVICES):
                if args.method_type == 'mobyv2al':
                    model = build_model(NO_CLASSES, 'moby', args.learner_architecture, BATCH, interleaved).cuda()
                    if args.learner_architecture == "vgg16":
                        dim_latent = 512
                        classifier = resnet.VGGNet2C(dim_latent, NO_CLASSES).cuda()
                    elif args.learner_architecture == "lenet5":
                        dim_latent = 400
                        classifier = resnet.ResNetC(dim_latent, num_classes=NO_CLASSES).cuda()
                    elif args.learner_architecture == "wideresnet28":
                        dim_latent = 640
                        classifier = resnet.ResNetC(dim_latent, num_classes=NO_CLASSES).cuda()
                    else:
                        dim_latent = 512
                        classifier = resnet.ResNetC(dim_latent, num_classes=NO_CLASSES).cuda()
                else:
                    if args.learner_architecture == "vgg16":
                        model = resnet.dnn_16(NO_CLASSES).cuda()
                    elif args.learner_architecture == "resnet18":
                        model = resnet.ResNet18(NO_CLASSES).cuda()
                    elif args.learner_architecture == "wideresnet28":
                        model = resnet.Wide_ResNet28(NO_CLASSES).cuda()
                    elif args.learner_architecture == "lenet5":
                        model = LeNet5(NO_CLASSES,)

                    no_param = 0
                    for parameter in model.parameters():
                        a = parameter.reshape(-1).size()
                        no_param += a[0]
                    print(no_param)
                if method == 'lloss':
                    loss_module = LossNet().cuda()

            models      = {'backbone': model}
            if method =='lloss':
                models = {'backbone': model, 'module': loss_module}
            
            torch.backends.cudnn.benchmark = True

            Nes_flag = False        
            criterion      = nn.CrossEntropyLoss(reduction='none')
            optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR*1, 
               momentum=MOMENTUM, weight_decay=WDECAY, nesterov=Nes_flag)

            sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
            num_steps =  int( args.no_of_epochs * len(subset) / BATCH)
            warmup_steps = int( 10 * len(subset) / BATCH)

            optimizers = {'backbone': optim_backbone}
            schedulers = {'backbone': sched_backbone}
            if method == 'lloss':
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                    momentum=MOMENTUM, weight_decay=WDECAY)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)
                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module}

            if args.ssl:
                models      = {'backbone': model, 'classifier': classifier}
                num_steps =  int( args.no_of_epochs *  (init_margin + ADDENDUM*cycle) / BATCH)
                warmup_steps = int( 10 * (init_margin + ADDENDUM*cycle) / BATCH)
                optim_classifier = optim.SGD(models['classifier'].parameters(), lr=0.01, 
                                    momentum=0.9, weight_decay=0.0, nesterov=Nes_flag)

                sched_classifier = lr_scheduler.MultiStepLR(optim_classifier, milestones=MILESTONES)

                optimizers = {'backbone': optim_backbone, 'classifier': optim_classifier}
                schedulers = {'backbone': sched_backbone, 'classifier': sched_classifier} 
          
            # Training and testing

            if args.ssl:
                if args.total:
                    acc = train_with_ssl(models, method, criterion, optimizers, schedulers, dataloaders, 
                                        args.no_of_epochs, EPOCHL, args, labeled_set, 
                                        data_unlabeled, cycle)
                else:
                    acc, arg = train_with_ssl2(models, method, criterion, optimizers, schedulers, dataloaders, 
                                    args.no_of_epochs, NO_CLASSES, args, labeled_set, 
                                    subset, data_train, cycle, last_interleaved, ADDENDUM, dim_latent)

            else:
                acc = train(models, method, criterion, optimizers, schedulers, dataloaders, 
                            args.no_of_epochs, EPOCHL, args, unlabeled_set, labeled_set, data_unlabeled)

            print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            np.array([method, trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc]).tofile(results, sep=" ")
            results.write("\n")


            if cycle == (CYCLES-1):
                # Reached final training cycle
                print("Finished.")
                break
            # Get the indices of the unlabeled samples to train on next cycle
            if (args.method_type != "mobyv2al"):
                arg = query_samples(models, method, data_unlabeled, subset, labeled_set, cycle, args, drop_flag, ADDENDUM)

            # Update the labeled dataset and the unlabeled dataset, respectively
            labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
            dataloaders['train'] = DataLoader(data_train, batch_size=BATCH, 
                                        sampler=SubsetRandomSampler(labeled_set), 
                                        pin_memory=True, drop_last=drop_flag)
            #listd = list(torch.tensor(subset)[arg][:ADDENDUM].numpy()) 
            unlabeled_set = [x for x in range(NUM_TRAIN) if x not in labeled_set]
            #unlabeled_set =  [x for x in unlabeled_set if x not in listd]
            random.shuffle(unlabeled_set)

            #unlabeled_set = listd + unlabeled_set
            print(len(labeled_set), min(labeled_set), max(labeled_set))

    results.close()
