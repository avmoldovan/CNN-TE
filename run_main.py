import argparse
import os
import sys
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torchvision.utils import save_image
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
from matplotlib.axes import *
import torchvision.models as models
from torch.utils.data import Subset, DataLoader, Dataset, TensorDataset
from torchvision.datasets import CIFAR10
import json, string
from RunConfig import settings
#from SGMsettings import settings
#import Cifar10CNN as ourcustom
#from PTConfig import settings
#import PTAlexNet as pretrainedourmodels
from SGD import SGD

from torch import optim



# from setuptools import setup, Extension
# from Cython.Build import cythonize
# import pyximport; pyximport.install()
# import subprocess
# ##from distutils.core import setup
# from distutils.extension import Extension
# ##module = Extension('addObservations', sources='./PyTE/add_te_sources.pyx', extra_compile_args=['language_level':'3'])
# #setup(name = "addObservations", ext_modules = cythonize("./PyTE/add_te_sources.pyx"))
from setuptools import setup, Extension
from Cython.Build import cythonize
#import PyTE.add_te_sources

#TODO: later
# import pyximport; pyximport.install()
# import subprocess
# subprocess.call(["cython", "-a", "./PyTE/add_te_sources.pyx"])


best_acc1 = 0
device = None


def main():
    plt.style.context('ggplot')
    #model_name = 'SVHN'
    #model_name = 'STL10'
    #model_name = 'fashionMNIST'
    #model_name = 'CIFAR10'
    model_name = 'USPS'
    config = settings[model_name]
    #config = settings['CIFAR10']
    #config = settings['CIFAR100']


    timestr = time.strftime("%Y%m%d-%H%M%S")
    filekey = os.path.join('./outputs/', config['run_title'] + "-" + timestr + "-" + config['suffix'])
    f = open(filekey + '.txt', 'w')
    # sys.stderr = f
    if (config['out_to_file'] == True):
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = f


    if(config['gpu'] != 'cpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if config['seed'] is not None:
        random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if config['gpu'] != 'cpu':
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if config['dist_url'] == "env://" and config['world_size'] == -1:
        config['world_size'] = int(os.environ["WORLD_SIZE"])

    #TODO: fix
    config['distributed'] = config['world_size'] > 1 or config['multiprocessing_distributed']

    ngpus_per_node = torch.cuda.device_count()
    if config['multiprocessing_distributed']:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        config['world_size'] = ngpus_per_node * config['world_size']
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        # Simply call main_worker function
        main_worker(ngpus_per_node, filekey, config, model_name)

    if (config['out_to_file'] == True):
        #plot_single_file(key, filekey)
        sys.stdout.flush()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


def main_worker(ngpus_per_node, filekey, configs, model_name):
    print(torch.__version__)

    gpu = configs['gpu']
    global best_acc1
    configs['gpu'] = gpu

    if configs['gpu'] != 'cpu':
        print("Use GPU: {} for training".format(configs['gpu']))

    if configs['distributed']:
        if configs['dist_url'] == "env://" and configs['rank'] == -1:
            configs['rank'] = int(os.environ["RANK"])
        if configs['multiprocessing_distributed']:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            configs['rank'] = configs['rank'] * ngpus_per_node + gpu
        dist.init_process_group(backend=configs['dist_backend'], init_method=configs['dist_url'],
                                world_size=configs['world_size'], rank=configs['rank'])
    # create model
    print("=> creating model '{}'".format(configs['arch']))

    if model_name == 'fashionMNIST':
        import FashionMNISTFULL as ourcustom
        classname = 'FashionMNISTFULL'
    elif model_name == 'STL10':
        import STL10CNN as ourcustom
        classname = 'STL10CNN'
    elif model_name == 'SVHN':
        import SVHNCNN as ourcustom
        classname = 'SVHNCNN'
    elif model_name == 'USPS':
        import USPSCNN as ourcustom
        classname = 'USPSCNN'
    elif model_name == 'CIFAR10' or model_name == 'CIFAR100':
        import Cifar10CNN as ourcustom
        classname = 'Cifar10CNN'

    model = eval('ourcustom.' + classname + '(configs=configs)')

    if configs['evaluate']:
        #if configs['pretrained']:
        if configs['pretrained_url'] is not None:
            print('[Evaluate] Using custom pretrained_url model')
            model = eval('ourcustom.' + classname + '(configs=configs)')
        else:
            #this will load the S3 imagenet
            print('[Evaluate] Loading the Imagenet S3 pretrained AlexNet...')
            #model = ourcustom.cifar10cnn(pretrained=True, progress=True)
            model = eval('ourcustom.' + classname + '(pretrained=True, progress=True)')
    else:
        #if configs['base_retrain']:
        if configs['partial_freeze']:
            if configs['pretrained_url'] is not None:
                print('[Partial train] Using custom pretrained_url model')
                #model = ourcustom.cifar10cnn(configs=configs)
                model = eval('ourcustom.' + classname + '(configs=configs)')
            else:
                print('[Partial train] training from scratch')
                #model = ourcustom.cifar10cnn(pretrained=False, progress=True)
                model = eval('ourcustom.' + classname + '(pretrained=False, progress=True)')
        else:
            print('[Base retrain] from scratch with configs')
            model = eval('ourcustom.' + classname + '(configs=configs)')

    if configs['gpu'] != 'cpu':
        model = model.to(device)

    if configs['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs['gpu'] != 'cpu':
            torch.cuda.set_device(configs['gpu'])
            model.cuda(configs['gpu'])
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs['batch_size'] = int(configs['batch_size'] / ngpus_per_node)
            configs['workers'] = int((configs['workers'] + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs['gpu']])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs['gpu'] != 'cpu':
        torch.cuda.set_device(configs['gpu'])
        model = model.cuda(configs['gpu'])
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if configs['arch'].startswith('alexnet') or configs['arch'].startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            if configs['gpu'] != 'cpu':
                model.cuda()
        else:
            if configs['gpu'] != 'cpu':
                model = torch.nn.DataParallel(model).cuda()
            else:
                model = torch.nn.DataParallel(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(configs['gpu'])

    if (configs['gpu'] == 'cpu'):
        model.module.criterion = criterion
        model.module.filekey = filekey
    else:
        model.criterion = criterion
        model.filekey = filekey

    #optimizer = SGD(model.parameters(), configs['lr'], momentum=configs['momentum'], weight_decay=configs['weight_decay'])
    #copy lr to calculated_lr first
    configs['calculated_lr'] = configs['lr']
    optimizer = optim.SGD(model.parameters(), lr=configs['lr'], momentum=configs['momentum'], weight_decay=configs['weight_decay'])

    # optionally resume from a checkpoint
    if configs['resume']:
        if os.path.isfile(configs['resume']):
            print("=> loading checkpoint '{}'".format(configs['resume']))
            if configs['gpu'] == 'cpu':
                checkpoint = torch.load(configs['resume'])
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(configs['gpu'])
                checkpoint = torch.load(configs['resume'], map_location=loc)
            configs['start_epoch'] = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if configs['gpu'] != 'cpu':
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(configs['gpu'])
            model.load_state_dict(checkpoint['state_dict'])
            model.filekey = filekey
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(configs['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(configs['resume']))

    cudnn.benchmark = True

    # # Data loading code
    # traindir = os.path.join(configs['data'], 'train')
    # valdir = os.path.join(configs['data'], 'val')

    if model_name == 'fashionMNIST':
        normalize = transforms.Normalize((0.1307), (0.3081))
    elif model_name == 'Cifar10' or model_name == 'Cifar100':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # picked from the blitz beginner tutorial
    transform = transforms.Compose(
        [transforms.ToTensor(),
         #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
    # transforms.RandomHorizontalFlip(),
    # transforms.ToTensor(),

    train_dataset = eval("datasets." + configs['dataset'] + "(root=configs['data'], download=True, transform=transform)")

    if configs['use_subset']:
        if configs['subset_classes'] != None:
            idx = (train_dataset.targets == 2) | (train_dataset.targets == 6)  # | (dataset_full.targets==5) | (dataset_full.targets==6)
            train_dataset.targets = train_dataset.targets[idx]
            train_dataset.data = train_dataset.data[idx]
        train_subset = torch.utils.data.Subset(train_dataset, range(0, configs['trainsubset']))
        train_loader = torch.utils.data.DataLoader(train_subset, batch_size=configs['batch_size'], shuffle=True, num_workers=2)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True, num_workers=2)

    if configs['distributed']:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_subset)
    else:
        train_sampler = None


    testset = eval("datasets." + configs['dataset'] + "(root=configs['data'], download=True, transform=transform)")


    if configs['use_subset']:
        if configs['subset_classes'] != None:
            idx2 = (testset.targets == 2) | (testset.targets == 6)
            testset.targets = testset.targets[idx]
            testset.data = testset.data[idx]
        testsubset = torch.utils.data.Subset(testset, range(0, int(configs['trainsubset'])))
        val_loader = torch.utils.data.DataLoader(testsubset, batch_size=configs['batch_size'], shuffle=True, num_workers=2)
    else:
        val_loader = torch.utils.data.DataLoader(testset, batch_size=configs['batch_size'], shuffle=True, num_workers=2)

    # clip end


    if configs['evaluate']:
        validate(val_loader, model, criterion, configs)
        return

    fb_avg_test_err = list()
    fb_avg_train_err = list()
    fb_avg_epochs = list()
    avg_test_err = list()
    avg_train_err = list()
    validf = pd.DataFrame(columns=["train_acc", "valid_acc", "batch", "lr", "tr", "time", "fb", "epoch"])
    allmetrics = pd.DataFrame(columns=["train_cost", "valid_cost", "train_acc1", "train_acc5", "valid_acc1", "valid_acc5", "batch", "lr", "calculated_lr", "tr1", "tr2", "time", "fb", "epoch"])

    for epoch in range(configs['start_epoch'], configs['epochs']):
        accfb_train, accfb_test = list(), list()  # training/test accuracy score
        start = dt.datetime.now()

        if configs['distributed']:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, configs)

        # train for one epoch
        top1avg, top5avg, lossesavg, progress = train(train_loader, model, criterion, optimizer, epoch, configs, metrics=allmetrics)

        # evaluate on validation set
        vacc1, vacc5, vlossesavg, vprogress = validate(val_loader, model, criterion, configs, metrics=allmetrics)

        # remember best acc@1 and save checkpoint
        is_best = vacc1 > best_acc1
        best_acc1 = max(vacc1, best_acc1)

        end = dt.datetime.now()
        tdelta = end - start

        allmetrics = allmetrics.append(
            {'train_cost': lossesavg,
             'valid_cost': vlossesavg,
             'train_acc1': top1avg.item(),
             'train_acc5': top5avg.item(),
             'valid_acc1': vacc1.item(),
             'valid_acc5': vacc5.item(),
             'epoch': epoch,
             'batch': configs['batch_size'],
             'lr': configs['lr'],
             'calculated_lr': configs['calculated_lr'],
             'tr1': configs['tr1'],
             'tr2': configs['tr2'],
             'time' : tdelta,
             'fb' : (not configs['base_retrain'])
        }, ignore_index = True, sort = False)

        if not configs['multiprocessing_distributed'] or \
                (configs['multiprocessing_distributed'] and configs['rank'] % ngpus_per_node == 0) and \
                configs['save_model']:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': configs['arch'],
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'configs' : configs,
            }, is_best, configs)

        if(configs['out_to_file'] == True):
            allmetrics.to_csv(filekey + '.csv', index=False)


def train(train_loader, model, criterion, optimizer, epoch, configs, metrics):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if configs['gpu'] != 'cpu':
            images = images.cuda(configs['gpu'], non_blocking=True)
        else:
            images = images.to(device, non_blocking=True)

        if configs['gpu'] != 'cpu':
            target = target.cuda(configs['gpu'], non_blocking=True)
        else:
            target = target.to(device, non_blocking=True)  # cuda(configs['gpu'], non_blocking=True)

        if (hasattr(model, 'module')):
            model.prev_epoch = model.module.current_epoch
            model.current_epoch = epoch
        else:
            model.module.prev_epoch = model.module.current_epoch
            model.module.current_epoch = epoch
        output = model(images, target)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        #optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        loss.backward()
        #model.fc7.weight = model.fc7.weight.matmul(0.01)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % configs['print_freq'] == 0:
            progress.display(i)

    return top1.avg, top5.avg, losses.avg, progress

def validate(val_loader, model, criterion, configs, metrics=None):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if configs['gpu'] != 'cpu':
                images = images.cuda(configs['gpu'], non_blocking=True)
            else:
                images = images.to(device, non_blocking=True)

            if configs['gpu'] != 'cpu':
                target = target.cuda(configs['gpu'], non_blocking=True)
            else:
                target = target.to(device, non_blocking=True)  # cuda(configs['gpu'], non_blocking=True)

            # compute output
            output = model(images, target)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % configs['print_freq'] == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg, progress


def save_checkpoint(state, is_best, configs):
    filename = 'checkpoint.pth.tar'

    if configs is not None:
        filename = str(configs['run_title']) + '-' + filename

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, str(configs['run_title']) + '-' + 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, configs):
    """Sets the learning rate to the initial LR decayed by 10 every 5 epochs"""
    if (configs['lr_decay'] > 0):
        #lr = configs['lr'] * (0.1 ** (epoch // 5))
        lr = configs['lr'] * (configs['lr_decay'] ** (epoch // configs['lr_decay_epochs']))
        configs['calculated_lr'] = lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

if __name__ == '__main__':
    main()