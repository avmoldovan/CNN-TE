from torchvision import transforms, datasets as ds
from torch.utils.data import DataLoader
import torchvision as tv
from torchvision.transforms import ToPILImage
import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn
# import matplotlib.pyplot as plt
import numpy as np
from torchvision.models import AlexNet
import torch
from torch.tensor import *
import torch.nn as nn
from skimage.util import view_as_windows
import numpy as np
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
# from fastai.callbacks.hooks import *
#from PyTE.TEDiscrete import TEDiscrete
from PyTE.TEWindow import TEWindow
from NNutils import *
from collections import OrderedDict
import math
# from TEDiscreteOnline import TEDiscreteOnline
# from FBLinear import FBLinear
import itertools
import random
import pandas as pd
import datetime as dt
import sys

# from TorchStandardScaler import *

__all__ = ['Cifar10CNN', 'cifar10cnn']

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}

DEBUG = False

class Context(object):

    def __init__(self):
        super(Context, self).__init__()
        self.ctx = None

    @classmethod
    def create(cls, value):
        if cls.ctx is not None:
            return cls.ctx
        else:
            cls.ctx = Context()
        return cls.ctx

    @property
    def ctx(self):
        return self.ctx


class Cifar10CNN(AlexNet):
    class GradUpdateFunc(torch.autograd.Function):
        """
        We can implement our own custom autograd Functions by subclassing
        torch.autograd.Function and implementing the forward and backward passes
        which operate on Tensors.
        """

        @staticmethod
        def forward(ctx, input):
            """
            In the forward pass we receive a Tensor containing the input and return
            a Tensor containing the output. ctx is a context object that can be used
            to stash information for backward computation. You can cache arbitrary
            objects for use in the backward pass using the ctx.save_for_backward method.
            """
            ctx.save_for_backward(input)
            return input  # input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            """
            In the backward pass we receive a Tensor containing the gradient of the loss
            with respect to the output, and we need to compute the gradient of the loss
            with respect to the input.
            """
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            # grad_input[input < 0] = 0
            grad_input = torch.zeros(grad_input.shape)

            return grad_input

    def __str__(self):
        return "Cifar10CNN"

    def __init__(self, configs):
        super(AlexNet, self).__init__()
        self.configs = configs
        self.batch_size = int(configs['batch_size'])
        self.skip_first = int(configs['skip_first'])  # 9#int(self.batch_size)
        self.g1 = float(configs['tr1'])
        self.g2 = float(configs['tr2'])
        self.te_length = int(self.configs['te_length'])
        self.withTE = bool(self.configs['withte'])
        self.fwd = bool(self.configs['forward'])
        DEBUG = configs['debug']
        self.conv_te = True
        self._forward_passes = 0
        self.clean_window = bool(configs['clean_window'])
        if self.batch_size == self.te_length or self.te_length == 0:
            self.windowed = False
        else:
            self.windowed = True
        self.window_batches = int(self.te_length / self.batch_size)
        self.gpu = self.configs['gpu'] != 'cpu'

        self.prev_epoch = 0
        self.current_epoch = 0

        # self.activations_size = math.floor(self.te_batch / configs['te_events_batch_multiple'] / configs[
        #   'batch_size']) - self.skip_first  ##this must be a batch_size multiple
        # self.activations_size = math.floor(self.te_batch / configs['batch_size'] ) - self.skip_first ##this must be a batch_size multiple
        # else:
        #     self.activations_size = math.floor(int(configs['epochs']) * \
        #                         int(configs['trainingset_size']) / \
        #                         int(configs['batch_size']))

        if configs['partial_freeze'] and not configs['pretrained'] and not configs['evaluate']:
            self.conv_te = False
        else:
            if not configs['base_retrain']:
                self.conv_te = True
            else:
                self.conv_te = False

        # TODO: remove this
        self.conv_te = True

        layers = []
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv1.name = 'conv1'
        layers.append(self.conv1)

        self.batchnorm2d1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d1.name = 'batchnorm2d1'
        layers.append(self.batchnorm2d1)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu1.name = 'relu1'
        layers.append(self.relu1)

        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2.name = 'conv2'
        layers.append(self.conv2)

        self.batchnorm2d2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d2.name = 'batchnorm2d2'
        layers.append(self.batchnorm2d2)

        self.relu2 = nn.ReLU(inplace=True)
        self.relu2.name = 'relu2'
        layers.append(self.relu2)

        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.maxPool1.name = 'maxPool1'
        layers.append(self.maxPool1)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3.name = 'conv3'
        layers.append(self.conv3)

        self.batchnorm2d3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d3.name = 'batchnorm2d3'
        layers.append(self.batchnorm2d3)

        self.relu3 = nn.ReLU(inplace=True)
        self.relu3.name = 'relu3'
        layers.append(self.relu3)

        # self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        # self.maxPool3.name = 'maxPool3'
        # layers.append(self.maxPool3)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4.name = 'conv4'
        layers.append(self.conv4)

        self.batchnorm2d4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d4.name = 'batchnorm2d4'
        layers.append(self.batchnorm2d4)

        self.relu4 = nn.ReLU(inplace=True)
        self.relu4.name = 'relu4'
        layers.append(self.relu4)

        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.maxPool2.name = 'maxPool2'
        layers.append(self.maxPool2)

        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv5.name = 'conv5'
        layers.append(self.conv5)

        self.batchnorm2d5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d5.name = 'batchnorm2d5'
        layers.append(self.batchnorm2d5)

        self.relu5 = nn.ReLU(inplace=True)
        self.relu5.name = 'relu5'
        layers.append(self.relu5)

        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv6.name = 'conv6'
        layers.append(self.conv6)

        self.batchnorm2d6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d6.name = 'batchnorm2d6'
        layers.append(self.batchnorm2d6)

        self.relu6 = nn.ReLU(inplace=True)
        self.relu6.name = 'relu6'
        layers.append(self.relu6)

        self.maxPool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.maxPool5.name = 'maxPool5'
        layers.append(self.maxPool5)

        self.conv7 = nn.Conv2d(512, 1024, kernel_size=3, stride=1)
        self.conv7.name = 'conv7'
        layers.append(self.conv7)

        self.batchnorm2d7 = nn.BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=False, track_running_stats=True)
        self.batchnorm2d7.name = 'batchnorm2d7'
        layers.append(self.batchnorm2d7)

        self.relu7 = nn.ReLU(inplace=True)
        self.relu7.name = 'relu7'
        layers.append(self.relu7)

        self.maxPool6 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.maxPool6.name = 'maxPool6'
        layers.append(self.maxPool6)

        self.features = nn.Sequential(*layers)

        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2)) ## added later
        # self.avgpool.name = 'avgpool'

        # self.drop6 = nn.Dropout(p=configs['dropout1'])
        # self.drop6.name = 'drop6'
        #
        # self.fc6 = nn.Linear(16 * 5 * 5, 120)
        # self.fc6.name = 'fc6'
        #
        # self.relu6 = nn.ReLU(inplace=True)
        # self.relu6.name = 'relu6'
        #
        # self.fc7 = nn.Linear(120, 84)
        # self.fc7.name = 'fc7'
        #
        # self.relu7 = nn.ReLU(inplace=True)
        # self.relu7.name = 'relu7'
        # self.fc7activ = torch.zeros(size=(self.fc7.out_features, 0), dtype=torch.uint8)  # .bool()
        # self.drop7 = nn.Dropout(p=configs['dropout2'])
        # self.drop7.name = 'drop7'

        self.fc8 = nn.Linear(1024, self.configs['num_classes'])
        self.fc8.name = 'fc8'
        #self.fc8activ = torch.zeros(size=(self.fc8.out_features, 0), dtype=torch.uint8, device='cpu')  # .bool()
        self.fc8activ = np.zeros(shape=(0, self.fc8.out_features), dtype=np.uint8)  # .bool()

        # if backward==False:
        #     self.hook = module.register_forward_hook(self.hook_fn)
        # else:
        self.hook0 = self.conv7.weight.register_hook(self.hook0_fn)
        self.hook = self.fc8.weight.register_hook(self.hook_fn)
        # self.hookF = Hook(self.fc8)
        # self.hookB = Hook(self.fc8, backward=True)

        self.softmax = nn.Softmax(dim=1)
        #self.softmaxactiv = torch.zeros(size=(self.fc8.out_features, 0), dtype=torch.uint8)  # .bool()
        self.softmaxactiv = np.zeros(shape=(0, self.fc8.out_features), dtype=np.uint8)  # .bool()

        # self.fc7tes = torch.zeros(size=(self.fc7.out_features, self.fc8.in_features), dtype=torch.float32)  # bool()
        # self.fc7activ = torch.zeros(size=(self.fc7.out_features, 0), dtype=torch.uint8)  # .bool()
        # self.fc7eye = torch.eye(self.fc7tes.shape[0], self.fc7tes.shape[1], dtype=torch.float32)
        # self.fc7pairidx = torch.cartesian_prod(torch.arange(end=self.fc7.out_features), torch.arange(end=self.fc8.out_features))

        self.fc8pairidx = torch.cartesian_prod(torch.arange(end=self.fc8.out_features), torch.arange(end=self.fc8.out_features))
        self.fc8tes = torch.zeros(size=(self.fc8.out_features, self.fc8.out_features), dtype=torch.float32)  # bool()
        self.fc8eye = torch.eye(self.fc8tes.shape[0], self.fc8tes.shape[1], dtype=torch.float32)

        #### original model from here: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)

        # initialize fcpai1 TE
        self.averages = dict()
        self.averages['fcpair1'] = dict()
        self.averages['fcpair2'] = dict()
        self.averages['fcpair3'] = dict()

        #self.inmatrix2 = torch.zeros(size=(self.fc7.out_features, self.fc8.out_features), dtype=torch.float32)
        self.inmatrix3 = torch.zeros(size=(self.fc8.out_features, self.fc8.out_features), dtype=torch.float32)

        self.grad_update = Cifar10CNN.GradUpdateFunc.apply

        if configs['partial_freeze'] == True:
            child_counter = 0
            for child in self.children():
                if child_counter < 20:
                    # print("child ", child_counter, " was frozen")
                    # for children_of_child in child.children():
                    for param in child.parameters():
                        param.requires_grad = False
                child_counter += 1

        self.start = dt.datetime.now()

        self.evol = getEvolutionDataFrame()

        self.logrow = getLogRowJson(self.g1, self.g2, self.te_length)

    def new_epoch(self):
        return self.current_epoch != self.prev_epoch

    def forward(self, x, target):
        start = dt.datetime.now()

        self.last_target = target
        if self.training and self.new_epoch() and self.conv_te:
            self._forward_passes = 0

            # self.fc7tes = torch.zeros(size=(self.fc7.out_features, self.fc8.in_features), dtype=torch.float32)  # bool()
            # self.fc7activ = torch.zeros(size=(self.fc7.out_features, 0), dtype=torch.uint8)  # .bool()

            self.fc8tes = torch.zeros(size=(self.fc8.out_features, self.fc8.out_features), dtype=torch.float32)  # bool()
            self.fc8activ = np.zeros(shape=(0, self.fc8.out_features), dtype=np.uint8) #torch.zeros(size=(self.fc8.out_features, 0), dtype=torch.uint8)  # .bool()
            self.softmaxactiv = np.zeros(shape=(0, self.fc8.out_features), dtype=np.uint8)#torch.zeros(size=(self.fc8.out_features, 0), dtype=torch.uint8)  # .bool()

            # if (self.configs['rolling_te_window'] == False):
            self.averages['fcpair2'] = dict()
            self.averages['fcpair3'] = dict()

        # x = self.conv1(x)
        # x = self.relu1(x)
        #
        # x = self.maxPool1(x)
        #
        # x = self.conv2(x)
        # x = self.relu2(x)
        #
        # x = self.maxPool2(x)

        # x = self.avgpool(x) #added later
        # x = torch.flatten(x, 1)

        x = self.features(x)
        #x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.size(0), -1)

        # x = self.fc6(x)
        # x = self.relu6(x)
        #
        # x = self.fc7(x)
        # self.fc7output = x.clone()
        # x = self.relu7(x)

        #fc7med = None
        fc8med = None
        softmaxmed = None

        # if self.training and self.conv_te and self.withTE:
        #     with torch.no_grad():
        #         # xb = x[-1,:].clone().detach().to(dtype=torch.uint8)
        #         xb = x.clone().detach()
        #         # fc7med = xb.mean()
        #         # xb[xb >= self.g1] = 1
        #         # xb[xb < self.g1] = 0
        #         xb[(xb <= -self.g1) | (xb >= self.g1)] = 1
        #         xb[(xb != 1.)] = 0
        #
        #         # if self.configs['gpu'] != None:
        #         # self.fc8activ = torch.cat((self.fc8activ.t().cuda(), toadd.cuda()), dim=0).t()
        #         # else:
        #         # self.fc8activ[:,self.fc8activ.shape[1] - 1 - self._forward_passes * self.configs['batch_size']:self.fc8activ.shape[1] - 1] = xb.t()
        #
        #         # if self._forward_passes < self.skip_first:
        #         #     if self.configs['gpu'] != None:
        #         #         self.fc8activ = torch.cat((self.fc8activ.t().cuda(), xb.cuda()), dim=0).t()
        #         #     else:
        #         #         self.fc8activ = torch.cat((self.fc8activ.t(), xb), dim=0).t()
        #         # else:
        #         #     self.fc8activ = xb.t()
        #
        #         if self.configs['gpu'] != 'cpu':
        #             self.fc7activ = xb.to(dtype=torch.uint8).cuda().t()  # self.fc8activ = torch.cat((self.fc8activ.t().cuda(), xb.to(dtype=torch.uint8).cuda()), dim=0).t()
        #         else:
        #             self.fc7activ = xb.to(dtype=torch.uint8).t()  # self.fc8activ = torch.cat((self.fc8activ.t(), xb.to(dtype=torch.uint8)), dim=0).t()

        x = self.fc8(x)
        # x = self.grad_update(x)
        self.fc8output = x.clone()

        if self.training and self.conv_te and self.withTE:
            with torch.no_grad():
                # xb = x[-1,:].clone().detach().to(dtype=torch.uint8)
                xb = x.clone().detach().cpu()
                # fc8med = xb.mean()
                # xb[xb >= self.g1] = 1
                # xb[xb < self.g1] = 0
                xb[(xb <= -self.g1) | (xb >= self.g1)] = 1
                xb[(xb != 1.)] = 0

                self.fc8activ = prepare_activations(self.configs, self.fc8activ, xb.to(dtype=torch.uint8).numpy(), self.windowed)

        self.softmaxoutput = self.softmax(x)

        if self.training and self.conv_te and self.withTE:
            with torch.no_grad():
                xb = self.softmaxoutput.clone().detach().cpu()

                xb[xb >= self.g2] = 1  # (self.g2 / self.configs['num_classes'])] = 1
                xb[xb != 1] = 0  # (self.g2 / self.configs['num_classes'])] = 0

                self.softmaxactiv = prepare_activations(self.configs, self.softmaxactiv, xb.to(dtype=torch.uint8).numpy(), self.windowed)

        ce = self.criterion(x, target)
        self.last_x = x

        if self.fwd and self.training and self.conv_te and (self.configs['fc8rate'] == 0 or self.configs['fc8rate'] is None) and \
                (self._forward_passes * self.batch_size) >= self.te_length and self.withTE:
            with torch.no_grad():
                update_with_te(self,
                    keypair='fcpair3',
                    layerleft=self.fc8,
                    layerleftactiv=self.fc8activ,
                    layertes=self.fc8tes,
                    CE=0.,
                    eye=self.fc8eye,
                    gradinput=None,
                    layerright=self.softmax,
                    layerrightactiv=self.softmaxactiv,
                    layerright_output=self.softmaxoutput.clone().detach(),
                    configs=self.configs)

        # if self.training and self._forward_passes > 0 and self.conv_te:
        #     if ((self._forward_passes + 1) * self.configs['batch_size']) >= self.configs['te_length']:
        #
        #         #layers 7-8
        #         # selection = None
        #         # if self.fc7activ.shape[1] > self.configs['te_length']:
        #         #     if self.configs['batch_size'] + self.configs['te_length'] < self.fc7activ.shape[1]:
        #         #         self.fc7activ = self.fc7activ.narrow(1, self.configs['batch_size'], self.configs['te_length'])
        #         #         self.fc8activ = self.fc8activ.narrow(1, self.configs['batch_size'], self.configs['te_length'])
        #         #         self.softmaxactiv = self.softmaxactiv.narrow(1, self.configs['batch_size'], self.configs['te_length'])
        #         #
        #         # if self.configs['fc7rate'] == 0 or self.configs['fc7rate'] is None:
        #         #     self.calc_fc('fcpair2', self.fc7activ, self.fc8activ)
        #         #     self.add_te_to('fcpair2', self.fc7tes, self.fc7, self.fc7eye)
        #         # else:
        #         #     selection = self.getRandomIndices(self.configs['fc7rate'], self.inmatrix2)
        #         #     self.calc_fc_sel('fcpair2', self.fc7activ, self.fc8activ, selection)
        #         #     self.add_te_to_sel(selection, 'fcpair2', self.fc7tes, self.fc7, self.fc7eye)
        #
        #         #layers 8-softmax
        #         selection = None
        #         # if self.fc8activ.shape[1] > self.configs['te_length']:
        #         #     if self.configs['batch_size'] + self.configs['te_length'] < self.fc8activ.shape[1]:
        #         #         self.fc8activ = self.fc8activ.narrow(1, self.configs['batch_size'], self.configs['te_length'])
        #         #         self.softmaxactiv = self.softmaxactiv.narrow(1, self.configs['batch_size'], self.configs['te_length'])
        #
        #         if self._forward_passes > self.skip_first:
        #             if self.configs['fc8rate'] == 0 or self.configs['fc8rate'] is None:
        #                 self.calc_fc('fcpair3', self.fc8activ, self.softmaxactiv)
        #                 self.add_te_to('fcpair3', self.fc8tes, self.fc8, self.fc8eye)
        #             else:
        #                 selection = self.getRandomIndices(self.configs['fc8rate'], self.inmatrix3)
        #                 self.calc_fc_sel('fcpair3', self.fc8activ, self.softmaxactiv, selection)
        #                 self.add_te_to_sel(selection, 'fcpair3', self.fc8tes, self.fc8, self.fc8eye)
        #
        #         if (self.configs['rolling_te_window'] == False):
        #             self.averages['fcpair2'] = dict()
        #             self.averages['fcpair3'] = dict()


        if self.training:
            updateLogRow(self, ce)
            self._forward_passes += 1

        return x

    def hook0_fn(self, grad):
        # self.input = input
        # self.output = output

        #         if self.training and self._forward_passes > 0 and self.conv_te:
        #             # if ((self._forward_passes + 1) * self.configs['batch_size']) >= self.configs['te_length']:
        #             if self._forward_passes > self.skip_first:
        #                 if self.configs['fc7rate'] == 0 or self.configs['fc7rate'] is None:
        #                     self.update_with_te(keypair='fcpair2',
        #                                         layerleft=self.fc7,
        #                                         layerleftactiv=self.fc7activ,
        #                                         layertes=self.fc7tes,
        #                                         CE=0.,
        #                                         eye=self.fc7eye,
        #                                         layerright=self.fc8,
        #                                         layerrightactiv=self.fc8activ,
        #                                         layerright_output=self.fc8output.clone().detach())

        updateLogGrad(self, grad)

        # if self.fc7 != None and self.fc7.weight != None and self.fc7.weight.grad != None:
        #     return self.fc7.weight.grad.data
        sys.stdout.flush()

    def hook_fn(self, grad):
        input = grad
        output = grad

        #self._forward_passes > self.skip_first:
        if self.fwd is False and self.training and self.conv_te and \
                (self.configs['fc8rate'] == 0 or self.configs['fc8rate'] is None) and \
            (self._forward_passes * self.batch_size) >= self.te_length and self.withTE:
            with torch.no_grad():
                update_with_te(self,
                               keypair='fcpair3',
                               layerleft=self.fc8,
                               layerleftactiv=self.fc8activ,
                               layertes=self.fc8tes,
                               CE=0.,
                               eye=self.fc8eye,
                               gradinput=None,
                               layerright=self.softmax,
                               layerrightactiv=self.softmaxactiv,
                               layerright_output=self.softmaxoutput.clone().detach(),
                               configs=self.configs)

        if self.training:
            self.logrow['fc2gradmin'] = grad.min().item()
            self.logrow['fc2gradmax'] = grad.max().item()
            self.logrow['fc2gradmean'] = grad.mean().item()
            self.logrow['fc2gradstd'] = grad.std().item()

        # if res != None:
        #     return res
        # return grad

    def close(self):
        self.hook0.remove()
        self.hook.remove()







def cifar10cnn(configs, progress=True, **kwargs):
    model = Cifar10CNN(configs=configs, **kwargs)
    if configs['pretrained']:
        if configs['pretrained_url'] == None:
            state_dict = load_state_dict_from_url(model_urls['alexnet'], map_location=configs['gpu'], progress=progress)
            model.load_state_dict(state_dict)
        else:
            state_dict = load_state_dict_from_path(model_path=configs['pretrained_url'], map_location=configs['gpu'],
                                                   progress=progress)
            model.load_state_dict(state_dict['state_dict'])
    elif configs['partial_freeze'] and configs['pretrained_url'] is not None:
        # state_dict = load_state_dict_from_path(configs['pretrained_url'], map_location=torch.device('cpu'), progress=progress)
        # if not torch.cuda.is_available():
        state_dict = load_state_dict_from_path(configs['pretrained_url'], map_location='cpu', progress=progress)
        # else:
        #     state_dict = load_state_dict_from_path(configs['pretrained_url'], progress=progress)

        tempDict = OrderedDict()
        if list(state_dict['state_dict'].keys())[0].startswith('module.'):
            for k in state_dict['state_dict'].keys():
                tempDict[k[7:]] = state_dict['state_dict'][k]
            model.load_state_dict(tempDict)
        else:
            model.load_state_dict(state_dict['state_dict'])

    return model


import os
import re
import torch
import zipfile

ENV_TORCH_HOME = 'TORCH_HOME'
ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
DEFAULT_CACHE_DIR = '~/.cache'

HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')


def _get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(ENV_TORCH_HOME,
                  os.path.join(os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch')))
    return torch_home


def load_state_dict_from_path(model_path, model_dir=None, map_location=None, progress=True, check_hash=False):
    # Note: extractall() defaults to overwrite file if exists. No need to clean up beforehand.
    #       We deliberately don't handle tarfile here since our legacy serialization format was in tar.
    #       E.g. resnet18-5c106cde.pth which is widely used.
    if zipfile.is_zipfile(model_path):
        with zipfile.ZipFile(model_path) as cached_zipfile:
            members = cached_zipfile.infolist()
            if len(members) != 1:
                raise RuntimeError('Only one file(not dir) is allowed in the zipfile')
            cached_zipfile.extractall(model_dir)
            extraced_name = members[0].filename
            model_path = os.path.join(model_dir, extraced_name)

    return torch.load(model_path, map_location=map_location)

# shamelessly copied from here https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/7
# original docs at https://pytorch.org/docs/master/nn.html#conv2d
#
# # def hooked_backward(self, module):
# #     with hook_output(module) as hook_a:
# #         preds = module(torch.random(1,3,224,224))
# #         preds[0, 0].backward()
# #     return hook_a
#
# # def hook_fn(m, grad_input, grad_output):
# #     # visualisation[m] = o
# #     return grad_output * 0.01
# #     #pass
#
# # def get_all_layers(self, net):
# #     handles = (None, None)
# #     for name, layer in net._modules.items():
# #         # If it is a sequential, don't register a hook on it
# #         # but recursively register hook on all it's module children
# #         if isinstance(layer, nn.Sequential):
# #             net.get_all_layers(layer)
# #         else:
# #             # it's a non sequential. Register a hook
# #             layer.register_forward_hook(self.hook_fn)
# #     return handles
#
# def memory_strided_im2col(self, x, kernel, step=1):
#     output_shape = (x.shape[0] - kernel) + 1
#     return view_as_windows(x.numpy(), kernel).reshape(output_shape * output_shape, kernel * 2)
#     # output_shape = (x.shape[0] - kernel.shape[0]) + 1
#     # return view_as_windows(x, kernel.shape, step).reshape(output_shape * output_shape, kernel.shape[0] * 2)
#     # view_as_windows has an additional step parameter that can be used with different strides
#
#
# # input_matrix = np.array([[3,9,0], [2, 8, 1], [1,4,8]])
# # kernel = np.array([[8,9], [4,4]])
# # output_shape = (input_matrix.shape[0] - kernel.shape[0]) + 1
# # mem_strided_mat = memory_strided_im2col(input_matrix, kernel)
# # mem_strided_conv = np.dot(kernel.flatten(), mem_strided_mat) + biasmem_strided_conv = mem_strided_conv.reshape(output_shape, output_shape)
# # PS: x_newview = np.lib.stride_tricks.as_strided(x, shape = (5, 4), strides = (8,8))
#
# def conv_2d(x, kernel):
#     # Assuming Padding = 0, stride = 1
#     output_shape = x.shape[0] - kernel + 1
#     result = np.zeros((output_shape, output_shape))
#
#     for row in range(x.shape[0] - 1):
#         for col in range(x.shape[1] - 1):
#             window = x[row: row + kernel, col: col + kernel]
#             result[row, col] = np.sum(np.multiply(kernel, window))
#     return result
#
#
# def calculateWindows(self, x):
#     windows = F.unfold(x, kernel_size=11, padding=2, dilation=1, stride=4)
#
#     windows = windows.transpose(1, 2).contiguous().view(-1, x.shape[1], 11 * 11)
#     windows = windows.transpose(0, 1)
#
#     return windows
#
#
# def add_pairs(self, module, keypair, inp, calc_te=False, left_layer=True):
#     if self.conv_te:
#         if isinstance(module.padding, tuple):
#             xpad = F.pad(inp.detach(),
#                          pad=([module.padding[0], module.padding[0], module.padding[1], module.padding[1]]),
#                          mode='constant', value=0)
#         elif module.padding != 0:
#             xpad = F.pad(inp.detach(),
#                          pad=([module.padding, module.padding, module.padding, module.padding]),
#                          mode='constant', value=0)
#         else:
#             xpad = inp.detach()
#
#         kernel_h_index = 0
#         if (isinstance(module.kernel_size, tuple)):
#             kernel_size = module.kernel_size[0]
#         else:
#             kernel_size = module.kernel_size
#         kernel_w_index = 0
#         if not keypair in self.averages:
#             self.averages[keypair] = dict()
#
#         for b in range(0, xpad.shape[0]):
#             if not b in self.averages[keypair]:
#                 self.averages[keypair][b] = dict()
#             for ic in range(0, xpad[b].shape[0]):
#                 if not ic in self.averages[keypair][b]:
#                     self.averages[keypair][b][ic] = TEDiscrete()
#                 row_windows = 0
#                 col_windows = 0
#                 while (kernel_h_index + kernel_size) <= xpad[b, ic].shape[0]:
#                     while (kernel_w_index + kernel_size) <= xpad[b, ic].shape[0]:
#                         # print('filter: ' + str(kernel_h_index) + ':' + str(kernel_h_index + kernel_size) + ', ' + str(kernel_w_index) + ':' + str(kernel_w_index + kernel_size))
#                         window = xpad[b, ic][kernel_h_index: kernel_h_index + kernel_size,
#                                  kernel_w_index: kernel_w_index + kernel_size]
#                         # can explore of having the same TE array built from multiple adjacent layers at once
#                         # having the average of a single layerused as a single entry
#                         med = window.median()
#                         if med <= self.g1:
#                             if calc_te == True:
#                                 self.averages[keypair][b][ic].add_y(0.)
#                             else:
#                                 self.averages[keypair][b][ic].add_x(0.)
#                         else:
#                             if calc_te == True:
#                                 self.averages[keypair][b][ic].add_y(1.)
#                             else:
#                                 self.averages[keypair][b][ic].add_x(1.)
#                         if (isinstance(module.stride, tuple)):
#                             kernel_w_index += module.stride[0]
#                         else:
#                             kernel_w_index += module.stride
#                     kernel_w_index = 0
#                     if (isinstance(module.stride, tuple)):
#                         kernel_h_index += module.stride[0]
#                     else:
#                         kernel_h_index += module.stride
#
#                 kernel_h_index = 0
#                 kernel_w_index = 0
#
#         # triger the TE calculus for the last pair of layers for all xs and ys gathered in TE
#         if calc_te == True and self._forward_passes >= self.skip_first:
#             for b in range(0, xpad.shape[0]):
#                 for ic in range(0, xpad[0].shape[0]):
#                     assert len(self.averages[keypair][b][ic].xs) == len(
#                         self.averages[keypair][b][ic].ys), "TE input series lengths are different"
#                     self.averages[keypair][b][ic].pair_xy()
#                     self.averages[keypair][b][ic].calc_te()
#                     # print(self.averages[keypair][b][ic].sum)
#
#
# # def add_fc_pairs(self, keypair, inp, inp2, interim, calc_te = False, left_layer = True):
# #     if (self._forward_passes % self.batch_size) == 0:
# #         idx = math.floor(self._forward_passes / self.batch_size)
# #         xb = inp[self.batch_size - 1].detach().to(torch.uint8)
# #         xb[xb >= self.g1] = 1
# #         xb[xb > self.g1] = 0
# #         interim[:, idx] = xb
# #         del xb
# #
# #     #torch.cartesian_prod(aa, bb)
# #
# #     if self.conv_te == False and calc_te:
# #         cp = torch.cartesian_prod(inp[:, -1], inp2[:, -1])
# #         #fcpair = self.product(inp[:, -1], inp2[:,-1])
# #
# #         #we look for non conv layer logic
# #         #xpad = inp.detach()
# #
# #         if not keypair in self.averages:
# #             self.averages[keypair] = dict()
# #
# #         for b in range(0, xpad.shape[0]):
# #             if not b in self.averages[keypair]:
# #                 self.averages[keypair][b] = dict()
# #             for ic in range(0, xpad[b].shape[0]):
# #                 if not ic in self.averages[keypair][b]:
# #                     self.averages[keypair][b][ic] = TEDiscrete()
# #                     self.averages[keypair][b][ic].initialise()
# #
# #                 if xpad[b][ic] <= self.g1:
# #                     if calc_te == True:
# #                         self.averages[keypair][b][ic].add_dest(0)
# #                     else:
# #                         self.averages[keypair][b][ic].add_source(0)
# #                 else:
# #                     if calc_te == True:
# #                         self.averages[keypair][b][ic].add_dest(1)
# #                     else:
# #                         self.averages[keypair][b][ic].add_source(1)
# #
# #         #triger the TE calculus for the last pair of layers for all xs and ys gathered in TE
# #         if calc_te == True and self._forward_passes >= self.skip_first:
# #             for b in range(0, xpad.shape[0]):
# #                 for ic in range(0, xpad[0].shape[0]):
# #                     assert len(self.averages[keypair][b][ic].xs) == len(self.averages[keypair][b][ic].ys), "TE input series lengths are different"
# #                     sum = self.averages[keypair][b][ic].calcLocalTE()
# #                     print(sum)
#
# # def product(self, *args, repeat=1):
# #     # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
# #     # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
# #     pools = [tuple(pool) for pool in args] * repeat
# #     result = [[]]
# #     for pool in pools:
# #         result = [x + [y] for x in result for y in pool]
# #     return result
#
#
#
# def num2tuple(num):
#     return num if isinstance(num, tuple) else (num, num)
#
#
# def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
#     h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
#                                               num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(
#         dilation)
#     pad = num2tuple(pad[0]), num2tuple(pad[1])
#
#     h = math.floor((h_w[0] + sum(pad[0]) - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
#     w = math.floor((h_w[1] + sum(pad[1]) - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
#
#     return h, w
#
#
# def convtransp2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, out_pad=0):
#     h_w, kernel_size, stride, pad, dilation, out_pad = num2tuple(h_w), \
#                                                        num2tuple(kernel_size), num2tuple(stride), num2tuple(
#         pad), num2tuple(dilation), num2tuple(out_pad)
#     pad = num2tuple(pad[0]), num2tuple(pad[1])
#
#     h = (h_w[0] - 1) * stride[0] - sum(pad[0]) + dilation[0] * (kernel_size[0] - 1) + out_pad[0] + 1
#     w = (h_w[1] - 1) * stride[1] - sum(pad[1]) + dilation[1] * (kernel_size[1] - 1) + out_pad[1] + 1
#
#     return h, w
#
#
# def conv2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1):
#     h_w_in, h_w_out, kernel_size, stride, dilation = num2tuple(h_w_in), num2tuple(h_w_out), \
#                                                      num2tuple(kernel_size), num2tuple(stride), num2tuple(dilation)
#
#     p_h = ((h_w_out[0] - 1) * stride[0] - h_w_in[0] + dilation[0] * (kernel_size[0] - 1) + 1)
#     p_w = ((h_w_out[1] - 1) * stride[1] - h_w_in[1] + dilation[1] * (kernel_size[1] - 1) + 1)
#
#     return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))
#
#
# def convtransp2d_get_padding(h_w_in, h_w_out, kernel_size=1, stride=1, dilation=1, out_pad=0):
#     h_w_in, h_w_out, kernel_size, stride, dilation, out_pad = num2tuple(h_w_in), num2tuple(h_w_out), \
#                                                               num2tuple(kernel_size), num2tuple(stride), num2tuple(
#         dilation), num2tuple(out_pad)
#
#     p_h = -(h_w_out[0] - 1 - out_pad[0] - dilation[0] * (kernel_size[0] - 1) - (h_w_in[0] - 1) * stride[0]) / 2
#     p_w = -(h_w_out[1] - 1 - out_pad[1] - dilation[1] * (kernel_size[1] - 1) - (h_w_in[1] - 1) * stride[1]) / 2
#
#     return (math.floor(p_h / 2), math.ceil(p_h / 2)), (math.floor(p_w / 2), math.ceil(p_w / 2))
# # self.features = nn.Sequential(OrderedDict( {
#         #     'Conv1' : nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),#padding 5
#         #     'Relu1' : nn.ReLU(inplace=True),
#         #     'MaxPool1' : nn.MaxPool2d(kernel_size=3, stride=2),#kernel_size=2.
#         #     'Conv2' : nn.Conv2d(64, 192, kernel_size=5, padding=2),
#         #     'Relu2' : nn.ReLU(inplace=True),
#         #     'MaxPool2' : nn.MaxPool2d(kernel_size=3, stride=2),#kernel_size=2
#         #     'Conv3' : nn.Conv2d(192, 384, kernel_size=3, padding=1),
#         #     'Relu3' : nn.ReLU(inplace=True),
#         #     'Conv4' : nn.Conv2d(384, 256, kernel_size=3, padding=1),
#         #     'Relu4' : nn.ReLU(inplace=True),
#         #     'Conv5' : nn.Conv2d(256, 256, kernel_size=3, padding=1),
#         #     'Relu5' : nn.ReLU(inplace=True),
#         #     'MaxPool5' : nn.MaxPool2d(kernel_size=3, stride=2),#kernel_size=2
#         # }))

