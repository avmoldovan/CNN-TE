import numpy as np
import pandas as pd
import random
import math
import torch
import datetime as dt
from PyTE.TEWindow import TEWindow
from PyTIE.TIEWindow import TIEWindow

DEBUG = False

def prepare_activations(configs, activations, outputs, windowed = False):
    te_length = int(configs['te_length'])
    batch_size = int(configs['batch_size'])
    if configs['gpu'] != 'cpu':
        if windowed:
            # activations = np.transpose(np.concatenate((np.transpose(activations), outputs)))#torch.cat((activations.t().cuda(), outputs.to(dtype=torch.uint8).cuda()), dim=0).t()
            activations = np.concatenate((activations, outputs))
        else:
            activations = outputs  # outputs.cuda().t()
    else:
        if windowed:
            # activations = torch.cat((activations.t(), outputs.to(dtype=torch.uint8)), dim=0).t()
            activations = np.concatenate((activations, outputs))
        else:
            activations = outputs  # .t()

    if activations.shape[
        1] > te_length and te_length > 0:  # and (self.configs['batch_size'] + self.configs['te_length']) < activations.shape[1]:
        ## activations = activations.narrow(1, self.configs['batch_size'], self.configs['te_length'])
        ## if activations.shape[1] + self.batch_size - 1 <= self.te_length:

        #############activations = activations.narrow(1, self.batch_size - 1, min(activations.shape[1] - self.batch_size, self.te_length))
        activations = activations[:, -min(activations.shape[1] - batch_size, activations.shape[1]):]

        ## else:
        ##     activations = activations.narrow(1, self.batch_size - 1, (self.te_length if activations.shape[1] > self.te_length else activations.shape[1]))

        ## self.averages['fcpair2'] = dict()
        ## self.averages['fcpair3'] = dict()
    return activations

def update_with_te(module, keypair, layerleft, layerleftactiv, layertes, CE, eye, gradinput, layerright=None,
                   layerrightactiv=None, layerright_output=None, configs=None):
    calc_fc(module.averages, keypair, np.transpose(layerleftactiv), np.transpose(layerrightactiv), useTIE = configs['useTIE'])
    return add_te_to(
        averages=module.averages,
        keypair=keypair,
        layertes=layertes,
        layerleft=layerleft,
        eye=eye,
        ce=module.criterion(module.last_x, module.last_target),
        layerright_output=layerright_output,
        gradinput=gradinput,
        gpu=module.gpu,
    )

def calc_fc(averages, keypair, layerleft, layerright, clean_window=False, useTIE=False):
    one_idx = 0
    two_idx = 0

    for one in layerleft:
        if not one_idx in averages[keypair]:
            averages[keypair][one_idx] = dict()

        for two in layerright:
            if not two_idx in averages[keypair][one_idx]:
                if useTIE:
                    averages[keypair][one_idx][two_idx] = TIEWindow(clean_window=clean_window)
                else:
                    averages[keypair][one_idx][two_idx] = TEWindow(clean_window=clean_window)
                averages[keypair][one_idx][two_idx].initialise()

            # if self.windowed:
            #     #if self.averages[keypair][one_idx][two_idx] is not TEWindow:
            #     self.averages[keypair][one_idx][two_idx] = TEWindow(windowed=self.windowed)
            #     self.averages[keypair][one_idx][two_idx].initialise()
            # else:
            #     #if self.windowed:
            #     # if not two_idx in self.averages[keypair][one_idx]:
            #     #     self.averages[keypair][one_idx][two_idx] = TEWindow()
            #     # else:
            #     #     self.averages[keypair][one_idx][two_idx] = TEDiscrete()
            #     self.averages[keypair][one_idx][two_idx].initialise()

            averages[keypair][one_idx][two_idx].add_source(one)
            averages[keypair][one_idx][two_idx].add_dest(two)
            two_idx += 1
        one_idx += 1
        two_idx = 0


def add_te_to(averages, keypair, layertes, layerleft, eye, ce, layerright_output, gradinput, gpu=False):
    for i in averages[keypair].keys():
        for j in averages[keypair][i].keys():
            te = averages[keypair][i][j].calcLocalTE()
            layertes[i][j] = te
    #                 if (sum > 0.001):
    #                     layertes[i][j] = sum
    #                 else:
    #                     layertes[i][j] = 0.
    # layertes[i][j] = (1. - ce / layerright_output.log().sum()) * sum

    # mean = layerleft.weight.grad.mean()#dim=0, keepdim=True)
    # std = layerleft.weight.grad.std()#dim=0, keepdim=True, unbiased=False)
    # layertes -= mean
    # layertes /= std + 1e-7

    if gpu:
        #layerleft.weight.grad.data += (layerleft.weight.grad.t().matmul(layertes.t().cuda())).t()

        # layerleft.weight.data = (layerleft.weight.t().matmul((eye - layertes.tanh()).cuda())).t()
        # layerleft.weight.data = (layerleft.weight.t().matmul((ce * layertes.tanh()).cuda())).t()
        # layerleft.weight.grad.data = (layerleft.weight.grad.t().matmul((eye - layertes).cuda())).t()
        #layerleft.weight.data += ((layertes).cuda()).t()
        #layerleft.weight.data = (layerleft.weight.t().matmul((eye - layertes.t()).cuda())).t()

        #layerleft.weight.data += (layerleft.weight.data.t().matmul(layertes.cuda())).t()
        layerleft.weight.data = (layerleft.weight.t().matmul((eye - layertes).cuda())).t()
        ##layerleft.weight.data.add_(layertes.cuda() * 10.)
    else:
        #layerleft.weight.grad.data += (layerleft.weight.grad.t().matmul(layertes.t())).t()

        # layerleft.weight.data = (layerleft.weight.t().matmul(eye - layertes.tanh())).t()
        # layerleft.weight.data = (layerleft.weight.t().matmul(ce * layertes.tanh())).t()
        # layerleft.weight.grad.data = (layerleft.weight.grad.t().matmul(eye - layertes)).t()
        #layerleft.weight.data += (layertes).t()
        #layerleft.weight.data = (layerleft.weight.t().matmul(eye - layertes.t())).t()

        #layerleft.weight.data += (layerleft.weight.data.t().matmul(layertes)).t()
        layerleft.weight.data = (layerleft.weight.t().matmul(eye - layertes)).t()
        #layerleft.weight.data.add_(layertes * 10.)

def updateLogGrad(module, grad):
    if module.training:
        module.logrow['fc1gradmin'] = grad.min().item()
        module.logrow['fc1gradmax'] = grad.max().item()
        module.logrow['fc1gradmean'] = grad.mean().item()
        module.logrow['fc1gradstd'] = grad.std().item()

        end = dt.datetime.now()
        tdelta = end - module.start
        module.logrow['time'] = tdelta
        module.evol = module.evol.append(module.logrow, ignore_index=True, sort=False)
        module.evol.to_csv(module.filekey + '-stds.csv', index=False)

def updateLogRow(module, ce):
    module.logrow['epoch'] = module.current_epoch
    # self.logrow['fc1']= self.fc7output #fc7med.item() if fc7med is not None else 0,
    if hasattr(module, 'fc7output'):
        module.logrow['fc1min'] = module.fc7output.min().item()
        module.logrow['fc1max'] = module.fc7output.max().item()
        module.logrow['fc1mean'] = module.fc7output.mean().item()
        module.logrow['fc1std'] = module.fc7output.std().item()
    # logrow['# 'fc1grad.min': self.fc7output.weight.grad.min(),
    # logrow['# 'fc1grad.max': self.fc7output.weight.grad.max(),
    # logrow['# 'fc1grad.mean': self.fc7output.weight.grad.mean(),
    # logrow['# 'fc1grad.std': self.fc7output.weight.grad.std(),
    # logrow['fc2']= fc8med.item() if fc8med is not None else 0
    module.logrow['fc2min'] = module.fc8output.min().item()
    module.logrow['fc2max'] = module.fc8output.max().item()
    module.logrow['fc2mean'] = module.fc8output.mean().item()
    module.logrow['fc2std'] = module.fc8output.std().item()
    # 'fc2grad.min': self.fc8output.weight.grad.min(),
    # 'fc2grad.max': self.fc8output.weight.grad.max(),
    # 'fc2grad.mean': self.fc8output.weight.grad.mean(),
    # 'fc2grad.std': self.fc8output.weight.grad.std(),
    module.logrow['softmax'] = module.softmaxoutput.sum().item()
    module.logrow['softmaxmin'] = module.softmaxoutput.min().item()
    module.logrow['softmaxmax'] = module.softmaxoutput.max().item()
    module.logrow['softmaxmean'] = module.softmaxoutput.mean().item()
    module.logrow['softmaxstd'] = module.softmaxoutput.std().item()
    module.logrow['g1'] = module.g1
    if hasattr(module, 'fc7tes'):
        module.logrow['tes1min'] = module.fc7tes.min().item() if module.fc7tes is not None else 0.
        module.logrow['tes1max'] = module.fc7tes.max().item() if module.fc7tes is not None else 0.
        module.logrow['tes1mean'] = module.fc7tes.mean().item() if module.fc7tes is not None else 0.
        module.logrow['tes1std'] = module.fc7tes.std().item() if module.fc7tes is not None else 0.
    module.logrow['g2'] = module.g2
    module.logrow['tes2min'] = module.fc8tes.min().item() if module.fc8tes is not None else 0.
    module.logrow['tes2max'] = module.fc8tes.max().item() if module.fc8tes is not None else 0.
    module.logrow['tes2mean'] = module.fc8tes.mean().item() if module.fc8tes is not None else 0.
    module.logrow['tes2std'] = module.fc8tes.std().item() if module.fc8tes is not None else 0.
    module.logrow['loss'] = ce.item()
    module.logrow['te_length'] = module.configs['te_length']
#
# def update_with_te(keypair, layerleft, layerleftactiv, layertes, CE, eye, gradinput, layerright=None,
#                    clean_window=False, layerrightactiv=None, layerright_output=None):
#     calc_fc(keypair, layerleftactiv, layerrightactiv, clean_window=clean_window)
#     add_te_to(keypair, layertes, layerleft, eye, averages, CE, smxout, gradinput,  layerright_output, gradinput)
#
# def calc_fc(keypair, layerleft, layerright, averages, clean_window=False):
#     one_idx = 0
#     two_idx = 0
#
#     for one in layerleft:
#         if not one_idx in averages[keypair]:
#             averages[keypair][one_idx] = dict()
#
#         for two in layerright:
#             if not two_idx in averages[keypair][one_idx]:
#                 averages[keypair][one_idx][two_idx] = TEWindow(clean_window=clean_window)
#                 averages[keypair][one_idx][two_idx].initialise()
#
#             # if self.windowed:
#             #     #if self.averages[keypair][one_idx][two_idx] is not TEWindow:
#             #     self.averages[keypair][one_idx][two_idx] = TEWindow(windowed=self.windowed)
#             #     self.averages[keypair][one_idx][two_idx].initialise()
#             # else:
#             #     #if self.windowed:
#             #     # if not two_idx in self.averages[keypair][one_idx]:
#             #     #     self.averages[keypair][one_idx][two_idx] = TEWindow()
#             #     # else:
#             #     #     self.averages[keypair][one_idx][two_idx] = TEDiscrete()
#             #     self.averages[keypair][one_idx][two_idx].initialise()
#
#             averages[keypair][one_idx][two_idx].add_source(one)
#             averages[keypair][one_idx][two_idx].add_dest(two)
#             two_idx += 1
#         one_idx += 1
#         two_idx = 0
#     return averages
#
#
# def add_te_to(keypair, layertes, layerleft, eye, averages, ce, smxout, gradinput, gpu=False):
#     for i in averages[keypair].keys():
#         for j in averages[keypair][i].keys():
#             sum = averages[keypair][i][j].calcLocalTE()
#             layertes[i][j] = sum
#     #                 if (sum > 0.001):
#     #                     layertes[i][j] = sum
#     #                 else:
#     #                     layertes[i][j] = 0.
#     # layertes[i][j] = (1. - ce / smxout.log().sum()) * sum
#
#     # mean = layerleft.weight.grad.mean()#dim=0, keepdim=True)
#     # std = layerleft.weight.grad.std()#dim=0, keepdim=True, unbiased=False)
#     # layertes -= mean
#     # layertes /= std + 1e-7
#
#     if gpu:
#         # layerleft.weight.grad.data += (layerleft.weight.grad.t().matmul(layertes.t().cuda())).t()
#
#         # layerleft.weight.data = (layerleft.weight.t().matmul((eye - layertes.tanh()).cuda())).t()
#         # layerleft.weight.data = (layerleft.weight.t().matmul((ce * layertes.tanh()).cuda())).t()
#         # layerleft.weight.grad.data = (layerleft.weight.grad.t().matmul((eye - layertes).cuda())).t()
#         # layerleft.weight.data += ((layertes).cuda()).t()
#         # layerleft.weight.data = (layerleft.weight.t().matmul((eye - layertes.t()).cuda())).t()
#
#         # layerleft.weight.data += (layerleft.weight.data.t().matmul(layertes.cuda())).t()
#         layerleft.weight.data = (layerleft.weight.t().matmul((eye - layertes).cuda())).t()
#         ##layerleft.weight.data.add_(layertes.cuda() * 10.)
#     else:
#         # layerleft.weight.grad.data += (layerleft.weight.grad.t().matmul(layertes.t())).t()
#
#         # layerleft.weight.data = (layerleft.weight.t().matmul(eye - layertes.tanh())).t()
#         # layerleft.weight.data = (layerleft.weight.t().matmul(ce * layertes.tanh())).t()
#         # layerleft.weight.grad.data = (layerleft.weight.grad.t().matmul(eye - layertes)).t()
#         # layerleft.weight.data += (layertes).t()
#         # layerleft.weight.data = (layerleft.weight.t().matmul(eye - layertes.t())).t()
#
#         # layerleft.weight.data += (layerleft.weight.data.t().matmul(layertes)).t()
#         layerleft.weight.data = (layerleft.weight.t().matmul(eye - layertes)).t()

def tuple2float(self, val):
    if type(val) == tuple:
        return val[0]
    return val

def getRandomIndices(self, count, inmatrix):
    ls = np.arange(0, inmatrix.shape[0] * inmatrix.shape[1], 1, dtype=int)
    sel = random.sample(list(ls), k=math.floor(inmatrix.shape[0] * inmatrix.shape[1] * count / 100))

    for i in sel:
        row = math.floor(i / (inmatrix.shape[1]))
        column = i % inmatrix.shape[1]
        inmatrix[row, column] = 1

    # rs = np.random.randint(0, size=inmatrix.shape[0])
    # cs = np.random.randint(inmatrix.shape[1], size=inmatrix.shape[0] * inmatrix.shape[1])
    #
    # ls = np.arange(0, inmatrix.shape[0] * inmatrix.shape[1], 1, dtype=int)
    # sel = random.sample(list(ls), k=math.floor(inmatrix.shape[0]*inmatrix.shape[1] * count / 100))
    #
    # np.ravel(inmatrix)[sel] = 1
    #
    # i = 0
    # for r in inmatrix:
    #     inmatrix[i] = random.sample(count * 100 / inmatrix.shape[0])
    #     i += 1

    # for i in sel:
    #     row = math.ceil(i / inmatrix.shape[0])
    #     column = i % (inmatrix.shape[1])
    #     inmatrix[row, column] = 1
    #
    #     #if len(str(i)) > 1:
    #     #inmatrix[math.floor(i / matrix.shape[0]), i % matrix.shape[0]] = 1
    #         #inmatrix[math.floor(i / 10), i % 10] = 1
    #     # else:
    #     #     inmatrix[0, i] = 1
    return inmatrix.to(dtype=torch.uint8)

    # ls.reshape(matrix.shape[0])

def getLogRowJson(g1, g2, te_length):
    return {'epoch': 0,
                       # 'fc1': 0.,
                       'fc1min': 0.,
                       'fc1max': 0.,
                       'fc1mean': 0.,
                       'fc1std': 0.,
                       'fc1gradmin': 0.,
                       'fc1gradmax': 0.,
                       'fc1gradmean': 0.,
                       'fc1gradstd': 0.,
                       # 'fc2': 0.,
                       'fc2min': 0.,
                       'fc2max': 0.,
                       'fc2mean': 0.,
                       'fc2std': 0.,
                       'fc2gradmin': 0.,
                       'fc2gradmax': 0.,
                       'fc2gradmean': 0.,
                       'fc2gradstd': 0.,
                       'softmax': 0.,
                       'softmaxmin': 0.,
                       'softmaxmax': 0.,
                       'softmaxmean': 0.,
                       'softmaxstd': 0.,
                       'g1': g1,
                       'tes1min': 0.,
                       'tes1max': 0.,
                       'tes1mean': 0.,
                       'tes1std': 0.,
                       'g2': g2,
                       'tes2min': 0.,
                       'tes2max': 0.,
                       'tes2mean': 0.,
                       'tes2std': 0.,
                       'loss': 0.,
                       'te_length': te_length,
                       'time': 0
                       }

def getEvolutionDataFrame():
    return pd.DataFrame(columns=[
            'epoch',
            # 'fc1',
            'fc1min',
            'fc1max',
            'fc1mean',
            'fc1std',
            'fc1gradmin',
            'fc1gradmax',
            'fc1gradmean',
            'fc1gradstd',
            # 'fc2',
            'fc2min',
            'fc2max',
            'fc2mean',
            'fc2std',
            'fc2gradmin',
            'fc2gradmax',
            'fc2gradmean',
            'fc2gradstd',
            'softmax',
            'softmaxmin',
            'softmaxmax',
            'softmaxmean',
            'softmaxstd',
            'g1',
            'tes1min',
            'tes1max',
            'tes1mean',
            'tes1std',
            'g2',
            'tes2min',
            'tes2max',
            'tes2mean',
            'tes2std',
            'loss',
            'te_length',
            'time'
        ])