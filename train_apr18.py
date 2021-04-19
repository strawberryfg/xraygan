import math
import numpy as np
from os import path as osp
import cv2
import random
import matplotlib.pyplot as plt
import pdb

#0. torch imports
import torch
from torch.utils.data import DataLoader,Dataset
from torch import optim,nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
import torch.nn.functional as F
import torchvision.utils as vutils
from torchvision import models

from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models.resnet import model_urls


## Data parallel
"""Encoding Data Parallel"""
import threading
import functools
from torch.autograd import Variable, Function
import torch.cuda.comm as comm
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import get_a_var
from torch.nn.parallel._functions import ReduceAddCoalesced, Broadcast

# [DATA PARALLEL]

__all__ = ['allreduce', 'DataParallelModel', 'DataParallelCriterion',
           'patch_replication_callback']

def allreduce(*inputs):
    """Cross GPU all reduce autograd operation for calculate mean and
    variance in SyncBN.
    """
    return AllReduce.apply(*inputs)


class AllReduce(Function):
    @staticmethod
    def forward(ctx, num_inputs, *inputs):
        ctx.num_inputs = num_inputs
        ctx.target_gpus = [inputs[i].get_device() for i in range(0, len(inputs), num_inputs)]
        inputs = [inputs[i:i + num_inputs]
                 for i in range(0, len(inputs), num_inputs)]
        # sort before reduce sum
        inputs = sorted(inputs, key=lambda i: i[0].get_device())
        results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
        outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
        return tuple([t for tensors in outputs for t in tensors])

    @staticmethod
    def backward(ctx, *inputs):
        inputs = [i.data for i in inputs]
        inputs = [inputs[i:i + ctx.num_inputs]
                 for i in range(0, len(inputs), ctx.num_inputs)]
        results = comm.reduce_add_coalesced(inputs, ctx.target_gpus[0])
        outputs = comm.broadcast_coalesced(results, ctx.target_gpus)
        return (None,) + tuple([Variable(t) for tensors in outputs for t in tensors])


class Reduce(Function):
    @staticmethod
    def forward(ctx, *inputs):
        ctx.target_gpus = [inputs[i].get_device() for i in range(len(inputs))]
        inputs = sorted(inputs, key=lambda i: i.get_device())
        return comm.reduce_add(inputs)

    @staticmethod
    def backward(ctx, gradOutput):
        return Broadcast.apply(ctx.target_gpus, gradOutput)


class DataParallelModel(DataParallel):
    """Implements data parallelism at the module level.

    This container parallelizes the application of the given module by
    splitting the input across the specified devices by chunking in the
    batch dimension.
    In the forward pass, the module is replicated on each device,
    and each replica handles a portion of the input. During the backwards pass, gradients from each replica are summed into the original module.
    Note that the outputs are not gathered, please use compatible
    :class:`encoding.parallel.DataParallelCriterion`.

    The batch size should be larger than the number of GPUs used. It should
    also be an integer multiple of the number of GPUs so that each chunk is
    the same size (so that each GPU processes the same number of samples).

    Args:
        module: module to be parallelized
        device_ids: CUDA devices (default: all devices)

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> y = net(x)
    """
    def gather(self, outputs, output_device):
        return outputs

    def replicate(self, module, device_ids):
        modules = super(DataParallelModel, self).replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules



class DataParallelCriterion(DataParallel):
    """
    Calculate loss in multiple-GPUs, which balance the memory usage for
    Semantic Segmentation.

    The targets are splitted across the specified devices by chunking in
    the batch dimension. Please use together with :class:`encoding.parallel.DataParallelModel`.

    Reference:
        Hang Zhang, Kristin Dana, Jianping Shi, Zhongyue Zhang, Xiaogang Wang, Ambrish Tyagi,
        Amit Agrawal. Context Encoding for Semantic Segmentation.
        *The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) 2018*

    Example::

        >>> net = encoding.nn.DataParallelModel(model, device_ids=[0, 1, 2])
        >>> criterion = encoding.nn.DataParallelCriterion(criterion, device_ids=[0, 1, 2])
        >>> y = net(x)
        >>> loss = criterion(y, target)
    """
    def forward(self, inputs, *targets, **kwargs):
        # input should be already scatterd
        # scattering the targets instead
        # if not self.device_ids:
            # return self.module(inputs, *targets, **kwargs)
        targets, kwargs = self.scatter(targets, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return self.module(inputs, *targets[0])
        replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
        outputs = _criterion_parallel_apply(replicas, inputs, targets, kwargs)
        return Reduce.apply(*outputs) / len(outputs)

        #return self.gather(outputs, self.output_device).mean()


def _criterion_parallel_apply(modules, inputs, targets, kwargs_tup=None, devices=None):
    assert len(modules) == len(inputs)
    assert len(targets) == len(inputs)
    if kwargs_tup:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)

    lock = threading.Lock()
    results = {}
    if torch_ver != "0.3":
        grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, target, kwargs, device=None):
        if torch_ver != "0.3":
            torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                output = module(input, *target)
            with lock:
                results[i] = output
        except Exception as e:
            with lock:
                results[i] = e

    if len(modules) > 1:
        threads = [threading.Thread(target=_worker,
                                    args=(i, module, input, target,
                                          kwargs, device),)
                   for i, (module, input, target, kwargs, device) in
                   enumerate(zip(modules, inputs, targets, kwargs_tup, devices))]

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0])

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, Exception):
            raise output
        outputs.append(output)
    return outputs


###########################################################################
# Adapted from Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
#
class CallbackContext(object):
    pass


def execute_replication_callbacks(modules):
    """
    Execute an replication callback `__data_parallel_replicate__` on each module created
    by original replication.

    The callback will be invoked with arguments `__data_parallel_replicate__(ctx, copy_id)`

    Note that, as all modules are isomorphism, we assign each sub-module with a context
    (shared among multiple copies of this module on different devices).
    Through this context, different copies can share some information.

    We guarantee that the callback on the master copy (the first copy) will be called ahead
    of calling the callback of any slave copies.
    """
    master_copy = modules[0]
    nr_modules = len(list(master_copy.modules()))
    ctxs = [CallbackContext() for _ in range(nr_modules)]

    for i, module in enumerate(modules):
        for j, m in enumerate(module.modules()):
            if hasattr(m, '__data_parallel_replicate__'):
                m.__data_parallel_replicate__(ctxs[j], i)


def patch_replication_callback(data_parallel):
    """
    Monkey-patch an existing `DataParallel` object. Add the replication callback.
    Useful when you have customized `DataParallel` implementation.

    Examples:
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
        > patch_replication_callback(sync_bn)
        # this is equivalent to
        > sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
        > sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    """

    assert isinstance(data_parallel, DataParallel)

    old_replicate = data_parallel.replicate

    @functools.wraps(old_replicate)
    def new_replicate(module, device_ids):
        modules = old_replicate(module, device_ids)
        execute_replication_callbacks(modules)
        return modules

    data_parallel.replicate = new_replicate





# 0. ResNet 18
# ResNet Classifier
#class BasicBlock(nn.Module):
#    expansion = 1

#    def __init__(self, in_planes, planes, stride=1):
#        super(BasicBlock, self).__init__()
#        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#        self.bn1 = nn.BatchNorm2d(planes)
#        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#        self.bn2 = nn.BatchNorm2d(planes)

#        self.shortcut = nn.Sequential()
#        if stride != 1 or in_planes != self.expansion*planes:
#            self.shortcut = nn.Sequential(
#                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                nn.BatchNorm2d(self.expansion*planes)
#            )

#    def forward(self, x):
#        out = F.relu(self.bn1(self.conv1(x)))
#        out = self.bn2(self.conv2(out))
#        out += self.shortcut(x)
#        out = F.relu(out)
#        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.embDim = 128 * block.expansion
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128 * block.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 16)
        emb = out.view(out.size(0), -1)
        out = self.linear(emb)
        return out#, emb
    def get_embedding_dim(self):
        return self.embDim

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

#1. DCGAN Generator
class DCGAN_generator(nn.Module):
  """

  Attributes
  ----------
    ngpu : int
      The number of available GPU devices

  """
  def __init__(self, ngpu):
    """Init function

    Parameters
    ----------
      ngpu : int
        The number of available GPU devices

    """
    super(DCGAN_generator, self).__init__()
    self.ngpu = ngpu
    
    nz = 100 # noise dimension
    ngf = 64 # number of features map on the first layer
    nc = 1 # number of channels

    self.main = nn.Sequential(
      # input is Z, going into a convolution
      nn.ConvTranspose2d(     nz, ngf * 4, 4, 1, 0, bias=False),
      nn.BatchNorm2d(ngf * 4),
      nn.ReLU(True),
      # state size. (ngf*8) x 4 x 4
      nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf * 2),
      nn.ReLU(True),
      # state size. (ngf*4) x 8 x 8
      nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ngf),
      nn.ReLU(True),
      # state size. (ngf*2) x 16 x 16
      nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 64 x 64
    )

  def forward(self, input):
    """Forward function

    Parameters
    ----------
    input : :py:class:`torch.Tensor`
    
    Returns
    -------
    :py:class:`torch.Tensor`
      the output of the generator (i.e. an image)

    """
    output = self.main(input)
    return output

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        nz = 100 # noise dimension
        ngf = 64 # number of features map on the first layer
        nc = 1 # number of channels
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 4 x 4
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 8 x 8
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 16 x 16 
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 32 x 32
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 64 x 64
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
    	output = self.main(input)
    	return output

#2. DCGAN Discriminator
class DCGAN_discriminator(nn.Module):
  """ 

  Attributes
  ----------
    ngpu : int
      The number of available GPU devices

  """
  def __init__(self, ngpu):
    """Init function

    Parameters
    ----------
      ngpu : int
        The number of available GPU devices

    """
    super(DCGAN_discriminator, self).__init__()
    self.ngpu = ngpu
        
    ndf = 64
    nc = 1
       
    self.main = nn.Sequential(
      nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf),
      nn.LeakyReLU(0.2, inplace=True),

      nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 8 x 8
      nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(ndf * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 4 x 4
      nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, input):
    """Forward function

    Parameters
    ----------
    input : :py:class:`torch.Tensor`
    
    Returns
    -------
    :py:class:`torch.Tensor`
      the output of the generator (i.e. an image)

    """
    output = self.main(input)

    return output.view(-1, 1).squeeze(1)

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 32 x 32
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 16 x 16 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 8 x 8
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 4 x 4
            nn.Conv2d(ndf * 16, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
    	output = self.main(input)
    	return output.view(-1, 1).squeeze(1)

#3. ResNet




class ResNetBackbone(nn.Module):

    def __init__(self, resnet_type, num_classes = 1000):
    
        resnet_spec = {18: (BasicBlock, [2, 2, 2, 2], [64, 64, 128, 256, 512], 'resnet18'),
               34: (BasicBlock, [3, 4, 6, 3], [64, 64, 128, 256, 512], 'resnet34'),
               50: (Bottleneck, [3, 4, 6, 3], [64, 256, 512, 1024, 2048], 'resnet50'),
               101: (Bottleneck, [3, 4, 23, 3], [64, 256, 512, 1024, 2048], 'resnet101'),
               152: (Bottleneck, [3, 8, 36, 3], [64, 256, 512, 1024, 2048], 'resnet152')}
        block, layers, channels, name = resnet_spec[resnet_type]
        
        self.name = name
        self.inplanes = 64
        self.outplanes = 3
        super(ResNetBackbone, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #128 -> 4
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)        
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)        
        x = self.fc(x)

        return x
        
    def load_my_state_dict(model, state_dict):
 
        own_state = model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            #if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
            #    param = param.data
            own_state[name].copy_(param)

    def init_weights(self):
        org_resnet = torch.utils.model_zoo.load_url(model_urls[self.name])
        # drop orginal resnet fc layer, add 'None' in case of no fc layer, that will raise error
        org_resnet.pop('fc.weight', None)
        org_resnet.pop('fc.bias', None)
        org_resnet.pop('conv1.weight', None)
        org_resnet.pop('conv1.bias', None)
        #self.load_state_dict(org_resnet)
        self.load_my_state_dict(org_resnet)
        print("Initialize resnet from model zoo")

#4. logging
import logging
import os

OK = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
END = '\033[0m'

PINK = '\033[95m'
BLUE = '\033[94m'
GREEN = OK
RED = FAIL
WHITE = END
YELLOW = WARNING
class colorlogger():
    def __init__(self, log_dir, log_name='train_logs.txt'):
        # set log
        self._logger = logging.getLogger(log_name)
        self._logger.setLevel(logging.INFO)
        log_file = os.path.join(log_dir, log_name)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_log = logging.FileHandler(log_file, mode='a')
        file_log.setLevel(logging.INFO)
        #console_log = logging.StreamHandler()
        #console_log.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "{}%(asctime)s{} %(message)s".format(GREEN, END),
            "%m-%d %H:%M:%S")
        file_log.setFormatter(formatter)
        #console_log.setFormatter(formatter)
        self._logger.addHandler(file_log)
        #self._logger.addHandler(console_log)

    def debug(self, msg):
        self._logger.debug(str(msg))

    def info(self, msg):
        self._logger.info(str(msg))

    def warning(self, msg):
        self._logger.warning(WARNING + 'WRN: ' + str(msg) + END)

    def critical(self, msg):
        self._logger.critical(RED + 'CRI: ' + str(msg) + END)

    def error(self, msg):
        self._logger.error(RED + 'ERR: ' + str(msg) + END)


#5. Configurations and arguments
root_dir = "E:/ml/ZhangLabData/CellData/chest_xray/"
batch_size = 32
img_size = 128
display_per_iters = 3 # how many iterations before outputting to the console window
save_gan_per_iters = 10 # save gan images per this iterations
save_gan_img_folder_prefix = root_dir + "train_fake/"
show_train_classifier_acc_per_iters = 10 # how many iterations before showing train acc of classifier
save_per_samples = 6000 # save a checkpoint per forward run of this number of samples
model_ckpt_prefix = 'ecgan'

# define device 
device = torch.device("cuda:0")

# The files that contain paths of all images
train_normal_file_prefix = root_dir + "train/NORMAL/"
train_normal_file_path = train_normal_file_prefix + "normal_files.txt"
train_pneumonia_file_prefix = root_dir + "train/PNEUMONIA/"
train_pneumonia_file_path = train_pneumonia_file_prefix + "pneumonia_files.txt"
test_normal_file_prefix = root_dir + "test/NORMAL/"
test_normal_file_path = test_normal_file_prefix + "normal_files.txt"
test_pneumonia_file_prefix = root_dir + "test/PNEUMONIA/"
test_pneumonia_file_path = test_pneumonia_file_prefix + "pneumonia_files.txt"

train_normal = []
train_pneumonia = []
test_normal = []
test_pneumonia = []

def load_train_file_paths():
	#1. normal
	f = open(train_normal_file_path, "r")
	l = f.readlines()	
	for i in range(len(l)):
		s = l[i]
		s = s[:len(s) - 1]
		train_normal.append(s)
	#print(train_normal)
	#print('LEN ', len(train_normal))
	f.close()

	#2. pneumonia
	f = open(train_pneumonia_file_path, "r")
	l = f.readlines()	
	for i in range(len(l)):
		s = l[i]
		s = s[:len(s) - 1]
		train_pneumonia.append(s)
	#print(train_pneumonia)
	#print('LEN ', len(train_pneumonia))
	f.close()


def load_test_file_paths():
	#1. normal
	f = open(test_normal_file_path, "r")
	l = f.readlines()	
	for i in range(len(l)):
		s = l[i]
		s = s[:len(s) - 1]
		test_normal.append(s)
	#print(test_normal)
	#print('LEN ', len(test_normal))
	f.close()

	#2. pneumonia
	f = open(test_pneumonia_file_path, "r")
	l = f.readlines()	
	for i in range(len(l)):
		s = l[i]
		s = s[:len(s) - 1]
		test_pneumonia.append(s)
	#print(test_pneumonia)
	#print('LEN ', len(test_pneumonia))
	f.close()

load_train_file_paths()
load_test_file_paths()

# Where to log outputs
logger = colorlogger("logs/", log_name="logs_all.txt")
discriminator_logger = colorlogger("logs/", log_name="logs_D(x);1.txt")
fake_logger = colorlogger("logs/", log_name="logs_D(G(z));0.txt")
generator_logger = colorlogger("logs/", log_name="logs_D(G(z));1.txt")
real_classifier_logger = colorlogger("logs/", log_name="logs_C(x).txt")
fake_classifier_logger = colorlogger("logs/", log_name="logs_C(G(z)).txt")
total_logger = colorlogger("logs/", log_name="logs_loss_total.txt")
train_accuracy_logger = colorlogger("logs/", log_name="logs_train_classifier_acc.txt")

epochs = 100
epoch_imgs = 10000 #how many images are defined as an epoch

#6. Randomly sample training images 
def sample_train_images_randomly():
    inputs = []
    labels = []
    for i in range(batch_size):
        normal_or_pneumonia = random.randint(0, 1)
        if normal_or_pneumonia == 0: #normal
        	sz = len(train_normal)
        	img_prefix = train_normal_file_prefix
        	img_paths_arr = train_normal
        else:
        	sz = len(train_pneumonia)
        	img_prefix = train_pneumonia_file_prefix
        	img_paths_arr = train_pneumonia
        
        img_id = random.randint(0, sz - 1)
        img_path = img_prefix + img_paths_arr[img_id]
        #print('Loading ', img_path)
        
        if not osp.exists(img_path):
        	print('Image ', img_path, 'does not exist?')
        else:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            img = img / 256.0
            #on my laptop it needs to be divided by 256 not sure elsewhere to be in the range[0, 1]
            img = cv2.resize(img, (img_size, img_size))
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
            img = img.reshape((img.shape[0], img.shape[1], 1))
            img = img.permute(2, 0, 1).data.cpu().numpy()
            inputs.append(img)
        
        if normal_or_pneumonia == 0:
        	labels.append(0)
        else:
        	labels.append(1)

    TRAIN_AUG = torch.nn.Sequential(
            
            T.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.33), ratio=(0.75, 1.3333333333333333)),
            T.Normalize(
                mean=torch.tensor([0.485]),
                std=torch.tensor([0.229])),
        )


    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs)
    inputs = TRAIN_AUG(inputs)
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]))
    labels = torch.from_numpy(labels).long()
    return inputs, labels



# models
netG = _netG(1) #DCGAN_generator(1) # #DCGAN_generator(1)
netD = _netD(1) #DCGAN_discriminator(1) # #DCGAN_discriminator(1)
netC = ResNetBackbone(50, num_classes = 2) #ResNet18() #normal or pneumonia
netC.init_weights()

netG = DataParallelModel(netG).cuda()
netD = DataParallelModel(netD).cuda()
netC = DataParallelModel(netC).cuda()

# optimizers 
blr_d = 0.0005
blr_g = 0.001
blr_c = 0.0005
optD = optim.Adam(netD.parameters(), lr=blr_d, betas=(0.5, 0.999), weight_decay = 1e-3)
optG = optim.Adam(netG.parameters(), lr=blr_g, betas=(0.5, 0.999))
optC = optim.Adam(netC.parameters(), lr=blr_c, betas=(0.5, 0.999), weight_decay = 1e-3)

# losses 
# 1) for discriminator and generator)
bce_loss = nn.BCELoss()
bce_loss = DataParallelCriterion(bce_loss, device_ids=[0])
# 2) for classifier
criterion = nn.CrossEntropyLoss()
criterion = DataParallelCriterion(criterion, device_ids=[0])

advWeight = 0.1 # adversarial weight

#5. Training loop
def train(total_trained_samples):
  netD.train()
  netG.train()
  for epoch in range(epochs):
    netC.train() #classifier

    total_train = 0
    correct_train = 0
    for steps in range(epoch_imgs // batch_size):
    
    	start = torch.cuda.Event(enable_timing=True)
    	end = torch.cuda.Event(enable_timing=True)
    	start.record()

    	inputs, labels = sample_train_images_randomly()
    	inputs = inputs.cuda()
    	labels = labels.cuda()
    	#print('sampling done')
    	# create label arrays
    	true_label = torch.ones(batch_size, device=device)
    	fake_label = torch.zeros(batch_size, device=device)

    	# noise vector z (r)
    	r = torch.randn(batch_size, 100, 1, 1, device=device)
    	fakeImageBatch = netG(r)
    	#print(fakeImageBatch.shape)

    	# train discriminator on real images
    	predictionsReal = netD(inputs)
    	lossDiscriminator = bce_loss(predictionsReal, true_label) #labels = 1
    	lossDiscriminator.backward(retain_graph = True)

    	# train discriminator on fake images
    	predictionsFake = netD(fakeImageBatch)
    	lossFake = bce_loss(predictionsFake, fake_label) #labels = 0
    	lossFake.backward(retain_graph = True)
    	for d_step in range(1): # update discriminator 5x per generator step
    		optD.step() # update discriminator parameters

    	# train generator
    	optG.zero_grad()
    	predictionsFake = netD(fakeImageBatch)
    	lossGenerator = bce_loss(predictionsFake, true_label) #labels = 1
    	lossGenerator.backward(retain_graph = True)
    	for g_step in range(3):
    		optG.step()

    	torch.autograd.set_detect_anomaly(True)
    	fakeImageBatch = fakeImageBatch.detach().clone()

    	# train classifier on real data
    	predictions = netC(inputs)
    	realClassifierLoss = criterion(predictions, labels)
    	realClassifierLoss.backward(retain_graph=True)

    	# update classifier's gradient on real data
    	optC.step()
    	optC.zero_grad()

    	# update the classifer on fake data
    	# run classifer on fake directly
    	predictionsFake = netC(fakeImageBatch)
    	# get a tensor of the labels that are most likely according to model
    	predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
    	confidenceThresh = .2

    	# psuedo labeling threshold
    	probs = F.softmax(predictionsFake, dim=1)
    	mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
    	toKeep = mostLikelyProbs > confidenceThresh
    	if sum(toKeep) != 0:
    		fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep]) * advWeight
    		fakeClassifierLoss.backward()

    	optC.step()

    	# reset the gradients
    	optD.zero_grad()
    	optG.zero_grad()
    	optC.zero_grad()


    	end.record()
    	# Waits for everything to finish running
    	torch.cuda.synchronize()
    	total_trained_samples += batch_size
    	# Logging
    	total_loss = lossDiscriminator + lossFake + lossGenerator + realClassifierLoss + fakeClassifierLoss
    	if steps % display_per_iters == 0:
    		print('Epoch {:3d} Step {:5d}/{:5d} L_total is {:6.2f} L_D is {:6.2f} L_G is {:6.2f} L_C is {:6.2f} BCE_D(x) is {:6.2f} BCE_D(G(z)) is {:6.2f} Tot Trained {:7d} '.format(epoch, steps, epoch_imgs // batch_size, total_loss.item(), lossDiscriminator.item() + lossFake.item(), lossGenerator.item(), realClassifierLoss.item(), lossDiscriminator.item(), lossFake.item(), total_trained_samples))
    	logger.info('Epoch {:3d} Step {:5d}/{:5d} L_total is {:6.2f} L_D is {:6.2f} L_G is {:6.2f} L_C is {:6.2f} BCE_D(x) is {:6.2f} BCE_D(G(z)) is {:6.2f} Tot Trained {:7d} '.format(epoch, steps, epoch_imgs // batch_size, total_loss.item(), lossDiscriminator.item() + lossFake.item(), lossGenerator.item(), realClassifierLoss.item(), lossDiscriminator.item(), lossFake.item(), total_trained_samples))
    	discriminator_logger.info('{:6.2f}'.format(lossDiscriminator.item()))
    	fake_logger.info('{:6.2f}'.format(lossFake.item()))
    	generator_logger.info('{:6.2f}'.format(lossGenerator.item()))
    	real_classifier_logger.info('{:6.2f}'.format(realClassifierLoss.item()))
    	if sum(toKeep) != 0:
    		fake_classifier_logger.info('{:6.2f}'.format(fakeClassifierLoss.item()))
    	total_logger.info('{:6.2f}'.format(total_loss.item()))

    	# save gan image
    	if steps % save_gan_per_iters == 0:    		
    		gridOfFakeImages = torchvision.utils.make_grid(fakeImageBatch.cpu())
    		torchvision.utils.save_image(gridOfFakeImages, save_gan_img_folder_prefix + str(epoch) + '_' + str(steps) + '.png')
    	
    	if steps % show_train_classifier_acc_per_iters == 0:
    		netC.eval()
    		# accuracy
    		_, predicted = torch.max(predictions, 1)
    		total_train += labels.size(0)
    		correct_train += predicted.eq(labels.data).sum().item()
    		train_accuracy = 100 * correct_train / total_train
    		logger.info('                             Train Accuracy (classifier) is {:6.2f}'.format(train_accuracy))
    		train_accuracy_logger.info('{:6.2f}'.format(train_accuracy))
    		netC.train()

    	# current state of the trained model
    	state = {
            'epoch': epoch,
            'iter': steps, 
            'netD': netD.state_dict(),
            'netG': netG.state_dict(),
            'netC': netC.state_dict(),
            'optD': optD.state_dict(),
            'optG': optG.state_dict(),
            'optC': optC.state_dict(),
            'total_trained_samples': total_trained_samples
                }
    	if (steps * batch_size) % save_per_samples == 0:
    		model_file = 'models/' + model_ckpt_prefix + 'bak_' + str(epoch + 1) + '_' + str((steps * batch_size) // save_per_samples) + '.pth'
    		torch.save(state, model_file)

    model_file = 'models/' + model_ckpt_prefix + 'epo_' + str(epoch + 1) + '.pth'
    torch.save(state, model_file)
  return total_trained_samples
total_trained_samples = 0
total_trained_samples = train(total_trained_samples)
 
            
  
      
      
