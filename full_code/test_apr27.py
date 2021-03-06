import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
from scipy import linalg
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



# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h * w)
        return G

class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return(out)


class ResDeconvNet(nn.Module):
    def __init__(self, backbone):
        super(ResDeconvNet, self).__init__()
        self.backbone = backbone

    def forward(self, x, y):
        x = torch.cat((x, y), dim = 1)
        x = self.backbone(x)

        return x


# Conv Layer
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride) #, padding)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Upsample Conv Layer
class UpsampleConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        if self.upsample:
            x = self.upsample(x)
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# Residual Block
#   adapted from pytorch tutorial
#   https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-
#   intermediate/deep_residual_network/main.py
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out 


#vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool6 = nn.MaxPool2d(kernel_size=8, stride=8)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool6 = nn.AvgPool2d(kernel_size=8, stride=8)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        #out['p6'] = self.pool6(out['r54'])
        return [out[key] for key in out_keys]



model_dir = 'F:/nst/ist/Models/' #os.getcwd() + '/Models/'
#get network
vgg = VGG()

vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
for param in vgg.parameters():
    param.requires_grad = False
if torch.cuda.is_available():
    vgg = DataParallelModel(vgg).cuda()

#3.    possible l

# Image Transform Network
class ImageTransformNet(nn.Module):
    def __init__(self):
        super(ImageTransformNet, self).__init__()
        
        # nonlineraity
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # encoding layers
        self.conv1 = ConvLayer(6, 32, kernel_size=9, stride=1)
        self.in1_e = nn.InstanceNorm2d(32, affine=True)

        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2_e = nn.InstanceNorm2d(64, affine=True)

        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3_e = nn.InstanceNorm2d(128, affine=True)

        # residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)

        # decoding layers
        self.deconv3 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2 )
        self.in3_d = nn.InstanceNorm2d(64, affine=True)

        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2 )
        self.in2_d = nn.InstanceNorm2d(32, affine=True)

        self.deconv1 = UpsampleConvLayer(32, 3, kernel_size=9, stride=1)
        self.in1_d = nn.InstanceNorm2d(3, affine=True)

    def forward(self, x):
        # encode
        y = self.relu(self.in1_e(self.conv1(x)))
        y = self.relu(self.in2_e(self.conv2(y)))
        y = self.relu(self.in3_e(self.conv3(y)))

        # residual layers
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)

        # decode
        y = self.relu(self.in3_d(self.deconv3(y)))
        y = self.relu(self.in2_d(self.deconv2(y)))
        #y = self.tanh(self.in1_d(self.deconv1(y)))
        y = self.deconv1(y)

        return y

    def init_weights(self):
        a = 1


def get_deconv_net(is_train):
    backbone_nst = ImageTransformNet()# ResNetBackbone(18, is_pose_net = False)# ImageTransformNet() #ResNetBackbone(18, is_pose_net = False)
    if is_train:
        backbone_nst.init_weights()
        
    model_deconv = ResDeconvNet(backbone_nst)

    return model_deconv


model_deconv = get_deconv_net(True)
model_deconv = DataParallelModel(model_deconv).cuda()

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


class _netG64(nn.Module):
    def __init__(self, ngpu):
        super(_netG64, self).__init__()
        self.ngpu = ngpu
        nz = 100 # noise dimension
        ngf = 64 # number of features map on the first layer
        nc = 1 # number of channels
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16 
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(True),            
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
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


class _netG256(nn.Module):
    def __init__(self, ngpu):
        super(_netG256, self).__init__()
        self.ngpu = ngpu
        nz = 100 # noise dimension
        ngf = 64 # number of features map on the first layer
        nc = 1 # number of channels
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 32),
            nn.ReLU(True),
            # state size. (ngf*32) x 4 x 4
            nn.ConvTranspose2d(ngf * 32, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. (ngf*16) x 8 x 8
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 16 x 16 
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 32 x 32
            nn.ConvTranspose2d(ngf * 4,     ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 64 x 64
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 128 x 128
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 256 x 256
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


class _netD64(nn.Module):
    def __init__(self, ngpu):
        super(_netD64, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )

    def forward(self, input):
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

class _netD256(nn.Module):
    def __init__(self, ngpu):
        super(_netD256, self).__init__()
        self.ngpu = ngpu
        ndf = 64
        nc = 1
        self.main = nn.Sequential(
            # input is (nc) x 256 x 256
            nn.Conv2d(nc, ndf, 4, stride=2, padding=1, bias=False), 
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 128 x 128
            nn.Conv2d(ndf, ndf * 2, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 64 x 64
            nn.Conv2d(ndf * 2, ndf * 4, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 32 x 32 
            nn.Conv2d(ndf * 4, ndf * 8, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 16 x 16
            nn.Conv2d(ndf * 8, ndf * 16, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*16) x 8 x 8
            nn.Conv2d(ndf * 16, ndf * 32, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 32),
            nn.LeakyReLU(0.2, inplace=True),            
            # state size. (ndf*32) x 4 x 4
            nn.Conv2d(ndf * 32, 1, 4, stride=1, padding=0, bias=False),
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
root_dir = "E:/ml/" # chest x-ray 14
n_classes = 15 # 0 is normal : no finding
batch_size = 22
img_size = 128
display_per_iters = 5 # how many iterations before outputting to the console window
save_gan_per_iters = 5 # save gan images per this iterations
save_gan_img_folder_prefix = root_dir + "train_fake/"
show_train_classifier_acc_per_iters = 1000000 # how many iterations before showing train acc of classifier
show_test_classifier_acc_per_iters = 15 # 
save_per_samples = 2000 # save a checkpoint per forward run of this number of samples
model_ckpt_prefix = 'ecgan-chest-xray14'

use_label_smoothing = True
smoothing = 0.1

# define device 
device = torch.device("cuda:0")

# The files that contain paths of all images
image_index_list_file = root_dir + "image_index.txt"
labels_file = root_dir + "labels.txt"
train_val_list_file = root_dir + "train_val_list.txt"
test_list_file = root_dir + "test_list.txt"
img_folders = { 'images_001/', 'images_002/', 'images_003/', 'images_005/', 'images_008/', 'images_011/', 'images_006/', 'images_007/', 'images_004/', 'images_009/', 'images_010/', 'images_012/'}
suffix = 'images/'
image_index_list = []    
labels_list = []
img_index_2_label_dict = {}
label_name_dict = { 'No Finding': 0,
                    'Atelectasis': 1, 
                    'Cardiomegaly': 2, 
                    'Effusion': 3, 
                    'Infiltration': 4, 
                    'Mass': 5, 
                    'Nodule': 6, 
                    'Pneumonia': 7, 
                    'Pneumothorax': 8, 
                    'Consolidation': 9, 
                    'Edema': 10, 
                    'Emphysema': 11, 
                    'Fibrosis': 12, 
                    'Pleural_Thickening': 13, 
                    'Hernia': 14}
# list of img paths                    
train_val_list = []
test_list = []
train_val_labels = []
test_labels = []

def load_image_index_and_list():
    #1. image index list (all) e.g. 00000583_023.png
    f_list = open(image_index_list_file, "r")
    l = f_list.readlines()
    for line in l:
        if line != '\n':
            image_index_list.append(line[0:len(line) - 1])
    f_list.close()

    #2. labels e.g. Cardiomegaly|Effusion
    f_list = open(labels_file, "r")
    l = f_list.readlines()
    for line in l:
        if line != '\n':
            labels_list.append(line[0:len(line) - 1])
    f_list.close()
    return 

def build_img_2_label_dict():
    for i in range(len(image_index_list)):
        img_id = image_index_list[i]
        label = labels_list[i]
        img_index_2_label_dict.update({img_id: label})
    
def load_train_val_list():
    #1. original train_val_list.txt
    f_list = open(train_val_list_file, "r")
    l = f_list.readlines()        
    for line in l:
        s = line 
        if s[len(s) - 1] == '\n':
            s = s[:len(s) - 1]
        img_name = s
        this_label = img_index_2_label_dict[img_name]        
        find_or = this_label.find('|')        
        if find_or != -1:
            continue
        # See if this image exists        
        for folders in img_folders:
            img_path = root_dir + folders + suffix + img_name            
            if not osp.exists(img_path):
                continue
            train_val_list.append(img_path)            
            this_label = label_name_dict[this_label]
            train_val_labels.append(this_label)
    print('There are {:6d} images in train/val.\n'.format(len(train_val_list)))
    f_list.close()

def load_test_list():
    #1. original test_list.txt
    f_list = open(test_list_file, "r")
    l = f_list.readlines()        
    for line in l:
        s = line 
        if s[len(s) - 1] == '\n':
            s = s[:len(s) - 1]
        img_name = s
        this_label = img_index_2_label_dict[img_name]        
        find_or = this_label.find('|')        
        if find_or != -1:
            continue
        # See if this image exists        
        for folders in img_folders:
            img_path = root_dir + folders + suffix + img_name            
            if not osp.exists(img_path):
                continue
            test_list.append(img_path)            
            this_label = label_name_dict[this_label]
            test_labels.append(this_label)
    print('There are {:6d} images in test.\n'.format(len(test_list)))
    f_list.close()

	
load_image_index_and_list()
build_img_2_label_dict()
load_train_val_list()
load_test_list()

# Where to log outputs
logger = colorlogger("logs/", log_name="logs_all.txt")
discriminator_logger = colorlogger("logs/", log_name="logs_D(x);1.txt")
fake_logger = colorlogger("logs/", log_name="logs_D(G(z));0.txt")
generator_logger = colorlogger("logs/", log_name="logs_D(G(z));1.txt")
real_classifier_logger = colorlogger("logs/", log_name="logs_C(x).txt")
fake_classifier_logger = colorlogger("logs/", log_name="logs_C(G(z)).txt")
total_logger = colorlogger("logs/", log_name="logs_loss_total.txt")
train_accuracy_logger = colorlogger("logs/", log_name="logs_train_classifier_acc.txt")
test_accuracy_logger = colorlogger("logs/", log_name="logs_test_classifier_acc.txt")
avg_kl_logger = colorlogger("logs/", log_name="logs_avg_kl.txt")

epochs = 100
epoch_imgs = 3483 #how many images are defined as an epoch

#6. Randomly sample training images 
def sample_train_images_randomly():
    inputs = []
    labels = []
    for i in range(batch_size):
        sz = len(train_val_list)
        img_id = random.randint(0, sz - 1)
        img_path = train_val_list[img_id]
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
        this_label = train_val_labels[img_id]
        #print(this_label)
        labels.append(this_label)        

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

#6. Randomly sample test images 
def sample_test_images_randomly():
    inputs = []
    labels = []
    for i in range(batch_size):
        sz = len(test_list)
        img_id = random.randint(0, sz - 1)
        img_path = test_list[img_id]
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
        this_label = test_labels[img_id]
        labels.append(this_label)        

    TEST_AUG = torch.nn.Sequential(
            
            #T.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.33), ratio=(0.75, 1.3333333333333333)),
            T.Normalize(
                mean=torch.tensor([0.485]),
                std=torch.tensor([0.229])),
        )
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs)
    inputs = TEST_AUG(inputs)
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]))
    labels = torch.from_numpy(labels).long()
    return inputs, labels


#6. Sequentially sample test images 
def sample_test_images_sequentially(lb, ub):
    inputs = []
    labels = []
    for i in range(batch_size):
        #sz = len(test_list)
        img_id = lb + i #random.randint(0, sz - 1)
        img_path = test_list[img_id]
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
        this_label = test_labels[img_id]
        labels.append(this_label)        

    TEST_AUG = torch.nn.Sequential(
            
            T.RandomResizedCrop((img_size, img_size), scale=(0.75, 1.33), ratio=(0.75, 1.3333333333333333)),
            T.Normalize(
                mean=torch.tensor([0.485]),
                std=torch.tensor([0.229])),
        )
    inputs = np.array(inputs)
    inputs = torch.from_numpy(inputs)
    inputs = TEST_AUG(inputs)
    labels = np.array(labels)
    labels = labels.reshape((labels.shape[0]))
    labels = torch.from_numpy(labels).long()
    return inputs, labels



#6.5 label smoothing
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()



def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, target, smoothing=0.2):        
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()
##
# version 1: use torch.autograd
class LabelSmoothSoftmaxCEV1(nn.Module):
    '''
    This is the autograd version, you can also try the LabelSmoothSoftmaxCEV2 that uses derived gradients
    '''

    def __init__(self, lb_smooth=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits, label):
        '''
        Same usage method as nn.CrossEntropyLoss:
            >>> criteria = LabelSmoothSoftmaxCEV1()
            >>> logits = torch.randn(8, 19, 384, 384) # nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nhw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        # overcome ignored label
        logits = logits.float() # use fp32 to avoid nan
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label.eq(self.lb_ignore)
            n_valid = ignore.eq(0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()

        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * lb_one_hot, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss



# models
# _net: 128x128
# DCGAN_: 64X64
netG = _netG(1) #DCGAN_generator(1) 
netD = _netD(1) #DCGAN_discriminator(1)
netC = ResNetBackbone(50, num_classes = 15) #ResNet18() #normal or pneumonia
netC.init_weights()

netG = DataParallelModel(netG).cuda()
netD = DataParallelModel(netD).cuda()
netC = DataParallelModel(netC).cuda()

# optimizers 
blr_d = 0.0001
blr_g = 0.0001
blr_c = 0.0001
optD = optim.Adam(netD.parameters(), lr=blr_d, betas=(0.5, 0.999), weight_decay = 1e-3)
optG = optim.Adam(netG.parameters(), lr=blr_g, betas=(0.5, 0.999))
optC = optim.Adam(netC.parameters(), lr=blr_c, betas=(0.5, 0.999), weight_decay = 1e-3)

# losses 
# 1) for discriminator and generator)
bce_loss = nn.BCELoss()
bce_loss = DataParallelCriterion(bce_loss, device_ids=[0])
# 2) for classifier
if use_label_smoothing:
    criterion = LabelSmoothSoftmaxCEV1(lb_smooth=smoothing, ignore_index=255, reduction='mean')
else:
    criterion = nn.CrossEntropyLoss() #LabelSmoothingCrossEntropy() #
criterion = DataParallelCriterion(criterion, device_ids=[0])

advWeight = 0.25 # adversarial weight

#5. Loading trained weights
def load_my_state_dict(model, state_dict):
 
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        #own_state[name] = deepcopy(param)
        own_state[name].copy_(param)
    #print(own_state)
    return own_state

def load_model(model_path):
    ckpt = torch.load(model_path) 
    start_epoch = ckpt['epoch'] + 1
    #netD.load_state_dict(ckpt['netD'])    
    #netG.load_state_dict(ckpt['netG'])    
    netC.load_state_dict(ckpt['netC'])
    #optD.load_state_dict(ckpt['optD'])
    #optG.load_state_dict(ckpt['optG'])
    #optC.load_state_dict(ckpt['optC'])
    total_trained_samples = ckpt['total_trained_samples']
    return start_epoch, total_trained_samples


def distance(X, Y, sqrt=True):
    nX = X.size(0)
    nY = Y.size(0)
    
    X = X.view(nX,-1).cuda()
    X2 = (X*X).sum(1).resize(nX,1)
    Y = Y.view(nY,-1).cuda()
    Y2 = (Y*Y).sum(1).resize(nY,1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX,nY) + Y2.expand(nY,nX).transpose(0,1) - 2*torch.mm(X,Y.transpose(0,1)))

    #del X, X2, Y, Y2
    
    if sqrt:
        M = ((M+M.abs())/2).sqrt()

    return M

def mmd(Mxx, Mxy, Myy, sigma = 1):
    scale = Mxx.mean()
    Mxx = torch.exp(-Mxx/(scale*2*sigma*sigma))
    Mxy = torch.exp(-Mxy/(scale*2*sigma*sigma))
    Myy = torch.exp(-Myy/(scale*2*sigma*sigma))
    a = Mxx.mean()+Myy.mean()-2*Mxy.mean()  
    if a.item() > 1e-6:
    	mmd = torch.sqrt(a)
    	#print(mmd)
    else:
    	return -1
    return mmd    

#6. Testing loop
def test_all(file, epoch, best_accuracy, best_epoch):
    netD.eval()
    netG.eval()
    total_test = 0
    correct_test = 0
    test_num = 100    
    # sample some test images
    with torch.no_grad():
         for steps in range(test_num):
             inputs, labels = sample_test_images_sequentially(steps * batch_size, (steps + 1) * batch_size)
             inputs = inputs.cuda()
             labels = labels.cuda()
             outputs = netC(inputs)
        
             # accuracy
             _, predicted = torch.max(outputs.data, 1)
             total_test += labels.size(0)
             correct_test += predicted.eq(labels.data).sum().item()
             test_accuracy = 100 * correct_test / total_test
    print('Epoch {:5d} test acc {:6.2f} Current Best {:6.2f} \n'.format(epoch, test_accuracy, best_accuracy))
    file.write('Epoch {:5d} test acc {:6.2f} Current Best {:6.2f} \n'.format(epoch, test_accuracy, best_accuracy))
    if test_accuracy > best_accuracy:
       best_accuracy = test_accuracy
       best_epoch = epoch
    return test_accuracy, best_accuracy, best_epoch     
total_trained_samples = 0
torch.manual_seed(42)
start_epoch = 0
file = open('best.txt', 'w')
best_acc = 0
best_epoch = 0
for epoch in range(36, 100):
    start_epoch, total_trained_samples = load_model('../models_apr28/ecgan-chest-xray14epo_' + str(epoch) + '.pth')
    netC.eval() #classifier
    test_acc, best_acc, best_epoch = test_all(file, epoch, best_acc, best_epoch)
    
file.close()
 
            
  
      
      
