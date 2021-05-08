import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math
import numpy as np
import sklearn.metrics as metrics
from scipy import linalg
from os import path as osp
import cv2
import random
import matplotlib.pyplot as plt
import pdb

# Hyper-parameters
blr_d = 0.0005
blr_g = 0.001
blr_c = 0.0005
alpha_g = 2.0
alpha_d = 0.5
alpha_kl = 0.25
alpha_mmd = 0
alpha_c = 1.0
alpha_nst = 0.000
confidenceThresh = 0.75 #disable
train_g_more = True
train_d_more = not(train_g_more)
g_vs_d = 3
d_vs_g = 3
per_class_sample = True
weights_prefix = '../models_allclasses_wis_only/'
weights_middle_name = "_allclasses_is_mmd_epo_"
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
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
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
        #org_resnet.pop('conv1.weight', None)
        #org_resnet.pop('conv1.bias', None)
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
batch_size = 12
img_size = 128
display_per_iters = 8 # how many iterations before outputting to the console window
save_gan_per_iters = 5 # save gan images per this iterations
save_gan_img_folder_prefix = root_dir + "train_fake_allclasses_wis_only/"
show_train_classifier_acc_per_iters = 1000000 # how many iterations before showing train acc of classifier
# Deprecated in favor of per-epoch test
show_test_classifier_acc_per_iters = 290 # 
save_per_samples = 2000 # save a checkpoint per forward run of this number of samples
model_ckpt_prefix = 'ecgan-chest-xray14'

use_label_smoothing = False
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
label_id_dict = { 0: 'No Finding',
                  1:  'Atelectasis', 
                  2:  'Cardiomegaly', 
                  3:  'Effusion', 
                  4:  'Infiltration', 
                  5:  'Mass', 
                  6:  'Nodule', 
                  7:  'Pneumonia', 
                  8:  'Pneumothorax', 
                  9:  'Consolidation', 
                 10:  'Edema', 
                 11: 'Emphysema', 
                 12: 'Fibrosis', 
                 13: 'Pleural_Thickening', 
                 14: 'Hernia'}

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
train_val_per_label_list = []

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

label_train_cnt = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# True useful labels
usable_label_arr = np.zeros(n_classes)
usable_label_arr[7] = 1
usable_label_arr[10] = 1
usable_label_arr[14] = 1
    
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
        this_label = label_name_dict[this_label]
        if usable_label_arr[this_label] == 0:        
           continue  
        
        #if this_label == 0: #imbalance 
        #   continue # no finding    
        # See if this image exists        
        for folders in img_folders:
            img_path = root_dir + folders + suffix + img_name            
            if not osp.exists(img_path):
                continue
            train_val_list.append(img_path)            
            #this_label = label_name_dict[this_label]
            train_val_labels.append(this_label)
            label_train_cnt[this_label] += 1
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
        this_label = label_name_dict[this_label]
        if usable_label_arr[this_label] == 0:        
           continue  
        
        if this_label == 0:
            continue #ignore no finding
        # See if this image exists        
        for folders in img_folders:
            img_path = root_dir + folders + suffix + img_name            
            if not osp.exists(img_path):
                continue
            test_list.append(img_path)            
            #this_label = label_name_dict[this_label]
            test_labels.append(this_label)
    print('There are {:6d} images in test.\n'.format(len(test_list)))
    #print(test_labels)
    print("Len of test list: {:5d}".format(len(test_list)))
    f_list.close()


def load_gan_and_vanilla(model_path_gan, model_path_cla):
    ckpt = torch.load(model_path_gan) 
    start_epoch = ckpt['epoch'] + 1
    netD.load_state_dict(ckpt['netD'])    
    netG.load_state_dict(ckpt['netG'])    
    
    total_trained_samples = ckpt['total_trained_samples']

    ckpt = torch.load(model_path_cla)     
    netC.load_state_dict(ckpt['netC'])    
    

    return start_epoch, total_trained_samples


def store_per_label_train_list():
    for cla in range(n_classes):
        train_val_per_label_list.append([])
    for i in range(len(train_val_list)):
        cur_label = train_val_labels[i]
        train_val_per_label_list[cur_label].append(i)
    for cla in range(n_classes):
        print("Class {:3d} {:20s} #(samples) {:5d}".format(cla, label_id_dict[cla], len(train_val_per_label_list[cla])))
    print("Start training ==========================\n\n")  
load_image_index_and_list()
build_img_2_label_dict()
load_train_val_list()
load_test_list()
store_per_label_train_list()
# Where to log outputs
logger = colorlogger("logs/", log_name="logs_all_is.txt")
discriminator_logger = colorlogger("logs/", log_name="logs_D(x);1.txt")
fake_logger = colorlogger("logs/", log_name="logs_D(G(z));0.txt")
generator_logger = colorlogger("logs/", log_name="logs_D(G(z));1.txt")
real_classifier_logger = colorlogger("logs/", log_name="logs_C(x).txt")
fake_classifier_logger = colorlogger("logs/", log_name="logs_C(G(z)).txt")
total_logger = colorlogger("logs/", log_name="logs_loss_total.txt")
train_accuracy_logger = colorlogger("logs/", log_name="logs_train_classifier_acc.txt")
test_accuracy_logger = colorlogger("logs/", log_name="logs_test_classifier_acc.txt")
avg_kl_logger = colorlogger("logs/", log_name="logs_avg_kl.txt")

epochs = 1000
epoch_imgs = 3483 #how many images are defined as an epoch

#6. Randomly sample training images 
def sample_train_images_randomly():
    inputs = []
    labels = []
    for i in range(batch_size):
        if per_class_sample:
            per_class_sample_prob = []
            for cla in range(n_classes):
                per_class_sample_prob.append(0.0)
            sum = 0.0
            for cla in range(n_classes):
                if usable_label_arr[cla] == 0:                
                    continue
                per_class_sample_prob[cla] = 1.0 / len(train_val_per_label_list[cla])
                #per_class_sample_prob[cla] = 1.0 / float(n_classes)
                sum += per_class_sample_prob[cla]
            for cla in range(n_classes):
                per_class_sample_prob[cla] /= sum
                #print(per_class_sample_prob[cla])
            mul_factor = 100000
            x = random.randint(0, mul_factor - 1)
            sum = 0.0
            for cla in range(n_classes):
                if usable_label_arr[cla] == 0:                
                    continue
                lb = sum * mul_factor
                ub = (sum + per_class_sample_prob[cla]) * mul_factor
                if lb <= x and x < ub:
                    break
                sum += per_class_sample_prob[cla]
            #print(cla)
            ll = len(train_val_per_label_list[cla])
            within_cla_id = random.randint(0, ll - 1)
            img_id = train_val_per_label_list[cla][within_cla_id]
        else:
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

advWeight = 0.125 # adversarial weight

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
    netD.load_state_dict(ckpt['netD'])    
    netG.load_state_dict(ckpt['netG'])    
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


#6. Testing loop
def test_all(file, epoch, best_accuracy, best_epoch, best_per_class_acc):
    netC.eval()
    total_test = 0
    correct_test = 0
    test_num = 24
    total_test_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    correct_test_per_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    per_class_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # sample some test images
    # [-1, 1] or [0, 1] 
    y_test = np.zeros((n_classes, test_num * batch_size))
    preds_test = np.zeros((n_classes, test_num * batch_size))
    test_id = 0
    with torch.no_grad():
         for steps in range(test_num):
             inputs, labels = sample_test_images_sequentially(steps * batch_size, (steps + 1) * batch_size)
             inputs = inputs.cuda()
             labels = labels.cuda()
             inputs_cat = torch.cat((inputs, inputs, inputs), dim = 1)
        
            
             outputs = netC(inputs_cat)
             # accuracy
             sig = nn.Softmax(dim = 1)
             outputs_normalized = sig(outputs)
             outputs_normalized = outputs_normalized.data.cpu().numpy()
             labels_data = labels.data.cpu().numpy()
             #print(outputs_normalized)
             for b in range(batch_size):                    
                 for cla in range(n_classes):
                     preds_test[cla][test_id] = outputs_normalized[b][cla]
                     y_test[cla][test_id] = (labels_data[b] == cla)
                     #print("{:5d} {:5d} {:5.0f}".format(b, cla, y_test[cla][test_id]))
                 test_id += 1   
             _, predicted = torch.max(outputs.data, 1)
             total_test += labels.size(0)
             for i in range(batch_size):
                 cur_label = labels[i]
                 pred_label = predicted[i]
                 if cur_label == pred_label:
                    correct_test_per_class[cur_label] += 1
                 total_test_per_class[cur_label] += 1
             correct_test += predicted.eq(labels.data).sum().item()
             test_accuracy = 100 * correct_test / total_test
    # AUROC
    for cla in range(n_classes):
        if usable_label_arr[cla] == 0:
            continue
        fpr, tpr, threshold = metrics.roc_curve(y_test[cla], preds_test[cla])
        roc_auc = metrics.auc(fpr, tpr)
        print("Class {:4d} Name {:20s} AUROC {:6.2f} #(Samples) {:5d}".format(cla, label_id_dict[cla], roc_auc , total_test_per_class[cla]))


    # Top 1 Acc
    for cla in range(n_classes):
        if total_test_per_class[cla] != 0:
           per_class_acc[cla] = 100 * correct_test_per_class[cla] / total_test_per_class[cla]
           print("Class {:4d} acc {:6.2f} all {:5d}".format(cla, 100 * correct_test_per_class[cla] / total_test_per_class[cla], total_test_per_class[cla]))
    
    #print('Epoch {:5d} test acc {:6.2f} Current Best {:6.2f} \n'.format(epoch, test_accuracy, best_accuracy))
    #file.write('Epoch {:5d} test acc {:6.2f} Current Best {:6.2f} \n'.format(epoch, test_accuracy, best_accuracy))
    if test_accuracy > best_accuracy:
       best_accuracy = test_accuracy
       best_epoch = epoch
       best_per_class_acc = per_class_acc
    logger.info('Epoch {:3d} Current Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(epoch, per_class_acc[0], per_class_acc[1], per_class_acc[2], per_class_acc[3], per_class_acc[4], per_class_acc[5], per_class_acc[6], per_class_acc[7], per_class_acc[8], per_class_acc[9], per_class_acc[10], per_class_acc[11], per_class_acc[12], per_class_acc[13], per_class_acc[14]))
  
    return test_accuracy, best_accuracy, best_epoch, best_per_class_acc


#6. Training loop
def train(total_trained_samples):
  netD.train()
  netG.train()
  total_train = 0
  correct_train = 0
  total_test = 0
  correct_test = 0
  best_acc = 0
  best_epoch = 0
  best_per_class_acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]    

  avg_most_likely_prob = 0.0
  cnt = 0
  print("Initial test before training")
  test_acc, best_acc, best_epoch, best_per_class_acc = test_all(logger, -1, best_acc, best_epoch, best_per_class_acc)
  print("Test acc {:6.2f} Best test acc {:6.2f} best epoch {:5d}".format(test_acc, best_acc, best_epoch))
  print('Epoch {:3d} Best Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(-1, best_per_class_acc[0], best_per_class_acc[1], best_per_class_acc[2], best_per_class_acc[3], best_per_class_acc[4], best_per_class_acc[5], best_per_class_acc[6], best_per_class_acc[7], best_per_class_acc[8], best_per_class_acc[9], best_per_class_acc[10], best_per_class_acc[11], best_per_class_acc[12], best_per_class_acc[13], best_per_class_acc[14]))
  logger.info('Epoch {:3d} Best Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(-1, best_per_class_acc[0], best_per_class_acc[1], best_per_class_acc[2], best_per_class_acc[3], best_per_class_acc[4], best_per_class_acc[5], best_per_class_acc[6], best_per_class_acc[7], best_per_class_acc[8], best_per_class_acc[9], best_per_class_acc[10], best_per_class_acc[11], best_per_class_acc[12], best_per_class_acc[13], best_per_class_acc[14]))
  netC.train()
  
  #N = 0
  #pynormal = torch.tensor([0.0]).cuda()
  #avg_kl = torch.tensor([0.0]) # we want to maximize E_G[D_{KL}(p(y|x) || p(y))] 
  #avg_kl = avg_kl.cuda()
  acc_d_step = 0 # update generator per n D steps e.g. n = 5
  acc_g_step = 0
  for epoch in range(start_epoch, epochs):
    netC.train() #classifier

    for steps in range(epoch_imgs // batch_size):
        for p in netD.parameters():
            p.requires_grad = True # to avoid computation 
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        inputs, labels = sample_train_images_randomly()
        inputs = inputs.cuda()
        labels = labels.cuda()
        inputs_cat = torch.cat((inputs, inputs, inputs), dim = 1)
        

    	#print('sampling done')
    	# create label arrays
        true_label = torch.ones(batch_size, device=device)
        fake_label = torch.zeros(batch_size, device=device)

    	# noise vector z (r)
        r = torch.randn(batch_size, 100, 1, 1, device=device)
        fakeImageBatch = netG(r)
    	#print(fakeImageBatch.shape)

    	# train discriminator on real images
        realImageBatch = inputs_cat.detach().clone()
        
        predictionsReal = netD(inputs)
    	#print(inputs.shape)
    	#print(predictionsReal.shape)
        lossDiscriminator = bce_loss(predictionsReal, true_label) * alpha_d #labels = 1
        lossDiscriminator.backward(retain_graph = True)

    	# train discriminator on fake images
        predictionsFake = netD(fakeImageBatch)
        lossFake = bce_loss(predictionsFake, fake_label) * alpha_d #labels = 0
        lossFake.backward(retain_graph = True)
        ld = lossDiscriminator.item() + lossFake.item()

        #optD.step()
        if train_g_more:
            if acc_g_step % g_vs_d == 0:
                optD.step()
                acc_d_step += 1
        else:
            optD.step()
            acc_d_step += 1


    	# train generator
        for p in netD.parameters():
            p.requires_grad = False # to avoid computation
        optG.zero_grad()
        predictionsFake = netD(fakeImageBatch)
        lossGenerator = bce_loss(predictionsFake, true_label) * alpha_g #labels = 1
        lg = lossGenerator.item()
        #lossGenerator *= (ld / lossGenerator.item())
        lossGenerator.backward(retain_graph = True)
        # count steps of updating generating
        #if acc_d_step % d_vs_g == 0:
        if train_d_more:
            if acc_d_step % d_vs_g == 0:
                optG.step()
                acc_g_step += 1
        else:
            optG.step()
            acc_g_step += 1
        torch.autograd.set_detect_anomaly(True)
        fakeImageBatch = fakeImageBatch.detach().clone()

        # Neural Style Transfer
        style = torch.cat((inputs, inputs, inputs), dim=1)
        content = torch.cat((fakeImageBatch, fakeImageBatch, fakeImageBatch), dim=1)
        # optimized image given style: real and content: GAN images
        opt = model_deconv(style, content)
        style_layers = ['r11','r21','r31','r41', 'r51'] 
        loss_layers = style_layers 
        # Gram Matrix
        loss_fns = [GramMSELoss()] * len(style_layers)
        if torch.cuda.is_available():
           loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
        #these are good weights settings:
        style_weights = [1e3/n**2 for n in [64,128,256,512,512]] #[1e0, 0.002, 5e-4, 1e-5, 0.01]
        weights = style_weights 
        style_targets = [GramMatrix()(A).detach() for A in vgg(style, style_layers)]

        targets = style_targets
        # USE Deconv opt -> loss network -> 
        # style & content from deconv opt
        out = vgg(opt, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a,A in enumerate(out)]
        nst_loss = sum(layer_losses) * alpha_nst
        nst_loss.backward()
        if steps % display_per_iters == 0:
            print('Style: Real Content: GAN NST Style Gram Matrix Loss is {:6.2f}'.format(nst_loss.item()))
        logger.info('Style: Real Content: GAN NST Style Gram Matrix Loss is {:6.2f}'.format(nst_loss.item()))
        
    	# train classifier on real data
        
        predictions = netC(inputs_cat)
        realClassifierLoss = criterion(predictions, labels) * alpha_c
        realClassifierLoss.backward(retain_graph=True)

    	# update classifier's gradient on real data
        optC.step()
        optC.zero_grad()

    	# update the classifer on fake data
    	# run classifer on fake directly
        fakeImageBatch_cat = torch.cat((fakeImageBatch, fakeImageBatch, fakeImageBatch), dim = 1)
             
        predictionsFake = netC(fakeImageBatch_cat)
        
        predictedLabels = torch.argmax(predictionsFake, 1) # -> [0 , 5, 9, 3, ...]
        
        klconfidenceThresh = 1e-6 # if prob is > klThresh include this in the expectation of KL term
        
        # 2 p(y|x)
        probs = F.softmax(predictionsFake, dim=1)
    	# Compute MMD score 
        realImageBatch = inputs_cat.detach().clone()
        #print(realImageBatch.shape)
    	# remember to detach this thing and clone and retain_graph = True for backprop
        predictions_real = netC(realImageBatch)
        probs_real = F.softmax(predictions_real, dim=1)
        real = probs_real
        fake = probs
        Mxx = distance(real, real, False)
        Mxy = distance(real, fake, False)
        Myy = distance(fake, fake, False) 
        cur_mmd = mmd(Mxx, Mxy, Myy)
        if cur_mmd != -1:
            cur_mmd = cur_mmd * alpha_mmd
            cur_mmd.require_grad = True
            cur_mmd.backward(retain_graph = True)
    		#print('			mmd    is {:12.6f}'.format(cur_mmd.item()))    	
        else:
            cur_mmd = torch.tensor(-1.0)
            cur_mmd = cur_mmd.cuda()
    	# Compute IS score (KL)
    	# update p(y = normal)
    	#pynormal = torch.tensor([0.0]).cuda()
    	
        #update all 15 classes p(Y=y) y \in [0, 15]
        py = np.zeros(n_classes)
        py = torch.from_numpy(py).cuda()
        N = 0
        for i in range(batch_size):
            for j in range(n_classes): 
                py[j] += probs[i, j]
                py[j] = (py[j] * N + probs[i, j]) / (N + 1)
            N += 1
        py = py / batch_size
    	#print('P(Y = y) = ', py)

        # KL(p(y|x) || p(y)) within the batch
        avg_kl = torch.tensor([0.0]) # we want to maximize E_G[D_{KL}(p(y|x) || p(y))] 
        avg_kl = avg_kl.cuda()
        for i in range(batch_size):
            for j in range(n_classes): #Y = y   
                pycondx = probs[i, j]
                eps = 1e-20
                kl = pycondx * ((pycondx + eps).log() - (py[j] + eps).log()) #py[j] = p(Y=j)
                avg_kl += kl
        avg_kl = avg_kl / batch_size
    	
    	# inception score D_KL loss 
    	#   want to minimize so take -avg_kl min(-avg_kl) = max(avg_kl)
        kl_loss = -avg_kl * alpha_kl
        
        kl_loss.require_grad = True
    	#print('			kl loss is {:12.6f}'.format(kl_loss.item()))    	
        kl_loss.backward(retain_graph = True) 
        
        # psuedo labeling threshold
        mostLikelyProbs = np.asarray([probs[i, predictedLabels[i]].item() for  i in range(len(probs))])
        avg_cur_mostLikelyProbs = mostLikelyProbs.mean()
        avg_most_likely_prob += avg_cur_mostLikelyProbs
        cnt += 1
        if steps % display_per_iters == 0:
           print("Pseudo Prob {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}  {:4.1f}".format(mostLikelyProbs[0], mostLikelyProbs[1], mostLikelyProbs[2], mostLikelyProbs[3], mostLikelyProbs[4], mostLikelyProbs[5], mostLikelyProbs[6], mostLikelyProbs[7], mostLikelyProbs[8], mostLikelyProbs[9], mostLikelyProbs[10], mostLikelyProbs[11],))
           print("Current mean mostlikely prob {:4.1f} ====> Accumulative average {:4.1f}\n".format(avg_cur_mostLikelyProbs, avg_most_likely_prob / float(cnt)))
        #print(mostLikelyProbs)
        toKeep = mostLikelyProbs > confidenceThresh
        #print(sum(toKeep))
        if sum(toKeep) != 0:
            fakeClassifierLoss = criterion(predictionsFake[toKeep], predictedLabels[toKeep])  * alpha_c * advWeight
            fakeClassifierLoss.backward(retain_graph = True)
            #optC.step()
            #optC.zero_grad()
            
            real_vs_fake = 0
            real_num = int(real_vs_fake * sum(toKeep) / batch_size)
            #print(real_num)
            for l in range(real_num):
                inputs2, labels2 = sample_train_images_randomly()
                inputs2 = inputs2.cuda()
                labels2 = labels2.cuda()
                predictions2 = netC(inputs2)
                realClassifierLoss2 = criterion(predictions2, labels2) * 0.5
                realClassifierLoss2.backward(retain_graph=True)
                #optC.step()
                #optC.zero_grad()
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
        total_loss = lossDiscriminator + lossFake + cur_mmd + realClassifierLoss #
        if sum(toKeep) != 0:
            total_loss += fakeClassifierLoss
        if steps % display_per_iters == 0:  
            print('Epoch {:3d} Step {:5d}/{:5d} L_total is {:6.2f} L_D is {:6.2f} L_G is {:6.2f} L_C is {:6.2f} aKL is {:4.2f}  mmd is {:6.2f} Tot Trained {:7d} '.format(epoch, steps, epoch_imgs // batch_size, total_loss.item(), lossDiscriminator.item() + lossFake.item(), lg, realClassifierLoss.item(), avg_kl.item(), cur_mmd.item(), total_trained_samples)) #lossDiscriminator.item(), lossFake.item(),
        logger.info('Epoch {:3d} Step {:5d}/{:5d} L_total is {:6.2f} L_D is {:6.2f} L_G is {:6.2f} L_C is {:6.2f} aKL is {:4.2f}  mmd is {:6.2f} Tot Trained {:7d} '.format(epoch, steps, epoch_imgs // batch_size, total_loss.item(), lossDiscriminator.item() + lossFake.item(), lg, realClassifierLoss.item(), avg_kl.item(), cur_mmd.item(), total_trained_samples))
        discriminator_logger.info('{:6.2f}'.format(lossDiscriminator.item()))
        fake_logger.info('{:6.2f}'.format(lossFake.item()))
        generator_logger.info('{:6.2f}'.format(lossGenerator.item()))
        real_classifier_logger.info('{:6.2f}'.format(realClassifierLoss.item()))
        avg_kl_logger.info('{:6.2f}'.format(avg_kl.item()))
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

    	# use test set of real image to guide the learning of GAN
        #if steps % show_test_classifier_acc_per_iters == 0:
        #    test_acc, best_acc, best_epoch, best_per_class_acc = test_all(logger, epoch, best_acc, best_epoch, best_per_class_acc)
        #    print("Test acc {:6.2f} Best test acc {:6.2f} best epoch {:5d}".format(test_acc, best_acc, best_epoch))
        #    print('Epoch {:3d} Best Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(epoch, best_per_class_acc[0], best_per_class_acc[1], best_per_class_acc[2], best_per_class_acc[3], best_per_class_acc[4], best_per_class_acc[5], best_per_class_acc[6], best_per_class_acc[7], best_per_class_acc[8], best_per_class_acc[9], best_per_class_acc[10], best_per_class_acc[11], best_per_class_acc[12], best_per_class_acc[13], best_per_class_acc[14]))
        #    logger.info('Epoch {:3d} Best Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(epoch, best_per_class_acc[0], best_per_class_acc[1], best_per_class_acc[2], best_per_class_acc[3], best_per_class_acc[4], best_per_class_acc[5], best_per_class_acc[6], best_per_class_acc[7], best_per_class_acc[8], best_per_class_acc[9], best_per_class_acc[10], best_per_class_acc[11], best_per_class_acc[12], best_per_class_acc[13], best_per_class_acc[14]))
        #    netC.train()

            #netC.eval()
    		# sample some test images
            #with torch.no_grad():
            #    inputs, labels = sample_test_images_randomly()
            #    inputs = inputs.cuda()
            #    labels = labels.cuda()
            #    outputs = netC(inputs)
    	
    			# accuracy
            #    _, predicted = torch.max(outputs.data, 1)
            #    total_test += labels.size(0)
            #    correct_test += predicted.eq(labels.data).sum().item()
            #    test_accuracy = 100 * correct_test / total_test
            #    print(' test acc', test_accuracy)
            #    logger.info('                             Test Accuracy (classifier) is {:6.2f}'.format(test_accuracy))
            #    test_accuracy_logger.info('{:6.2f}'.format(test_accuracy))
            #    netC.train()
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
        #if (steps * batch_size) % save_per_samples == 0:
        #    model_file = weights_prefix + model_ckpt_prefix + 'bak_' + str(epoch + 1) + '_' + str((steps * batch_size) // save_per_samples) + '.pth'
        #    torch.save(state, model_file)

    # Test acc at the end of each epoch    
    test_acc, best_acc, best_epoch, best_per_class_acc = test_all(logger, epoch, best_acc, best_epoch, best_per_class_acc)
    print("Test acc {:6.2f} Best test acc {:6.2f} best epoch {:5d}".format(test_acc, best_acc, best_epoch))
    print('Epoch {:3d} Best Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(epoch, best_per_class_acc[0], best_per_class_acc[1], best_per_class_acc[2], best_per_class_acc[3], best_per_class_acc[4], best_per_class_acc[5], best_per_class_acc[6], best_per_class_acc[7], best_per_class_acc[8], best_per_class_acc[9], best_per_class_acc[10], best_per_class_acc[11], best_per_class_acc[12], best_per_class_acc[13], best_per_class_acc[14]))
    logger.info('Epoch {:3d} Best Per Class Acc {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f} {:6.2f}\n'.format(epoch, best_per_class_acc[0], best_per_class_acc[1], best_per_class_acc[2], best_per_class_acc[3], best_per_class_acc[4], best_per_class_acc[5], best_per_class_acc[6], best_per_class_acc[7], best_per_class_acc[8], best_per_class_acc[9], best_per_class_acc[10], best_per_class_acc[11], best_per_class_acc[12], best_per_class_acc[13], best_per_class_acc[14]))
    netC.train()
    
    # Save per-epoch weights
    model_file = weights_prefix + model_ckpt_prefix + weights_middle_name + str(epoch + 1) + '.pth'
    torch.save(state, model_file)
  return total_trained_samples
total_trained_samples = 0
torch.manual_seed(42)
resume_training = True
start_epoch = 0
if resume_training:
	start_epoch, total_trained_samples = load_model('../models_allclasses_wis_mmd/ecgan-chest-xray14_allclasses_is_mmd_epo_101.pth')
    #load_gan_and_vanilla('../models_apr28/ecgan-chest-xray14epo_98.pth', '../models_vanilla/ecgan-chest-xray14epo_99_.pth')  
    #
    #
    #
    
total_trained_samples = train(total_trained_samples)
 
            
  
      
      
