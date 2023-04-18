import timm
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.dropblock import DropBlock
from models.vit import ViT
from torchvision import models
import random

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) *
                                (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * \
                    feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate,
                                training=self.training, inplace=True)

        return out


class ResNet(nn.Module):

    def __init__(self, block, keep_prob=1.0, avg_pool=False, drop_rate=0.0, dropblock_size=5):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(
            block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(
            block, 160, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(
            block, 320, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(
            block, 640, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.linear1_1 = nn.Linear(2560, 512)
        self.linear1_2 = nn.Linear(512, 64)
        self.linear2_1 = nn.Linear(2560, 512)
        self.linear2_2 = nn.Linear(512, 64)
        self.linear3_1 = nn.Linear(2560, 512)
        self.linear3_2 = nn.Linear(512, 64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        # x1 = F.relu(self.linear1_1(x))
        # x1 = self.linear1_2(x1)
        # x2 = F.relu(self.linear2_1(x))
        # x2 = self.linear2_2(x2)
        # x3 = F.relu(self.linear3_1(x))
        # x3 = self.linear3_2(x3)
        # return [x1, x2, x3]
        return x
        # return x



class custom_model(nn.Module):

    def __init__(self,num_layer=5):
        super(custom_model, self).__init__()
        # self.classifier = timm.create_model('densenet121', pretrained=True)
        self.classifier = timm.create_model('tf_efficientnet_b7_ns', pretrained=True)
        # self.classifier = nn.Sequential(*list(classifier.children())[:-1])
        self.num_layer = num_layer
        # self.classifier = models.resnet34(pretrained=True, progress=True)
        # self.classifier = ViT(
        #     image_size = 32,
        #     patch_size = 3,
        #     dim = 512,
        #     depth = 6,
        #     heads = 8,
        #     mlp_dim = 1000,
        #     dropout = 0.1,
        #     emb_dropout = 0.1
        # )

        # self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.dropout = nn.Dropout(0.4)

        for i in range(num_layer):
            setattr(self, "linear%d_1" % i, nn.Linear(1000,512))
            setattr(self, "batch_norm%d_1" % i, nn.BatchNorm1d(num_features=512))
            setattr(self, "linear%d_2" % i, nn.Linear(512,64))
            # setattr(self, "linear%d_3" % i, nn.Linear(128,64))

        for m in self.modules():

            # if isinstance(m, nn.Conv2d):
            #     nn.init.kaiming_normal_(
            #         m.weight, mode='fan_out', nonlinearity='leaky_relu')
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.constant_(m.weight, 1)
            #     nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.Linear):

                # m.weight = nn.parameter.Parameter(torch.randn(m.out_features,m.in_features) * torch.sqrt(torch.tensor(2/m.in_features,requires_grad = True)))
                y = m.in_features
                y = random.randint(m.in_features/2,m.in_features)
                m.weight.data.normal_(0.0, 1/np.sqrt(y))
                # m.bias.data should be 0
                # m.bias.data.fill_(0)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.classifier(x)
        feat = []

        # print(self.classifier.classifier.weight)

        # feat = torch.zeros((self.num_layer,256))
        for i in range(self.num_layer):

            x1 = self.act(getattr(self,"batch_norm%d_1" % i)(getattr(self, "linear%d_1" % i)(x)))
            x2 = getattr(self, "linear%d_2" % i)(x1)
            feat.append(x2)
            # print(getattr(self, "linear%d_1" % i).weight)
            # weights.append(getattr(self, "linear%d_1" % i).weight)


        feat = torch.stack(feat ,dim = 0)
        # weights = torch.stack(weights,dim = 0)
        # print(feat.size())
        return feat




def conv_block(in_channels, out_channels):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ProtoNet(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return [x.view(x.size(0), -1)]


def resnet12(keep_prob=1.0, avg_pool=False,num_layer=5, **kwargs):
    """Constructs a ResNet-12 model.
    """
    # model = ResNet(BasicBlock, keep_prob=keep_prob,
    #                avg_pool=avg_pool, **kwargs)
    model = custom_model(num_layer=num_layer)
    # model = ProtoNet()

    return model

