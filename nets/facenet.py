import torch.nn as nn
import torch
from torch.nn import functional as F
from nets.mobilenet import MobileNetV1, MobileNetV2
from nets.inception_resnetv1 import InceptionResnetV1
from nets.shufflenet import shufflenet_v2_x1_0


class ShuffleNet(nn.Module):
    def __init__(self):
        super(ShuffleNet, self).__init__()
        self.model = shufflenet_v2_x1_0()

    def forward(self, x):
        x = self.model(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, version=1):
        super(MobileNet, self).__init__()
        self.version = version
        if self.version == 1:
            self.model = MobileNetV1()
            # 删除池化层和全连接层，下面转为输出128维特征向量
            del self.model.fc
            del self.model.avg
        elif self.version == 2:
            self.model = MobileNetV2()
            del self.model.avgpool
            del self.model.classifier

    def forward(self, x):
        if self.version == 1:
            x = self.model.stage1(x)
            x = self.model.stage2(x)
            x = self.model.stage3(x)  # 7, 7, 1024
        elif self.version == 2:
            x =self.model.features(x)
        return x


class InceptionResNet(nn.Module):
    def __init__(self):
        super(InceptionResNet, self).__init__()
        self.model = InceptionResnetV1()

    def forward(self, x):
        x = self.model.conv2d_1a(x)
        x = self.model.conv2d_2a(x)
        x = self.model.conv2d_2b(x)
        x = self.model.maxpool_3a(x)
        x = self.model.conv2d_3b(x)
        x = self.model.conv2d_4a(x)
        x = self.model.conv2d_4b(x)
        x = self.model.repeat_1(x)
        x = self.model.mixed_6a(x)
        x = self.model.repeat_2(x)
        x = self.model.mixed_7a(x)
        x = self.model.repeat_3(x)
        x = self.model.block8(x)
        # 无需avg层
        return x


class FaceNet(nn.Module):
    def __init__(self, backbone="mobilenet", dropout_keep_prob=0.5, embedding_size=128, num_classes=None, mode="train",
                 ):
        super(FaceNet, self).__init__()
        if backbone == "mobilenet":
            self.backbone = MobileNet(version=1)
            flat_shape = 1024
        elif backbone == 'mobilenetv2':
            self.backbone = MobileNet(version=2)
            flat_shape = 1280
        elif backbone == 'shufflenet':
            self.backbone = ShuffleNet()
            flat_shape = 1024
        elif backbone == "inception_resnetv1":
            self.backbone = InceptionResNet()
            flat_shape = 1792
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))  # 1, 1, 1024
        self.Dropout = nn.Dropout(1 - dropout_keep_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        x = self.last_bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward_feature(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)
        x = F.normalize(before_normalize, p=2, dim=1)
        return before_normalize, x

    def forward_classifier(self, x):
        x = self.classifier(x)
        return x