import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.facenet import FaceNet
from nets.facenet_training import LossHistory, triplet_loss, weights_init
from utils.datasets_definition import DogFaceDataset, dataset_collate
from utils.utils_fit import fit_one_epoch
from utils.file_utils import get_num_classes
import platform
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('train_logs/tensorboard')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == '__main__':
    Cuda = True
    # --------------------------------------------------------#
    #   cls_train.txt，读取狗脸路径与标签
    # --------------------------------------------------------#
    annotation_path = 'cls_train.txt'
    # --------------------------------------------------------#
    #   输入图像大小与通道
    # --------------------------------------------------------#
    input_shape = [160, 160, 3]
    backbone = "inception_resnetv1"
    pretrained = False
    model_path = "weights/for_training/backbone_inception_resnetv1.pth"
    num_workers = 4
    # ---------------------------------#
    #   不同狗的个数
    # ---------------------------------#
    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#
    model = FaceNet(backbone=backbone, num_classes=num_classes)
    # 如果不使用主干网络的预训练权重，以normal/kaiming/xavier方式初始化网络权重
    if not pretrained:
        weights_init(model, init_type='kaiming')
    # 如果使用预训练模型的所有权重，将其加载进来
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        # 解决键值不匹配的问题
        for key in list(pretrained_dict.keys()):
            pretrained_dict['backbone.model.' + key] = pretrained_dict.pop(key)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()
    # 如果使用GPU训练，移动至GPU，开启 cudnn benchmark
    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()
    # 损失函数
    loss = triplet_loss()
    # 损失记录
    loss_history = LossHistory("train_logs")

    # -------------------------------------------------------#
    #   0.05用于验证，0.95用于训练
    # -------------------------------------------------------#
    val_split = 0.05
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101), np.random.shuffle(lines), np.random.seed(None)
    num_val = int(len(lines) * val_split)  # 验证集图片数
    num_train = len(lines) - num_val  # 训练集图片数
    # ---------------------------------------------------------#
    #   迁移学习
    #   冻结训练阶段
    # ---------------------------------------------------------#
    if True:
        lr = 1e-3
        Batch_size = 16  # 256
        Init_Epoch = 0
        Interval_Epoch = 50

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DogFaceDataset(input_shape, lines[:num_train], num_train, num_classes)
        val_dataset = DogFaceDataset(input_shape, lines[num_train:], num_val, num_classes)

        train_loader = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                  drop_last=True, collate_fn=dataset_collate)
        val_loader = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)
        # 冻结
        for param in model.backbone.parameters():
            param.requires_grad = False
        # 正式开始训练
        for epoch in range(Init_Epoch, Interval_Epoch):
            total_loss = fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, train_loader,
                          val_loader, Interval_Epoch, Cuda, Batch_size)
            lr_scheduler.step()
            writer.add_scalar('Total Loss', total_loss, epoch)

    if True:
        # ----------------------------------------------------#
        #   解冻阶段训练参数
        # ----------------------------------------------------#
        lr = 1e-4
        Batch_size = 96
        Interval_Epoch = 50
        Epoch = 100

        epoch_step = num_train // Batch_size
        epoch_step_val = num_val // Batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.94)

        train_dataset = DogFaceDataset(input_shape, lines[:num_train], num_train, num_classes)
        val_dataset = DogFaceDataset(input_shape, lines[num_train:], num_val, num_classes)

        train_loader = DataLoader(train_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                  drop_last=True, collate_fn=dataset_collate)
        val_loader = DataLoader(val_dataset, batch_size=Batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate)

        # 解冻
        for param in model.backbone.parameters():
            param.requires_grad = True
        # 正式开始训练
        for epoch in range(Interval_Epoch, Epoch):
            total_loss = fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, train_loader,
                          val_loader, Epoch, Cuda, Batch_size)
            lr_scheduler.step()
            writer.add_scalar('Total Loss', total_loss, epoch)

    # 训练结束后自动打包结果并上传
    if platform.system().lower() == 'linux':
        os.system('shutdown')