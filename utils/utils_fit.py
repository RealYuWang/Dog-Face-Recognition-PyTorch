import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


# 获取当前学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 训练一个 epoch
def fit_one_epoch(model_train, model, loss_history, loss, optimizer, epoch, epoch_step, epoch_step_val, train_loader,
                  val_loader,
                  Epoch, cuda, Batch_size):
    '''
    :param model_train: 训练模式的模型
    :param model: 模型
    :param loss_history: LossHistory类对象，用于记录损失历史并写入txt文件保持
    :param loss: TripletLoss 函数
    :param optimizer: 权重更新器
    :param epoch: 第几个epoch
    :param epoch_step: 训练集batch数
    :param epoch_step_val: val集batch数
    :param train_loader: 训练集dataloader
    :param val_loader: 测试集dataloader
    :param Epoch: 冻结训练时=IntervalEpoch，解冻时=总Epoch
    :param cuda: 是否使用gpu
    :param Batch_size: batch大小
    :return: 在val集上的平均损失
    '''
    total_triple_loss = 0  # 总Triplet Loss
    total_CE_loss = 0  # 总Cross Entropy Loss
    total_accuracy = 0  # 总准确率

    val_total_triple_loss = 0
    val_total_CE_loss = 0
    val_total_accuracy = 0
    # ------------------------
    # 训练阶段
    # ------------------------
    model_train.train()
    print('Start Training')
    with tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict,
              mininterval=0.3) as pbar:  # pbar: program bar 进度条
        for iteration, batch in enumerate(train_loader):
            if iteration >= epoch_step:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    labels = torch.from_numpy(labels).long().cuda()
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    labels = torch.from_numpy(labels).long()

            optimizer.zero_grad()
            before_normalize, outputs1 = model.forward_feature(images)
            outputs2 = model.forward_classifier(before_normalize)

            _triplet_loss = loss(outputs1, Batch_size)
            _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
            _loss = _triplet_loss + _CE_loss

            _loss.backward()
            optimizer.step()

            with torch.no_grad():
                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            total_triple_loss += _triplet_loss.item()
            total_CE_loss += _CE_loss.item()
            total_accuracy += accuracy.item()

            pbar.set_postfix(**{'total_triple_loss': total_triple_loss / (iteration + 1),
                                'total_CE_loss': total_CE_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Finish Train')
    # ------------------------
    # 验证阶段
    # ------------------------
    model_train.eval()
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(val_loader):
            if iteration >= epoch_step_val:
                break
            images, labels = batch
            with torch.no_grad():
                if cuda:
                    images = torch.from_numpy(images).type(torch.FloatTensor).cuda()
                    labels = torch.from_numpy(labels).long().cuda()
                else:
                    images = torch.from_numpy(images).type(torch.FloatTensor)
                    labels = torch.from_numpy(labels).long()

                optimizer.zero_grad()
                before_normalize, outputs1 = model.forward_feature(images)
                outputs2 = model.forward_classifier(before_normalize)

                _triplet_loss = loss(outputs1, Batch_size)
                _CE_loss = nn.NLLLoss()(F.log_softmax(outputs2, dim=-1), labels)
                _loss = _triplet_loss + _CE_loss

                accuracy = torch.mean(
                    (torch.argmax(F.softmax(outputs2, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

                val_total_triple_loss += _triplet_loss.item()
                val_total_CE_loss += _CE_loss.item()
                val_total_accuracy += accuracy.item()

            pbar.set_postfix(**{'val_total_triple_loss': val_total_triple_loss / (iteration + 1),
                                'val_total_CE_loss': val_total_CE_loss / (iteration + 1),
                                'val_accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)
    print('Finish Validation')
    # ------------------------
    # 记录loss数据
    # ------------------------
    loss_history.append_loss(total_accuracy / epoch_step, (total_triple_loss + total_CE_loss) / epoch_step,
                             (val_total_triple_loss + val_total_CE_loss) / epoch_step_val)

    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f' % ((total_triple_loss + total_CE_loss) / epoch_step))
    # 不存储所有模型
    if Epoch >= 75:
        torch.save(model.state_dict(), 'train_logs/Epoch%d-Total_Loss%.4f.pth-Val_Loss%.4f.pth'% ((epoch + 1),
                                   (total_triple_loss + total_CE_loss) / epoch_step,
                                   (val_total_triple_loss + val_total_CE_loss) / epoch_step_val))

    return (val_total_triple_loss + val_total_CE_loss) / epoch_step_val
