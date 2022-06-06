import os.path
import random
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.dataset_utils import letterbox_image


# 随机数生成，用于随机数据增强
def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


# DataLoader中collate_fn参数 将一个batch中的np数组类型的图像和标签拼接起来
# batchsize=64时，images (192, 3, 224, 224)
def dataset_collate(batch):
    images = []
    labels = []
    for img, label in batch:
        images.append(img)
        labels.append(label)

    images1 = np.array(images)[:, 0, :, :, :]
    images2 = np.array(images)[:, 1, :, :, :]
    images3 = np.array(images)[:, 2, :, :, :]
    images = np.concatenate([images1, images2, images3], 0)

    labels1 = np.array(labels)[:, 0]
    labels2 = np.array(labels)[:, 1]
    labels3 = np.array(labels)[:, 2]
    labels = np.concatenate([labels1, labels2, labels3], 0)
    return images, labels


class DogFaceDataset(Dataset):
    # input_shape (H, W, C) (224, 224, 3)
    def __init__(self, input_shape, dataset_path, num_train, num_classes):
        super(DogFaceDataset, self).__init__()
        self.dataset_path = dataset_path
        self.image_height = input_shape[0]
        self.image_width = input_shape[1]
        self.channel = input_shape[2]

        self.paths = []
        self.labels = []
        self.num_train = num_train

        self.num_classes = num_classes
        self.load_dataset()

    def __len__(self):
        return self.num_train

    # 从cls_train.txt中读取信息，获得路径和标签
    def load_dataset(self):
        for path in self.dataset_path:
            # cls_train.txt 中，;前为类别，后为路径
            path_split = path.split(";")
            self.paths.append(path_split[1].split()[0])
            self.labels.append(int(path_split[0]))
        self.paths = np.array(self.paths, dtype=np.object)
        self.labels = np.array(self.labels)

    # 随机给定一张图片途径，对图片进行预处理和增强 包括缩放、翻转、旋转和颜色调整
    def get_random_data(self, image, input_shape, jitter=0.1, hue=.05, sat=1.3, val=1.3, flip_signal=True):
        image = image.convert("RGB")

        h, w = input_shape
        rand_jit1 = rand(1 - jitter, 1 + jitter)
        rand_jit2 = rand(1 - jitter, 1 + jitter)
        new_ar = w / h * rand_jit1 / rand_jit2
        # 随机缩放
        scale = rand(0.9, 1.1)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)
        # 随机翻转
        flip = rand() < .5
        if flip and flip_signal:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        dx = int(rand(0, w - nw))
        dy = int(rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # 随机旋转
        rotate = rand() < .5
        if rotate:
            angle = np.random.randint(-5, 5)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])
        # 随机调整色调和饱和度
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
        val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue * 360
        x[..., 0][x[..., 0] > 1] -= 1
        x[..., 0][x[..., 0] < 0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255
        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")  # 从array转换成img
        return image_data

    def __getitem__(self, index):
        # images包含anchor positive negative  (N=3, C, H, W)
        images = np.zeros((3, self.channel, self.image_height, self.image_width))
        labels = np.zeros(3)
        # ------------------------------#
        #   先获得两张同一只狗的狗脸,作为anchor和positive
        #   随机选择一只狗，获取它的所有照片的路径
        # ------------------------------#
        c = random.randint(0, self.num_classes - 1)
        selected_path = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]
        # ------------------------------#
        #   从中随机选择两张
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)
        # 1st image
        image = Image.open(selected_path[image_indexes[0]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[0, 0, :, :] = image
        else:
            images[0, :, :, :] = image
        labels[0] = c
        # 2nd image
        image = Image.open(selected_path[image_indexes[1]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[1, 0, :, :] = image
        else:
            images[1, :, :, :] = image
        labels[1] = c
        # ------------------------------#
        #   取得一张negative作为对照
        # ------------------------------#
        different_c = list(range(self.num_classes))
        different_c.pop(c)  # 去掉已选择的狗
        different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
        current_c = different_c[different_c_index[0]]
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]
        # ------------------------------#
        #   随机选择一张
        # ------------------------------#
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        image = Image.open(selected_path[image_indexes[0]])
        image = self.get_random_data(image, [self.image_height, self.image_width])
        image = np.transpose(np.asarray(image).astype(np.float64), [2, 0, 1]) / 255
        if self.channel == 1:
            images[2, 0, :, :] = image
        else:
            images[2, :, :, :] = image
        labels[2] = current_c

        return images, labels

    # --------------
    # 用于可视化展示 返回三张Image类型图片
    # --------------
    def get_one_triplet(self):
        c = random.randint(0, self.num_classes - 1)
        selected_path = self.paths[self.labels[:] == c]
        while len(selected_path) < 2:
            c = random.randint(0, self.num_classes - 1)
            selected_path = self.paths[self.labels[:] == c]
        image_indexes = np.random.choice(range(0, len(selected_path)), 2)
        anchor = Image.open(selected_path[image_indexes[0]])
        positive = Image.open(selected_path[image_indexes[1]])

        different_c = list(range(self.num_classes))
        different_c.pop(c)  # 去掉已选择的狗
        different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
        current_c = different_c[different_c_index[0]]
        selected_path = self.paths[self.labels == current_c]
        while len(selected_path) < 1:
            different_c_index = np.random.choice(range(0, self.num_classes - 1), 1)
            current_c = different_c[different_c_index[0]]
            selected_path = self.paths[self.labels == current_c]
        image_indexes = np.random.choice(range(0, len(selected_path)), 1)
        negative = Image.open(selected_path[image_indexes[0]])
        return anchor, positive, negative


# ------------------------------------------
# 每个样本有两张图片。样本分为正样本、负样本两种。
# 正样本中使用同一只狗的照片，负样本不同狗。
# 同时返回一个is_same标识，用来区分正负样本
# ------------------------------------------
class EvalDataset(Dataset):
    def __init__(self, eval_set_path, pairs_path, image_size):
        '''
        :param eval_set_path: 验证数据集的路径
        :param pairs_path: 验证数据集标签txt的路径
        :param image_size: 图片尺寸
        '''
        super(EvalDataset, self).__init__()
        self.image_shape = image_size
        self.pairs_path = pairs_path
        self.samples_list = self.get_samples(eval_set_path)

    def get_random_pair(self):
        index = random.randint(0, len(self.samples_list) - 1)
        return self.samples_list[index]

    def get_samples(self, eval_set_path, file_ext='jpg'):
        # 正样本：pairs_list[i] = ['Name', '1', '4']  1表示为该狗第一张图片，4表示为第四张
        # 负样本：pairs_list[j] = ['Name_1', '1', 'Name_2', '2']
        pairs_list = []
        with open(self.pairs_path, 'r') as f:
            for line in f.readlines()[1:]:  # 从第二行开始读，第一行记录了fold数和每个fold的正负样本数量
                pair = line.strip().split()
                pairs_list.append(pair)

        samples_list = []  # 存储样本信息 该list的每一个元素皆为tuple，tuple中包含两张图片的路径和正负样本判别信号is_same
        for i in range(len(pairs_list)):
            pair = pairs_list[i]
            if len(pair) == 3:  # 正样本
                path_1st_dog = os.path.join(eval_set_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path_2nd_dog = os.path.join(eval_set_path, pair[0], pair[0] + '_' + '%04d' % int(pair[2]) + '.' + file_ext)
                is_same_dog = True
            elif len(pair) == 4:  # 负样本
                path_1st_dog = os.path.join(eval_set_path, pair[0], pair[0] + '_' + '%04d' % int(pair[1]) + '.' + file_ext)
                path_2nd_dog = os.path.join(eval_set_path, pair[2], pair[2] + '_' + '%04d' % int(pair[3]) + '.' + file_ext)
                is_same_dog = False
            if os.path.exists(path_1st_dog) and os.path.exists(path_2nd_dog):  # Only add the pair if both paths exist
                samples_list.append((path_1st_dog, path_2nd_dog, is_same_dog))

        return samples_list

    def __len__(self):
        return len(self.samples_list)

    def __getitem__(self, index):
        (path_1st_dog, path_2nd_dog, is_same_dog) = self.samples_list[index]
        # letterbox填充处理
        img_1st_dog, img_2nd_dog = Image.open(path_1st_dog), Image.open(path_2nd_dog)
        img_1st_dog = letterbox_image(img_1st_dog, [self.image_shape[1], self.image_shape[0]])
        img_2nd_dog = letterbox_image(img_2nd_dog, [self.image_shape[1], self.image_shape[0]])
        # 标准化处理
        img_1st_dog, img_2nd_dog = np.array(img_1st_dog) / 255, np.array(img_2nd_dog) / 255
        img_1st_dog = np.transpose(img_1st_dog, [2, 0, 1])
        img_2nd_dog = np.transpose(img_2nd_dog, [2, 0, 1])

        return img_1st_dog, img_2nd_dog, is_same_dog

