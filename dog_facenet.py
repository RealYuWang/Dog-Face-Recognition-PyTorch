import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from nets.facenet import FaceNet


# ------------------------------------------------#
#    基于训练好的模型搭建识别/检测工具类
#    用于狗脸检测和狗脸识别 dnr.py
# ------------------------------------------------#

class DogFaceNet(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   训练好的狗脸识别模型的路径
        # --------------------------------------------------------------------------#
        "model_path": "weights/recognition/Epoch78.pth",
        # --------------------------------------------------------------------------#
        #   输入图片的大小
        # --------------------------------------------------------------------------#
        "input_shape": [160, 160, 3],
        # --------------------------------------------------------------------------#
        #   所使用到的主干特征提取网络 mobilenet/inception_resnetv1
        # --------------------------------------------------------------------------#
        "backbone": "mobilenet",
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化DogFaceNet
    #   通过使用__dict__.update()方法,配合 _defaults 字典批量构建类的属性
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)  # 对已存在的属性进行赋值
        self.generate()

    # ---------------------------------------------------#
    #   载入模型与权值
    # ---------------------------------------------------#
    def generate(self):
        print('Loading weights into state dict...')
        model = FaceNet(backbone=self.backbone, mode="predict")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def letterbox_image(self, image, size):
        if self.input_shape[-1] == 1:
            image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # ---------------------------------------------------#
    # 计算两张脸部图片的距离
    # 这里的image_1, image_2均为Image.open()过后得到的结果
    # ---------------------------------------------------#
    def get_faces_distance(self, image_1, image_2):
        # ---------------------------------------------------#
        #   图片预处理，归一化
        # ---------------------------------------------------#
        with torch.no_grad():
            image_1 = self.letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]])
            image_2 = self.letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]])

            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_1, np.float32) / 255, (2, 0, 1)), 0))
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(np.array(image_2, np.float32) / 255, (2, 0, 1)), 0))

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()
            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            output1 = self.net(photo_1).cpu().numpy()
            output2 = self.net(photo_2).cpu().numpy()
            # ---------------------------------------------------#
            #   计算二者之间的距离
            # ---------------------------------------------------#
            l1 = np.linalg.norm(output1 - output2, axis=1)
        # # matplotlib 画图，可视化结果
        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))
        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Distance:%.3f' % l1, ha='center', va='bottom', fontsize=11)
        # plt.show()

        return l1

    # ---------------------------------------------------#
    # 狗脸图片输入网络，获得128维numpy数组特征向量
    # image参数是Image.open(img_path)之后得到的输出
    # ---------------------------------------------------#
    def get_face_feature(self, image):
        with torch.no_grad():
            # 归一化预处理
            image = self.letterbox_image(image, [self.input_shape[1], self.input_shape[0]])
            photo = torch.from_numpy(np.expand_dims(np.transpose(np.array(image, np.float32) / 255, (2, 0, 1)), 0))
            if self.cuda:
                photo = photo.cuda()
            output = self.net(photo).cpu().numpy()
            return output
