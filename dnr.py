import os
from PIL import Image
from dog_facenet import DogFaceNet
from db import save_and_get as db
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch
from colorama import Fore, Back, Style
from utils.dataset_utils import crop_without_detector
import dlib
import cv2
import time
from utils.datasets_definition import EvalDataset


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# ------------------------------------------------#
#    载入检测和识别模型
# ------------------------------------------------#
print(Back.LIGHTCYAN_EX + '************  Loading face detection model, please wait ...  ************')
detect_model = torch.hub.load('yolov5', 'custom', path='weights/detection/best_S.pt', source='local')
predictor = dlib.shape_predictor('weights/landmarkDetector.dat')  # 关键点检测
print('************  Face detection model loaded!                   ************\n')

print(Back.LIGHTYELLOW_EX + '$$$$$$$$$$$$  Loading face recognition model, please wait ...  $$$$$$$$$$$$')
model = DogFaceNet()
print('$$$$$$$$$$$$  Face recognition model loaded!                   $$$$$$$$$$$$')
# ------------------------------------------------#
#    读取数据库信息，均为numpy数组
# ------------------------------------------------#
ids, names, features, paths = db.get_all_features()
print(Fore.BLUE + '\nDatabase Information loaded!')
print(Style.RESET_ALL)


# ------------------------------------------------#
#    给定两张图片，计算二者间的距离，判断是否为同一只狗
# ------------------------------------------------#
def is_same_dog(img_path_1, img_path_2, threshold=1.15):
    image_1 = Image.open(img_path_1)
    image_2 = Image.open(img_path_2)
    # 二者间距离
    distance = model.get_faces_distance(image_1, image_2)
    print('Distance of these two images: ', distance)
    if distance <= threshold:
        return True
    else:
        return False


def show_eval_pairs(nums=100):
    dataset = EvalDataset('F://datasets//test_200', 'eval_pairs.txt', [160, 160, 3])
    for i in range(nums):
        (path_1st_dog, path_2nd_dog, is_same) = dataset.get_random_pair()
        verification_res = is_same_dog(path_1st_dog, path_2nd_dog)
        if verification_res == is_same:
            continue
        elif verification_res is True and is_same is False:
            print('False positive: ', path_1st_dog, path_2nd_dog)
        elif verification_res is False and is_same is True:
            print('False negative: ', path_1st_dog, path_2nd_dog)


# ------------------------------------------------#
#    显示原图片和检测到的k张最近图片
# ------------------------------------------------#
def plot(k, original_img_path, closest_paths):
    try:
        ori_img = Image.open(original_img_path)
        closest_imgs_list = []
        for path in closest_paths:
            closest_imgs_list.append(Image.open(path))
        idxs = np.arange(k + 1, 2 * k + 1, 1)
        plt.subplot(2, 1, 1)
        plt.imshow(np.array(ori_img))
        for idx in idxs:
            plt.subplot(2, k, idx)
            plt.imshow(np.array(closest_imgs_list[idx - k - 1]))
        plt.show()
    except Exception as e:
        print(e)


# ------------------------------------------------#
#    狗脸识别测试 两个模式：single 和 test
#    single：给定一张狗脸图片和k值，可视化结果并根据KNN给出最终识别结果
#    test: 不展示图片，遍历测试集中每一张狗脸获得识别结果，计算总体准确率
# ------------------------------------------------#
def recognize_test(k=3, mode='single', single_img_path=None, test_set_path=None):
    if mode == 'single':
        assert os.path.exists(single_img_path), 'Image does not exist!'
        image = Image.open(single_img_path)
        start = time.time()
        image_feature = model.get_face_feature(image)
        end = time.time()
        print('Time cost: ', end - start)
        ND_feature = np.resize(image_feature, (1, 128))  # (1, 128)  (N, D)
        # dists 存储该图片与数据库内每一张图片的距离  shape:(1, 615)
        dists = np.linalg.norm(ND_feature - features, axis=1, keepdims=True).T
        # 获取最短距离 判断是否在数据库中
        min_dist = np.min(dists[0])
        if min_dist >= 0.85:
            print('The dog is probably not in our database, please unload some pictures of him/her and try later.')
            return False
        # 获取最短的k个下标
        idxs = np.argsort(dists[0])[:k]
        closest_names = names[idxs]
        result_dict = dict(Counter(closest_names))  # 统计列表中各元素出现次数
        final_name = list(result_dict.keys())[list(result_dict.values()).index(max(result_dict.values()))]
        print('Recognition Result: ', final_name)
    if mode == 'test':
        print('Reading files in ', test_set_path, '......')
        img_list = os.listdir(test_set_path)
        total = len(img_list)
        print('Total test images: ', total)
        # 初始ND距离表
        ND_feature = np.zeros((1, 128))
        start = time.time()
        for img in img_list:
            image = Image.open(os.path.join(test_set_path, img))
            img_feature = model.get_face_feature(image)
            ND_feature = np.concatenate((ND_feature, img_feature), axis=0)
        end = time.time()
        print('Cost per image: {}'.format((end - start) / total))
        # 删除第一行
        ND_feature = np.delete(ND_feature, 0, axis=0)  # (184, 128)   features: (615, 128)
        # dists 存储测试图片图片与数据库内每一张图片的距离 (184, 615)
        dists = np.sqrt(
            np.sum(ND_feature ** 2, axis=1, keepdims=True) + np.sum(features ** 2, axis=1) - 2 * ND_feature.dot(
                features.T))

        correct_1 = 0
        correct_k = 0
        for i in range(total):
            idxs = np.argsort(dists[i])[:k]
            closest_names = names[idxs]
            # 获取当前测试图片的狗的名字，判断其是否在 closest_names 中
            dog_name = img_list[i].split('.')[0]
            # Rank 1 Accuracy
            if dog_name == closest_names[0]:
                correct_1 += 1
            # Rank K Accuracy
            if dog_name in closest_names:
                correct_k += 1
        print('Rank 1 Accuracy: {:.2%}'.format(correct_1 / total))
        print('Rank {} Accuracy: {:.2%}'.format(k, correct_k / total))


# ------------------------------------------------#
#   狗脸检测测试
#   给定一张任意分辨率图片，若检测到狗脸则裁剪并保存狗脸
#   recognize 为 True 时识别检测到的狗脸
# ------------------------------------------------#
def detect_test(recognize=False):
    while True:
        img_path = input('Input path of the image(enter s to stop): ')
        if img_path == 's':
            break
        assert os.path.exists(img_path), 'File not found!'

        res = detect_model(img_path)
        pred = res.get_pred()
        pred_list = pred[0].cpu().numpy().tolist()
        if len(pred_list) == 0:
            print(Fore.RED + 'No faces detected, please upload another image.' + Fore.RESET)
        else:
            print('    {} faces detected, ready to save ......'.format(len(pred_list)))
            # crop 为字典数组，[{'box', 'conf', 'cls', 'label', 'im', 'file'}]
            crop = res.crop(save=False)
            cropped_paths_list = []
            for index in range(len(crop)):
                cropped_image = crop[index]['im']
                rect = dlib.rectangle(0, 0, cropped_image.shape[1], cropped_image.shape[0])
                final_resized_image = crop_without_detector(predictor, rect, cropped_image, [160, 160])
                save_path = os.path.join('detect_results', 'cropped_' + str(index + 1) + '_' + os.path.basename(img_path))
                cropped_paths_list.append(save_path)
                cv2.imwrite(save_path, final_resized_image)
            if recognize:
                if len(crop) > 1:
                    choice = input(
                        '{} faces detected, input index of the face you want to recognize[1, {}]): '.format(len(crop),
                                                                                                            len(crop)))
                    recognize_test(k=3, mode='single', single_img_path=cropped_paths_list[int(choice) - 1])


# ------------------------------------------------#
#   供 GUI 使用的接口
# ------------------------------------------------#
# 返回裁剪出的图片
def pure_detect(img_path):
    res = detect_model(img_path)
    pred = res.get_pred()
    pred_list = pred[0].cpu().numpy().tolist()
    if len(pred_list) == 0:
        return None
    else:
        crop = res.crop(save=False)
        cropped_image = crop[0]['im']
        rect = dlib.rectangle(0, 0, cropped_image.shape[1], cropped_image.shape[0])
        final_resized_image = crop_without_detector(predictor, rect, cropped_image, [160, 160])
        return final_resized_image


# 识别并返回确定的名字和相似图片路径
def pure_recognize(cropped_img_path, k=3):
    image = Image.open(cropped_img_path)
    image_feature = model.get_face_feature(image)
    ND_feature = np.resize(image_feature, (1, 128))
    dists = np.linalg.norm(ND_feature - features, axis=1, keepdims=True).T
    min_dist = np.min(dists[0])
    if min_dist >= 0.85:
        return None
    idxs = np.argsort(dists[0])[:k]
    closest_names = names[idxs]
    closest_ids = ids[idxs]
    closest_paths = paths[idxs]
    # result_dict = dict(Counter(closest_names))
    # K近邻中，名字出现最多的是谁
    # final_name = list(result_dict.keys())[list(result_dict.values()).index(max(result_dict.values()))]
    final_name = closest_names[0]
    dog_id = closest_ids[0]
    return {
        'dog_id': dog_id,
        'final_name': final_name,
        'closest_paths': closest_paths
    }


def get_model():
    return model


if __name__ == "__main__":
    # detect_test(recognize=False)
    # recognize_test(k=5, mode='test', test_set_path='data/testing')
    # db.save_all_features(model, 'F://datasets//test_200')
    # db.save_all_features(model, 'F://datasets//train_200')
    recognize_test(k=5, mode='single', single_img_path='data/testing/Adagio.jpg')
    # show_eval_pairs()
