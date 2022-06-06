import os
import cv2
from PIL import Image
import dlib
from imutils import face_utils
import math
import time
import random
import shutil


# ------------------------------------------------#
#   对数据集进行随机数据增强
# ------------------------------------------------#
def random_augment():
    path = '../detect_results'
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        print('dir processed now:', dir_path)
        img_list = os.listdir(dir_path)
        if len(img_list) > 6:
            print('dir has more than 6 images, skip')
            continue
        for img_file in img_list:
            # 获取图片的完整路径
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)
            new_name_prefix = img_path.split('.')[0]
            # 降低图片的饱和度
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 2] = img_hsv[:, :, 2] * round(random.uniform(0.6, 0.95), 1)
            img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            cv2.imwrite(new_name_prefix + '_saturation.jpg', img_hsv)
            # 水平镜像
            h_flip = cv2.flip(img, 1)
            cv2.imwrite(new_name_prefix + '_h_flip.jpg', h_flip)
            # 随机轻微旋转
            random_angle = random.randint(-7, 7)
            rows, cols = img.shape[:2]
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), random_angle, 1)
            dst_rotate = cv2.warpAffine(img, M, (cols, rows))
            cv2.imwrite(new_name_prefix + '_dst_random.jpg', dst_rotate)
            # 随机放大图片
            random_scale = random.uniform(1.0, 1.1)
            margin = (int(math.floor(160 * random_scale)) - 160) // 2
            crop_img = img[margin:160-margin, margin:160-margin]
            zoomed_img = cv2.resize(crop_img, (160, 160), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(new_name_prefix + '_zoomed.jpg', zoomed_img)


# ------------------------------------------------#
#   将长方形图片更改为正方形，直接resize会造成图片失真
#   letterbox方法使用灰色填充图片，使其变成正方形
# ------------------------------------------------#
def letterbox_image(image, size):
    image = image.convert("RGB")
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def cv2_letterbox_image(image, expected_size):
    ih, iw = image.shape[0:2]  # 输入图片的长度和宽度
    ew, eh = expected_size  # 输出图片的长宽 160, 160
    scale = min(eh / ih, ew / iw)
    nh = int(ih * scale)
    nw = int(iw * scale)
    image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
    top = (eh - nh) // 2
    bottom = eh - nh - top
    left = (ew - nw) // 2
    right = ew - nw - left
    new_img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return new_img


# ------------------------------------------------#
#   遍历原始数据集，检测狗脸、对齐、裁剪、保存
# ------------------------------------------------#
def detect_and_align_and_save_faces():
    expected_size = [160, 160]

    detector = dlib.cnn_face_detection_model_v1('../weights/dogHeadDetector.dat')
    predictor = dlib.shape_predictor('../weights/landmarkDetector.dat')

    source_path = 'F://datasets//test_200//'
    dir_list = os.listdir(source_path)

    start_time = time.time()
    for dir in dir_list:  # dir: 000001 - 001393
        print('******Directory processed now:', dir)
        dir_path = os.path.join(source_path, dir)  # F://datasets//after_4_bis//000001
        img_list = os.listdir(dir_path)
        for img_file in img_list:  # 001.jpg
            print('      image processed:', os.path.join(dir, img_file))
            img_path = os.path.join(dir_path, img_file)  # F://datasets//after_4_bis//000001\001.jpg
            img = cv2.imread(img_path)
            dets = detector(img, upsample_num_times=1)
            # 遍历检测到的狗脸框
            for i, d in enumerate(dets):
                final_resized_img = crop_and_align(predictor, d, img, expected_size)
                final_path = source_path + dir + '//' + img_file
                cv2.imwrite(final_path, final_resized_img)
                print('      processed image saved to ', final_path)
    end_time = time.time()
    print('Processed all directories in', round(end_time - start_time, 2), 'seconds')


# ------------------------------------------------#
# 图像处理函数
# ------------------------------------------------#
def rotate_to_align(img, margin, margin_x,  threshold):
    if margin != 0 and margin >= threshold:
        # 需旋转角度
        degree = round(math.degrees(math.atan(abs(margin) / margin_x)))
        rows, cols = img.shape[:2]
        if margin > 0:  # 顺时针
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -degree, 1)
        else:  # 逆时针
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), degree, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
    return img


def crop_and_align(predictor, det, img, expected_size):
    # 根据框进行裁剪获得狗脸图片
    x1 = det.rect.left() if det.rect.left() >= 0 else 0
    y1 = det.rect.top() if det.rect.top() >= 0 else 0
    x2 = det.rect.right() if det.rect.right() >= 0 else 0
    y2 = det.rect.bottom() if det.rect.bottom() >= 0 else 0

    crop_img = img[y1:y2, x1:x2]
    # 原图关键点检测
    shape = predictor(img, det.rect)
    shape = face_utils.shape_to_np(shape)  # np数组（6， 2）
    # 左眼坐标
    left_eye_x, left_eye_y = shape[5][0], shape[5][1]
    # 右眼坐标
    right_eye_x, right_eye_y = shape[2][0], shape[2][1]
    margin = left_eye_y - right_eye_y
    margin_x = right_eye_x - left_eye_x

    crop_img = rotate_to_align(crop_img, margin, margin_x, 10)
    final_resized_img = cv2_letterbox_image(crop_img, expected_size)
    return final_resized_img


def crop_without_detector(predictor, rect, img, expected_size):
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    width, height = x2 - x1, y2 - y1
    residual = abs(width - height) // 2
    if width > height:
        x1 += residual
        x2 -= residual
    else:
        y1 += residual
        y2 -= residual
    img = img[y1:y2, x1:x2]

    key_points = predictor(img, rect)
    key_points = face_utils.shape_to_np(key_points)  # np数组（6， 2）
    left_eye_x, left_eye_y = key_points[5][0], key_points[5][1]
    right_eye_x, right_eye_y = key_points[2][0], key_points[2][1]
    margin = left_eye_y - right_eye_y
    margin_x = right_eye_x - left_eye_x

    img = rotate_to_align(img, margin, margin_x, 10)
    final_resized_img = cv2_letterbox_image(img, expected_size)
    return final_resized_img


# ------------------------------------------------#
#   删除没有检测到狗脸的图片
# ------------------------------------------------#
def delete_224_img():
    num = 0
    path = 'F://datasets//test_200//'
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        img_list = os.listdir(dir_path)
        for img_file in img_list:
            img_path = os.path.join(dir_path, img_file)
            img = cv2.imread(img_path)
            if img.shape[0] == 224:
                num += 1
                os.remove(img_path)
                print('delete:', img_path)
    print('delete', num, 'images')


# ------------------------------------------------#
#   使用shutil库的rmtree函数删除只有一张图片的文件夹
# ------------------------------------------------#
def delete_one_img_folder():
    num = 0
    path = 'F://datasets//test_200'
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        if len(os.listdir(dir_path)) == 1:
            num += 1
            shutil.rmtree(dir_path)
            print('delete:', dir_path)
    print('delete', num, 'folders')


# ------------------------------------------------#
#   计算文件夹下图片数量
# ------------------------------------------------#
def count_img():
    num = 0
    path = 'F://datasets//pre_processed_faces//'
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        img_list = os.listdir(dir_path)
        num += len(img_list)
    print('total', num, 'images')


def mini_img(img):
    height, width = img.shape[0:2]
    baseline = max(height, width)
    if baseline > 250:
        img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    return img


if __name__ == '__main__':
    # random_augment()
    # detect_and_align_and_save_faces()
    # delete_224_img()
    # delete_empty_folder()
    # delete_zoomed_img()
    # count_img()
    random_augment()
