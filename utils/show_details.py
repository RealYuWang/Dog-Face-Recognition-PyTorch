from utils.datasets_definition import DogFaceDataset, EvalDataset
import numpy as np
import matplotlib.pyplot as plt
import cv2
import dlib
import os
from imutils import face_utils
from sklearn.metrics import auc


def show_one_triplet():
    annotation_path = '../cls_train.txt'
    val_split = 0.05
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    input_shape = [160, 160, 3]
    val_set = DogFaceDataset(input_shape, lines[num_train:], num_val, 1101)
    anchor, positive, negative = val_set.get_one_triplet()

    plt.subplot(1, 3, 1), plt.imshow(anchor), plt.axis('off'), plt.title('anchor')
    plt.subplot(1, 3, 2), plt.imshow(positive), plt.axis('off'), plt.title('positive')
    plt.subplot(1, 3, 3), plt.imshow(negative), plt.axis('off'), plt.title('negative')
    plt.show()


# 根据yolo标注画出边界框
def show_bbox_yolo():
    label_path = 'F://datasets//Detection//Oxford_III//annotations//oxford_pug_106.txt'
    img_path = 'F://datasets//Detection//Oxford_III//images//oxford_pug_106.jpg'

    f = open(label_path, 'r+')
    img = cv2.imread(img_path)
    w = img.shape[1]
    h = img.shape[0]

    while True:
        line = f.readline()
        if line:
            img_tmp = img.copy()
            msg = line.split(' ')
            x1 = int((float(msg[1]) - float(msg[3]) / 2) * w)  # x_center - width/2
            y1 = int((float(msg[2]) - float(msg[4]) / 2) * h)  # y_center - height/2
            x2 = int((float(msg[1]) + float(msg[3]) / 2) * w)  # x_center + width/2
            y2 = int((float(msg[2]) + float(msg[4]) / 2) * h)  # y_center + height/2
            cv2.rectangle(img_tmp, (x1, y1), (x2, y2), (0, 0, 255), 5)
            cv2.imwrite('F://' + 'oxford_pug.jpg', img_tmp)
            cv2.imshow('show', img_tmp)
            cv2.waitKey(0)


# cv横向拼接三张图片,将三张图片的大小设置为500*500
def show_three_img():
    img_list = os.listdir('F://breeds')
    img_list.sort()
    img1 = cv2.imread('F://breeds//' + img_list[0])
    img2 = cv2.imread('F://breeds//' + img_list[1])
    img1 = cv2.resize(img1, (500, 500))
    img2 = cv2.resize(img2, (500, 500))
    dst = np.hstack((img1, img2))
    cv2.imwrite('F://' + 'all_img_5.jpg', dst)
    cv2.imshow('show', dst)
    cv2.waitKey(0)


def show_detected_face_and_landmarks():
    detector = dlib.cnn_face_detection_model_v1('../weights/dogHeadDetector.dat')
    predictor = dlib.shape_predictor('../weights/landmarkDetector.dat')

    img_path = "C://Users//danie//Desktop//lab.jpg"
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dets = detector(img, 1)
    img_result = img.copy()
    for i,d in enumerate(dets):
        x1, y1, x2, y2 = d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()
        cv2.rectangle(img_result, pt1=(x1, y1), pt2=(x2, y2), thickness=2, color=(255, 0, 0), lineType=cv2.LINE_AA)
        shape = predictor(img, d.rect)
        shape = face_utils.shape_to_np(shape)

        for i, p in enumerate(shape):
            cv2.circle(img_result, center=tuple(p), radius=3, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

    cv2.imwrite('../detected_face_and_landmarks.jpg', img_result)
    cv2.imshow('show', img_result)
    cv2.waitKey(0)


def plot_roc():
    epoch72_fpr = np.loadtxt('train_logs/epoch72_fpr.txt')
    epoch72_tpr = np.loadtxt('train_logs/epoch72_tpr.txt')
    epoch78_fpr = np.loadtxt('train_logs/epoch78_fpr.txt')
    epoch78_tpr = np.loadtxt('train_logs/epoch78_tpr.txt')
    mobile_best_fpr = np.loadtxt('train_logs/mobile_best_fpr.txt')
    mobile_best_tpr = np.loadtxt('train_logs/mobile_best_tpr.txt')
    resnet_fpr = np.loadtxt('train_logs/resnet_fpr.txt')
    resnet_tpr = np.loadtxt('train_logs/resnet_tpr.txt')

    auc_72 = auc(epoch72_fpr, epoch72_tpr)
    auc_78 = auc(epoch78_fpr, epoch78_tpr)
    auc_mobile = auc(mobile_best_fpr, mobile_best_tpr)
    auc_resnet = auc(resnet_fpr, resnet_tpr)

    fig = plt.figure(dpi=163)
    plt.plot(epoch72_fpr, epoch72_tpr, color='darkorange', lw=2, label='DogFaceNet (area = %0.2f)' % auc_72)
    plt.plot(epoch78_fpr, epoch78_tpr, color='green', lw=2, label='MobileNetV1 (area = %0.2f)' % auc_78)
    plt.plot(mobile_best_fpr, mobile_best_tpr, color='blue', lw=2, label='MobileNetV2 (area = %0.2f)' % auc_mobile)
    plt.plot(resnet_fpr, resnet_tpr, color='black', lw=2, label='Inception-ResNet-v1 (area = %0.2f)' % auc_resnet)
    plt.legend(loc="lower right")
    plt.xlim([-0.02, 1.0])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')

    fig.savefig('roc.png')


if __name__ == '__main__':
    pass