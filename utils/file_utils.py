import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import shutil


# ------------------------------------------------#
#   获取训练集狗的个数，利用交叉熵损失辅助收敛
# ------------------------------------------------#
def get_num_classes(annotation_path):
    with open(annotation_path) as f:
        dataset_path = f.readlines()
    labels = []
    for path in dataset_path:
        path_split = path.split(";")
        labels.append(int(path_split[0]))
    num_classes = np.max(labels) + 1
    return num_classes


# ------------------------------------------------#
#   该函数批量修改一个文件夹下所有子文件/子文件夹的名称
# ------------------------------------------------#
def change_dir_name(dir_path):
    dir_list = os.listdir(dir_path)
    num = 1
    for dir in dir_list:
        old_name = os.path.join(dir_path, dir)
        new_name = os.path.join(dir_path, '%06d' % num)
        os.rename(old_name, new_name)
        num += 1


# ------------------------------------------------#
#   修改一个文件夹下所有子文件夹中的文件的名字
# ------------------------------------------------#
def change_dir_file_name(father_dir_path):
    dir_list = os.listdir(father_dir_path)
    for dir in dir_list:
        dir_path = os.path.join(father_dir_path, dir)
        print('Directory processed now: ' + dir_path)
        file_list = os.listdir(dir_path)

        num = 1
        for file in file_list:
            old_name = os.path.join(dir_path, file)
            # new_name = os.path.join(dir_path, '%04d' % num + '.jpg')
            new_name = os.path.join(dir_path, dir + '_' + '%04d' % num + '.jpg')
            print(new_name)
            os.rename(old_name, new_name)
            num += 1


# ------------------------------------------------#
#   删除包含n张图片的文件夹
# ------------------------------------------------#
def delete_empty_folder(path, n):
    num = 0
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        if len(os.listdir(dir_path)) == n:
            num += 1
            shutil.rmtree(dir_path)
            print('delete:', dir_path)
    print('delete', num, 'folders')


# ------------------------------------------------#
#   使用狗脸检测模型批量检测图片并生成YOLO标记文件
# ------------------------------------------------#
def image_to_yolo(detect_model):
    img_path = 'example_imgs/many_faces.jpg'
    img_name = img_path.split('/')[-1]
    label_txt_name = img_name.split('.')[0] + '.txt'
    print(label_txt_name)
    print('name of the image: ', img_name)

    res = detect_model(img_path)

    xywhn = res.get_xywhn()
    detected_list = xywhn[0].cpu().numpy().tolist()

    for single_res_list in detected_list:
        x = round(single_res_list[0], 6)
        y = round(single_res_list[1], 6)
        w = round(single_res_list[2], 6)
        h = round(single_res_list[3], 6)
        cls = int(single_res_list[4])
        anno = str(cls) + ' ' + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h) + '\n'
        with open(label_txt_name, 'a') as f:
            f.write(anno)


# ------------------------------------------------#
#   将VOC格式标注转换为YOLO格式标注
# ------------------------------------------------#
def xml_reader(xml_file_path):
    tree = ET.parse(xml_file_path)
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = {}
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [int(bbox.find('xmin').text),
                            int(bbox.find('ymin').text),
                            int(bbox.find('xmax').text),
                            int(bbox.find('ymax').text)]
        objects.append(obj_dict)
    return width, height, objects


def voc_to_yolo():
    xml_dir_path = 'F://datasets//Detection//Oxford_III//annotations//'
    voc_xml_list = os.listdir(xml_dir_path)
    for xml_file in voc_xml_list:
        file_path = xml_dir_path + xml_file
        print('xml file processed now: ', xml_file)
        width, height, objects = xml_reader(file_path)

        lines = []
        for obj in objects:
            x, y, x2, y2 = obj['bbox']
            label = 0
            cx = (x2 + x) * 0.5 / width
            cy = (y2 + y) * 0.5 / height
            w = (x2 - x) * 1. / width
            h = (y2 - y) * 1. / height
            line = "%s %.6f %.6f %.6f %.6f\n" % (label, cx, cy, w, h)
            lines.append(line)

        txt_name = file_path.replace('.xml', '.txt').replace('annotations', 'labels_yolo_format')
        with open(txt_name, 'w') as f:
            f.writelines(lines)


# ------------------------------------------------#
#   生成评估模型所需的评估集的 eval_pairs.txt 文件
# ------------------------------------------------#
def generate_eval_pair_txt(dir_path):
    dog_list = os.listdir(dir_path)

    dog_nums = len(dog_list)
    fold_num = 10
    sample_per_fold = 50

    for fold in range(fold_num):
        # 正样本
        rand_idxes = random.sample(range(0, dog_nums), sample_per_fold)
        # 随机选择50只狗 每只狗选两张
        for index in rand_idxes:
            dog_name = dog_list[index]  # 获取狗名，即文件夹名
            chosen_dog_path = os.path.join(dir_path, dog_name)
            img_list = os.listdir(chosen_dog_path)
            img_num = len(img_list)  # 图片数量
            if img_num == 2:
                with open('../eval_pairs.txt', 'a') as f:
                    f.write(dog_name + '    ' + '1' + '    ' + '2\n')
            else:
                rand_two = random.sample(range(1, img_num + 1), 2)
                with open('../eval_pairs.txt', 'a') as f:
                    f.write(dog_name + '    ' + str(rand_two[0]) + '    ' + str(rand_two[1]) + '\n')
        # 负样本
        rand_idxes = random.sample(range(1, dog_nums), sample_per_fold)
        for index in rand_idxes:
            first_dog_name = dog_list[index]
            another_rand_index = random.randint(0, dog_nums - 1)  # [0, dog_nums-1]
            while another_rand_index == index:  # another_rand_index与index不同
                another_rand_index = random.randint(0, dog_nums - 1)
            second_dog_name = dog_list[another_rand_index]

            list_1 = os.listdir(os.path.join(dir_path, first_dog_name))
            first_dog_pic = random.randint(1, len(list_1))
            list_2 = os.listdir(os.path.join(dir_path, second_dog_name))
            second_dog_pic = random.randint(1, len(list_2))
            with open('../eval_pairs.txt', 'a') as f:
                f.write(first_dog_name + '    ' + str(first_dog_pic) + '    ' + second_dog_name + '    ' + str(
                    second_dog_pic) + '\n')


# ------------------------------------------------#
#   读取csv文件生成狗名列表
# ------------------------------------------------#
def get_dog_names_list():
    csv_file = 'NYC_Dog_Names.csv'
    df = pd.read_csv(csv_file)
    names = df['Name'].tolist()
    names_list = []
    for name in names:
        if name.isalpha() and len(name) >= 3:
            name = name.lower()
            name = name[0].upper() + name[1:]
            names_list.append(name)
    return names_list


# ------------------------------------------------#
#   文件夹改狗名
# ------------------------------------------------#
def change_dir_name_with_dog_name(path):
    dog_names_list = get_dog_names_list()
    dir_list = os.listdir(path)
    dir_num = len(dir_list)
    random_index = random.sample(range(0, len(dog_names_list)), dir_num)
    random_names = [dog_names_list[i] for i in random_index]
    print(random_names)

    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        os.rename(dir_path, os.path.join(path, random_names[dir_list.index(dir)]))


# ------------------------------------------------#
#    从测试集中的每一只狗的多张图片中选取一张照片作为测试图片
# ------------------------------------------------#
def choose_one_image_from_each_folder(path):
    dir_list = os.listdir(path)
    for dir in dir_list:
        dir_path = os.path.join(path, dir)
        img_list = os.listdir(dir_path)
        img_num = len(img_list)
        if img_num == 0:
            print(dir_path)
        else:
            img_path = os.path.join(dir_path, img_list[0])
            shutil.move(img_path, 'F://datasets//test_200_choose')


# ------------------------------------------------#
#    删除狗名中的下划线和数字
# ------------------------------------------------#
def delete_underscore_and_numbers_in_the_names_of_the_images(dir_path):
    img_list = os.listdir(dir_path)
    for img in img_list:
        img_path = os.path.join(dir_path, img)
        img_name = img.split('.')[0]
        new_img_name = ''
        for c in img_name:
            if c.isalpha():
                new_img_name += c
        new_img_name += '.jpg'
        os.rename(img_path, os.path.join(dir_path, new_img_name))


if __name__ == '__main__':
    # delete_empty_folder('F://datasets//test_200', n=1)
    # choose_one_image_from_each_folder('F://datasets//test_200')
    # change_dir_file_name('F://datasets//test_200')
    generate_eval_pair_txt('F://datasets//test_200')
