from PIL import Image
from dog_facenet import DogFaceNet
import os
import numpy as np
from db.mysql_helper import MySqlHelper
# ------------------------------------------------#
#   存储狗脸特征向量到数据库中
#   获得数据库信息，用于狗脸比对和识别
# ------------------------------------------------
db = MySqlHelper()


# ------------------------------------------------#
#    特征向量转化为字符串数组
# ------------------------------------------------#
def change_feature_vector_to_str_list(feature):
    encoding_list = feature.tolist()
    encoding_str_list = [str(i) for i in encoding_list]
    encoding_str = ','.join(encoding_str_list)
    return encoding_str


# ------------------------------------------------#
#    计算指定图片的特征向量并编码后存入数据库
# ------------------------------------------------#
def save_one_feature(img_path, name, model):
    image = Image.open(img_path)
    feature = model.get_face_feature(image)
    encoding_str = change_feature_vector_to_str_list(feature)

    sql = 'insert into face_2(name, encoding, path) values(%s, %s, %s)'
    ret = db.insert_one(sql, (name, encoding_str, img_path))
    if ret == 0:
        print('Insertion failed, please try again!')
    else:
        print(img_path + '  Insertion succeeded!')


def save_one_feature_name_needed(img_path, model, name, original_image_path):
    image = Image.open(img_path)
    feature = model.get_face_feature(image)
    encoding_str = change_feature_vector_to_str_list(feature)

    sql = 'insert into face_2(name, encoding, path) values(%s, %s, %s)'
    ret = db.insert_one(sql, (name, encoding_str, original_image_path))
    return ret


# ------------------------------------------------#
#    计算指定文件夹下所有图片的特征向量并编码后存入数据库
# ------------------------------------------------#
def save_all_features(model, dir_path, mode='have_sub_dir'):
    # model = DogFaceNet()
    dir_list = os.listdir(dir_path) # 获取文件夹下所有文件名
    if mode == 'have_sub_dir':
        for subdir in dir_list:
            subdir_path = os.path.join(dir_path, subdir)
            img_list = os.listdir(subdir_path)
            for img in img_list:
                img_path = os.path.join(subdir_path, img)
                save_one_feature(img_path, subdir, model)
    else:
        for img in dir_list:
            path = os.path.join(dir_path, img)
            save_one_feature(path, model)


# ------------------------------------------------#
#    获取狗脸信息
# ------------------------------------------------#
def get_all_features(return_type='np', table='face_2'):
    ids = []
    names = []
    encodings = []
    paths = []
    if table == 'face_2':
        sql = 'select * from face_2'
    elif table == 'face_single':
        sql = 'select * from face_single'
    # sql = 'select * from face_single'
    res = db.select_all(sql)
    for row in res:
        # 获取各属性
        id = row[0]
        name = bytes.decode(row[1])
        encoding = bytes.decode(row[2])
        path = bytes.decode(row[3])
        # 将encoding转换为浮点np数组
        data_list = encoding.strip('[').strip(']').split(',')
        float_list = list(map(float, data_list))
        encoding_arr = np.array(float_list)

        ids.append(id)
        names.append(name)
        encodings.append(encoding_arr)
        paths.append(path)
    # 若返回类型为numpy数组
    if return_type == 'np':
        return np.array(ids), np.array(names), np.array(encodings), np.array(paths)
    # 否则返回list类型
    else:
        return ids, names, encodings, paths


# ------------------------------------------------#
#    从dog_owner表获取狗主信息
# ------------------------------------------------#
def get_owner_info(dog_id):
    sql = 'select name, address, tel, email from dog_owner where id = %s'
    res = db.select_one(sql, dog_id)

    owner_name = bytes.decode(res[0])
    address = bytes.decode(res[1])
    tel = bytes.decode(res[2])
    email = bytes.decode(res[3])
    return owner_name, address, tel, email


def insert_owner_infos():
    ids, _, _, _ = get_all_features(return_type='list')
    for id in ids:
        sql = 'insert into dog_owner(id, name, address, tel, email) values(%s, %s, %s, %s, %s)'
        ret = db.insert_one(sql, (id, 'Alex', 'No. 1 Weigang', '18888888888', 'imosino1@gmail.com'))
        if ret == 1:
            print('insert success')
        else:
            print('insert failed')


def insert_one_owner_info(id, name, address, tel, email):
    sql = 'insert into dog_owner(id, name, address, tel, email) values(%s, %s, %s, %s, %s)'
    ret = db.insert_one(sql, (id, name, address, tel, email))
    return ret


def get_id_by_path(path):
    sql = 'select id from face_2 where path = %s'
    ret = db.select_one(sql, path)
    return ret[0]


if __name__ == '__main__':
    # sql_1 = 'insert into face_2(name, encoding, path) values(%s, %s, %s)'
    # ret = db.insert_one(sql_1, ("Test", "Test", "Test"))
    id = get_id_by_path("F:/datasets/Previous_Version_Recognition_Set/test_292_dogs/Wang/WangCai_ww6.jpg")
    print(id)






