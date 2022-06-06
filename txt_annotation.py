# ------------------------------------------------#
#   进行训练前需要利用这个文件生成cls_train.txt
# ------------------------------------------------#
import os

if __name__ == "__main__":
    datasets_path = "data/training"
    types_name = os.listdir(datasets_path)  # 获取所有类别名，即文件夹名
    types_name = sorted(types_name)  # 排序

    list_file = open('cls_train.txt', 'w')  # 对应注解文件
    for cls_id, type_name in enumerate(types_name):
        photos_path = os.path.join(datasets_path, type_name)
        if not os.path.isdir(photos_path):
            continue
        photos_name = os.listdir(photos_path)
        for photo_name in photos_name:
            list_file.write(
                str(cls_id) + ";" + '%s' % (os.path.join(os.path.abspath(datasets_path), type_name, photo_name)))
            list_file.write('\n')
    list_file.close()
