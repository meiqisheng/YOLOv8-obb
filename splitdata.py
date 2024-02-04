# 将图片和标注数据按比例切分为 训练集和测试集
import shutil
import random
import os
import argparse
 
 
# 检查文件夹是否存在
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
 
def main(image_dir, txt_dir, save_dir):
    # 创建文件夹
    mkdir(save_dir)
    images_dir = os.path.join(save_dir, 'images')
    labels_dir = os.path.join(save_dir, 'labels')
 
    img_train_path = os.path.join(images_dir, 'train')
    img_test_path = os.path.join(images_dir, 'test')
    img_val_path = os.path.join(images_dir, 'val')
 
    label_train_path = os.path.join(labels_dir, 'train')
    label_test_path = os.path.join(labels_dir, 'test')
    label_val_path = os.path.join(labels_dir, 'val')
 
    mkdir(images_dir)
    mkdir(labels_dir)
    mkdir(img_train_path)
    mkdir(img_test_path)
    mkdir(img_val_path)
    mkdir(label_train_path)
    mkdir(label_test_path)
    mkdir(label_val_path)
 
    # 数据集划分比例，训练集75%，验证集15%，测试集15%，按需修改
    train_percent = 0.85
    val_percent = 0.10
    test_percent = 0.05
 
    total_txt = os.listdir(txt_dir)
    num_txt = len(total_txt)
    list_all_txt = range(num_txt)  # 范围 range(0, num)
 
    num_train = int(num_txt * train_percent)
    num_val = int(num_txt * val_percent)
    num_test = num_txt - num_train - num_val
 
    train = random.sample(list_all_txt, num_train)
    # 在全部数据集中取出train
    val_test = [i for i in list_all_txt if not i in train]
    # 再从val_test取出num_val个元素，val_test剩下的元素就是test
    val = random.sample(val_test, num_val)
 
    print("训练集数目：{}, 验证集数目：{},测试集数目：{}".format(len(train), len(val), len(val_test) - len(val)))
    for i in list_all_txt:
        name = total_txt[i][:-4]
 
        srcImage = os.path.join(image_dir, name + '.jpg')
        srcLabel = os.path.join(txt_dir, name + '.txt')
 
        if i in train:
            dst_train_Image = os.path.join(img_train_path, name + '.jpg')
            dst_train_Label = os.path.join(label_train_path, name + '.txt')
            shutil.copyfile(srcImage, dst_train_Image)
            shutil.copyfile(srcLabel, dst_train_Label)
        elif i in val:
            dst_val_Image = os.path.join(img_val_path, name + '.jpg')
            dst_val_Label = os.path.join(label_val_path, name + '.txt')
            shutil.copyfile(srcImage, dst_val_Image)
            shutil.copyfile(srcLabel, dst_val_Label)
        else:
            dst_test_Image = os.path.join(img_test_path, name + '.jpg')
            dst_test_Label = os.path.join(label_test_path, name + '.txt')
            shutil.copyfile(srcImage, dst_test_Image)
            shutil.copyfile(srcLabel, dst_test_Label)
 
 
if __name__ == '__main__':
    """
    python split_datasets.py --image-dir my_datasets/color_rings/imgs --txt-dir my_datasets/color_rings/txts --save-dir my_datasets/color_rings/train_data
    """
    parser = argparse.ArgumentParser(description='split datasets to train,val,test params')
    parser.add_argument('--image-dir', type=str,default='D:/Backup/Documents/GitHub/表记指针数据集/1/train/images', help='image path dir')
    parser.add_argument('--txt-dir', type=str,default='D:/Backup/Documents/GitHub/表记指针数据集/1/train/labels' , help='txt path dir')
    parser.add_argument('--save-dir', default='D:/Backup/Documents/GitHub/表记指针数据集/1/meter', type=str, help='save dir')
    args = parser.parse_args()
    image_dir = args.image_dir
    txt_dir = args.txt_dir
    save_dir = args.save_dir
 
    main(image_dir, txt_dir, save_dir)