import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
import shutil
import json
import random
"""
为主函数提供几个功能接口，以供调用
1.build_xml2voc 
2.voc2yolo
3.voc2coco
4.check_xml
"""


# 检查.xml文件和.jpg文件是否对应
# path是异常文件的保存路径，file_groups是一个字典。键为后缀名，值为一个列表，存放对应的文件
def check_xml(path, file_groups):
    # 如果文件存在，先删了
    if os.path.exists(path):
        shutil.rmtree(path)
    print('Checking...... The abnormal files will be stored in:', path)
    xml_set = set()
    jpg_set = set()
    for xml in file_groups.get('xml'):
        xml_set.add(xml.split('.')[0])
    for jpg in file_groups.get('jpg'):
        jpg_set.add(jpg.split('.')[0])

    # 对比，如果非空，创建文件夹，移动文件
    diff_xml = xml_set - jpg_set  # xml多
    diff_jpg = jpg_set - xml_set
    while len(diff_xml) == 0 and len(diff_jpg) == 0:
        return print("xml与jpg文件完全对应，未进行任何操作！")

    if len(diff_xml) != 0:
        os.makedirs(path+'/xml')
        for xml in diff_xml:
            shutil.move(xml+'.xml', path+'/xml')
    if len(diff_jpg) != 0:          # 注意不能用elif
        os.makedirs(path + '/jpg')
        for jpg in diff_jpg:
            shutil.move(jpg + '.jpg', path + '/jpg')
    return print('共移动', len(diff_xml)+len(diff_jpg), '个文件')


# 创建标准格式的VOC2012数据集，
# 传入参数为：名字列表，训练样本的比率。
def xml2voc(names, ratio):
    while (ratio <= 0) or (ratio >= 1):
        return print("Error! The ratio must be a float between 0 and 1, please input again!")
    else:
        print('VOC2012 Building...... , Train ratio is:', ratio)
        # 在当前路径创建VOC2012，用于存放生成3个子文件夹
        os.mkdir('./voc2012')
        file_name = ['./JPEGImages', './Annotations', './ImageSets']
        for name in file_name:
            os.mkdir('./voc2012' + name)
        # 复制path路径中的.xml和.jpg文件，分别进入Annotations文件夹和JPEGImages文件夹
        for file in names['xml']:
            # print(file)
            shutil.copy2(file, './voc2012/Annotations')
        for img in names['jpg']:
            # print(img)
            shutil.copy2(img, './voc2012/JPEGImages')

        # 编辑ImageSets/Main
        # 打乱顺序并创建新的名字列表(去掉路径与后缀)
        # 按给定比例分为train，test，val
        random.shuffle(names['jpg'])
        new_namelist = []
        for image in names['jpg']:
            image = (image.split('\\')[-1]).split('.')[0]
            new_namelist.append(image)
            # print(image)
        train_num = int(len(new_namelist) * ratio)
        print('train img number is:', train_num)
        train_imgs = new_namelist[:train_num]
        test_imgs = new_namelist[train_num:]
        # print(train_imgs)

        # 编辑ImageSets完成分组
        # 将文件名存储到 train.txt 文件
        os.makedirs('./voc2012/ImageSets/Main')
        with open('./voc2012/ImageSets/Main/train.txt', 'w') as train_file:
            for name in train_imgs:
                train_file.write(name + '\n')

        # 将文件名存储到 test.txt 文件
        with open('./voc2012/ImageSets/Main/test.txt', 'w') as test_file:
            for name in test_imgs:
                test_file.write(name + '\n')

        # 将文件名存储到 val.txt 文件
        with open('./voc2012/ImageSets/Main/val.txt', 'w') as test_file:
            for name in test_imgs:
                test_file.write(name + '\n')
    return print('finish!')


def convert_sb(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return x, y, w, h


# 此函数被voc2yolo调用
def read_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().splitlines()
        # print(content)
    return content


# 此函数被voc2yolo调用
def covert_xml2txt(xml, classes, path):
    '''
    :param xml: one xml file path
    :return: make one txt file and return
    '''
    # print(classes)
    txt = open(os.path.join(path, xml.split('s/')[-1].split('.')[0] + '.txt'), 'w')
    tree = ET.parse(xml)
    root = tree.getroot()

    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls not in classes:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert_sb((w, h), b)
        txt.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
    return


# 1.从voc2012/ImageSets/Main中读取3个txt文件，保存到3个列表.
# 2.按列表提取voc2012/JPEGImages中的图片，保存到path/images/的三个文件夹中.
# 3.按列表提取voc2012/Annotations/中的xml文件，将这些xml文件转化为txt文件。
# 4.保存txt文件到path/labels/
def voc2yolo(path, classes):
    # 如果文件存在，先删了
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path+'/images/test')
    os.makedirs(path+'/images/train')
    os.makedirs(path+'/images/val')
    os.makedirs(path + '/labels/train')
    os.makedirs(path + '/labels/test')
    os.makedirs(path + '/labels/val')
    print('Converting...... The yolo datasets will be stored in:', path)

    # 1. 保存到3个列表
    default_voc_path = './voc2012'
    train_name_list = read_file(default_voc_path+'/ImageSets/Main/train.txt')
    # test_name_list = read_file(default_voc_path+'/ImageSets/Main/test.txt')
    # val_name_list = read_file(default_voc_path+'/ImageSets/Main/val.txt')
    # print(val_name_list)
    # 2.按列表提取jpg
    print("moving images......................................")
    for jpg in tqdm(os.listdir(default_voc_path+'/JPEGImages/')):
        if os.path.splitext(jpg)[0] in train_name_list:
            shutil.copy(default_voc_path+'/JPEGImages/'+jpg, path+'/images/train')
        else:
            shutil.copy(default_voc_path + '/JPEGImages/' + jpg, path + '/images/val')
            shutil.copy(default_voc_path + '/JPEGImages/' + jpg, path + '/images/test')
    # 3+4
    print("creating txts......................................")
    for xml in tqdm(os.listdir(default_voc_path+'/Annotations/')):
        if os.path.splitext(xml)[0] in train_name_list:
            covert_xml2txt(default_voc_path+'/Annotations/'+xml, classes, path+'/labels/train')
        else:
            covert_xml2txt(default_voc_path + '/Annotations/' + xml, classes, path + '/labels/test')
            covert_xml2txt(default_voc_path + '/Annotations/' + xml, classes, path + '/labels/val')
    return print('Finished!!!')


#
def voc2coco(path, classes):
    print('The COCO dataset will be saved in:', path)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path+'/coco2017/train2017')
    os.makedirs(path+'/coco2017/val2017')
    os.makedirs(path+'/coco2017/test2017')
    os.makedirs(path + '/coco2017/annotations')
    # 1. 保存到3个列表
    default_voc_path = './voc2012'
    train_name_list = read_file(default_voc_path+'/ImageSets/Main/train.txt')

    # 2.按列表提取jpg
    print("moving images......................................")
    for jpg in tqdm(os.listdir(default_voc_path+'/JPEGImages/')):
        if os.path.splitext(jpg)[0] in train_name_list:
            shutil.copy(default_voc_path+'/JPEGImages/'+jpg, path+'/coco2017/train2017')
        else:
            shutil.copy(default_voc_path + '/JPEGImages/' + jpg, path + '/coco2017/val2017')
            shutil.copy(default_voc_path + '/JPEGImages/' + jpg, path + '/coco2017/test2017')

    # 创建一个存放的xml文件的文件夹，并按上面列表分一下train和val
    train_xml = []
    val_xml = []

    for xml in os.listdir(default_voc_path+'/Annotations/'):
        if os.path.splitext(xml)[0] in train_name_list:
            # print(xml)
            train_xml.append(default_voc_path + '/Annotations/' + xml)
        else:
            val_xml.append(default_voc_path + '/Annotations/' + xml)

    for data_type in ['train', 'val', 'test']:
        json_file = path + '/coco2017/annotations/instances_' + data_type + '2017.json'
        # xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
        if data_type == 'train':
            xml_files = train_xml
        else:
            xml_files = val_xml

        print(data_type + '数据集的xml文件数量为' + format(len(xml_files)))
        print('开始制作' + json_file)
        xml2json(xml_files, classes, json_file)
        print(json_file + '制作完毕')
    return


# 此函数被voc2coco调用
def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise ValueError("Can not find %s in %s." % (name, root.tag))
    if length > 0 and len(vars) != length:
        raise ValueError(
            "The size of %s is supposed to be %d, but is %d."
            % (name, length, len(vars))
        )
    if length == 1:
        vars = vars[0]
    return vars


# 此函数被voc2coco调用
def xml2json(xml_lists, classes, json_file):
    json_dict = {"images": [], "annotations": [], "categories": []}
    image_id = 1
    bnd_id = 1
    for xml_file in xml_lists:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        path = root.findall("path")
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, "filename", 1).text
        else:
            raise ValueError("%d paths found in %s" % (len(path), xml_file))

        size = get_and_check(root, "size", 1)
        width = int(get_and_check(size, "width", 1).text)
        height = int(get_and_check(size, "height", 1).text)

        images = {
            "file_name": filename,  # 图片名
            "height": height,
            "width": width,
            "id": image_id,  # 图片的ID编号（每张图片ID是唯一的）
        }
        json_dict["images"].append(images)

        for obj in root.findall("object"):
            category = get_and_check(obj, "name", 1).text
            if category not in classes:
                print('Warning', category)
            category_id = classes.index(category) + 1
            bndbox = get_and_check(obj, "bndbox", 1)
            xmin = int(get_and_check(bndbox, "xmin", 1).text) - 1
            ymin = int(get_and_check(bndbox, "ymin", 1).text) - 1
            xmax = int(get_and_check(bndbox, "xmax", 1).text)
            ymax = int(get_and_check(bndbox, "ymax", 1).text)
            assert xmax > xmin
            assert ymax > ymin
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            ann = {
                "area": o_width * o_height,
                "iscrowd": 0,
                "image_id": image_id,  # 对应的图片ID（与images中的ID对应）
                "bbox": [xmin, ymin, o_width, o_height],
                "category_id": category_id,
                "id": bnd_id,  # 同一张图片可能对应多个 ann
                "ignore": 0,
                "segmentation": [],
            }
            json_dict["annotations"].append(ann)
            bnd_id = bnd_id + 1
        image_id += 1

    for cid, cate in enumerate(classes):
        cat = {"supercategory": "none", "id": cid+1, "name": cate}
        json_dict["categories"].append(cat)
    os.makedirs(os.path.dirname(json_file), exist_ok=True)
    json.dump(json_dict, open(json_file, 'w'), indent=4)  # 存储json_dict
