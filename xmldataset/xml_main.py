import os
from tqdm import tqdm
from tabulate import tabulate
from converter import xml2voc, voc2yolo, voc2coco, check_xml
import argparse
import xml.etree.ElementTree as ET


def parse_args():
    parser = argparse.ArgumentParser(description='oooooo')
    # 功能1 统计文件夹内的文件数量
    parser.add_argument('path', help='dataset file path')
    # 功能2.检查xml文件和对应的图片名称是否对应
    parser.add_argument('--check_xml', help='need save path, Check .xml&.img correspondence, Save abnormal', type=str)
    # 功能3.将标注好的xml文件转化为标准的voc格式训练集
    parser.add_argument('--xml2voc', help='need a train_ratio float, build standard VOC2012 datasets', type=float)
    # 功能4.将xml文件（即voc格式数据集）转化为txt文件（即yolo格式数据集）
    parser.add_argument('--voc2yolo', help='need a save path, convert .xml to .txt', type=str)
    # 功能5.将xml文件（即voc格式数据集）转化为json文件（即COCO格式数据集）
    parser.add_argument('--voc2coco', help='need a save path, convert .xml to .json', type=str)

    args = parser.parse_args()
    # print(args)
    return args


# 读取此文件夹中的全部文件，并按后缀分组。
# 返回分组，分组是一个字典，键为后缀，值为路径列表。
def group_files_by_extension(folder_path):
    file_count = 0  # 文件夹中子文件数量
    # 创建一个字典，键为后缀名，值为一个列表，存放对应的文件。
    file_groups= {}
    # 遍历文件夹中的文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_count += 1  # 文件总数+1
            # 获取文件的后缀
            _, extension = os.path.splitext(file)

            # 如果存在后缀
            if extension:
                extension = extension[1:]  # 去除后缀中的点号
                file_path = os.path.join(root, file)
                if extension not in file_groups:
                    file_groups[extension] = []
                file_groups[extension].append(file_path)
    print("文件数量:", file_count)
    print("后缀数量:")
    for extension, count in file_groups.items():
        # print(f".{extension}: {count}")
        print(f".{extension}: {len(count)}")
    return file_groups


# 读取.xml文件中的信息，输入参数为第一步读取的字典
# 取入参字典中的'xml'键，对应的路径列表，逐一读取。
# 返回keydict,键是类别名，值是对应的object数量
def count_xml(xml_list):
    print('Counting xml files objects...... ')
    labeldict = {}
    for xml_flie in tqdm(xml_list['xml']):
        tree = ET.parse(xml_flie)
        root = tree.getroot()
        for obj in root.findall("object"):
            if obj[0].text not in labeldict.keys():
                labeldict[obj[0].text] = 1
            else:
                labeldict[obj[0].text] += 1
    keydict = dict(sorted(labeldict.items(), key=lambda x: x[1]))
    print("Finished！！！")
    print("%d classes and %d objects in total, Label Name and it's counts are:" % (len(keydict), sum(keydict.values())))
    # # 输出类别名称和个数
    # for key in keydict:
    #     print("%s:\t%d" % (key, keydict[key]))
    # print(keydict)
    table = {'class name': keydict.keys(), 'number': keydict.values()}
    print(tabulate(table, headers='keys', tablefmt='fancy_grid', showindex=True))
    return keydict


# 主函数
def main():
    args = parse_args()
    path = args.path       # 加载数据集文件夹路径
    file_groups = group_files_by_extension(path)   # 按文件名分组
    # count_xml(file_groups)  # 统计所有xml文件中，目标的类别和个数。
    classes = list(count_xml(file_groups).keys())  # 类名列表
    # print(classes)

    # 检查文件名与图像名是否一致，如果不一致。创建新文件夹保存，并输出名称.
    if args.check_xml:
        abnormal_path = args.check_xml
        check_xml(abnormal_path, file_groups)

    # 创建标准化VOC2012数据集(val=test)
    if args.xml2voc:
        train_ratio = args.xml2voc
        xml2voc(file_groups, train_ratio)

    if args.voc2yolo:
        save_path = args.voc2yolo
        voc2yolo(save_path, classes)

    if args.voc2coco:
        save_path = args.voc2coco
        voc2coco(save_path, classes)


if __name__ == '__main__':
    main()
