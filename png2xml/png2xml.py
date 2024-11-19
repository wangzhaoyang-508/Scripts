import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from scipy.ndimage import label
from tqdm import tqdm  # 导入tqdm库

import os
import numpy as np
from PIL import Image
from tqdm import tqdm



        # Additional code to handle the images, boxes, and save results would go here

def create_xml(filename, path, width, height, classes, boxes, savedir):
    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = filename
    ET.SubElement(root, "path").text = path
    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"

    for cl, box in zip(classes, boxes):
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = cl
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(int(box[0]))
        ET.SubElement(bndbox, "ymin").text = str(int(box[1]))
        ET.SubElement(bndbox, "xmax").text = str(int(box[2]))
        ET.SubElement(bndbox, "ymax").text = str(int(box[3]))

    tree = ET.ElementTree(root)
    tree.write(os.path.join(savedir, f"{filename.split('.')[0]}.xml"))


def masks_to_boxes(mask):
    """Convert a binary mask ndarray to bounding boxes for each object in the mask."""
    labeled_mask, num_features = label(mask)
    boxes = []
    for region in range(1, num_features + 1):
        positions = np.where(labeled_mask == region)
        xmin = positions[1].min()
        xmax = positions[1].max()
        ymin = positions[0].min()
        ymax = positions[0].max()
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def process_dataset(image_dir, mask_dir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    # 获取所有文件名
    filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".jpg")]

    # 使用 tqdm 包装文件列表，显示处理进度
    for filename in tqdm(filenames, desc="Processing images", unit="image"):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace(".jpg", ".png"))

        try:
            image = Image.open(image_path)
            mask = Image.open(mask_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue  # Skip this file and move to the next

        mask_array = np.array(mask)
        # print(mask_array)
        boxes = masks_to_boxes(mask_array)
        # print(boxes)

        # 假设每个框属于 'object' 类
        create_xml(filename, image_path, image.width, image.height, ['fracture'] * len(boxes), boxes, savedir)
# fracture   groove   ablation  breakdown

# 设置相应的目录
image_directory = r'F:/codes/github上传/png2xml/AEBAD/源域'
mask_directory = r'F:/codes/github上传/png2xml/AEBAD/源域'
save_directory = r'F:/codes/github上传/png2xml/AEBAD/xml/'

process_dataset(image_directory, mask_directory, save_directory)

