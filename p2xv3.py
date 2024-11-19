import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import xml.etree.cElementTree as ET


def create_xml(filename, boxes, folder, image_size):
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = folder
    ET.SubElement(root, "filename").text = filename

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(image_size[1])
    ET.SubElement(size, "height").text = str(image_size[0])
    ET.SubElement(size, "depth").text = str(image_size[2])

    for box in boxes:
        color, x_min, y_min, x_max, y_max = box
        object = ET.SubElement(root, "object")
        ET.SubElement(object, "name").text = color
        bndbox = ET.SubElement(object, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(x_min)
        ET.SubElement(bndbox, "ymin").text = str(y_min)
        ET.SubElement(bndbox, "xmax").text = str(x_max)
        ET.SubElement(bndbox, "ymax").text = str(y_max)

    tree = ET.ElementTree(root)
    tree.write(os.path.join(folder, filename.replace('.bmp', '.xml')))


def find_connected_components(mask):
    from scipy.ndimage import label
    structure = np.ones((3, 3))  # 8-connectivity
    connected_components, num_features = label(mask, structure)
    return connected_components, num_features


def masks_to_boxes(mask_array):
    color_to_class = {(0, 0, 255): 'Blue', (0, 255, 0): 'Green', (255, 0, 0): 'Red'}
    boxes = []
    for color, class_name in color_to_class.items():
        class_mask = np.all(mask_array == np.array(color), axis=-1)
        connected_components, num_features = find_connected_components(class_mask)

        for feature in range(1, num_features + 1):
            where = np.where(connected_components == feature)
            if where[0].size > 0 and where[1].size > 0:
                y_min, x_min, y_max, x_max = np.min(where[0]), np.min(where[1]), np.max(where[0]), np.max(where[1])
                boxes.append((class_name, x_min, y_min, x_max, y_max))
    return boxes


def process_dataset(image_dir, mask_dir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    filenames = [filename for filename in os.listdir(image_dir) if filename.endswith(".bmp")]

    for filename in tqdm(filenames, desc="Processing images", unit="image"):
        image_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, filename.replace(".bmp", ".png"))

        try:
            image = Image.open(image_path)
            mask = Image.open(mask_path).convert('RGB')  # Ensure the mask is in RGB
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        mask_array = np.array(mask)
        boxes = masks_to_boxes(mask_array)
        image_size = mask_array.shape

        create_xml(filename, boxes, savedir, image_size)  # Generate XML file for the image

        # Additional code to handle the images, boxes, and save results would go here

image_directory = r'F:\codes\github上传\png2xml\images'
mask_directory = r'F:/codes/github上传/png2xml/labels/'
save_directory = r'F:/codes\github上传\png2xml\xmls2'

process_dataset(image_directory, mask_directory, save_directory)