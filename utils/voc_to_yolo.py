import os
import xml.etree.ElementTree as ET
import shutil

# Modified from: https://blog.csdn.net/qq_65966646/article/details/149064056

# Config path
VOC_ROOT = "VOC2012"    # Original VOC dataset path
YOLO_ROOT = "dataset"  # YOLO format output path

# VOC type list - only keep person class
CLASSES = ["person"]

# Create YOLO dir structure
os.makedirs(f"{YOLO_ROOT}/images/train", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/images/val", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/labels/train", exist_ok=True)
os.makedirs(f"{YOLO_ROOT}/labels/val", exist_ok=True)


# Convert XML annotation to YOLO format
def convert_annotation(xml_file):
    tree = ET.parse(f"{VOC_ROOT}/Annotations/{xml_file}")
    root = tree.getroot()
    
    # Check if the XML contains a person object
    has_person = False
    for obj in root.iter('object'):
        cls = obj.find('name').text
        if cls == "person":
            has_person = True
            break
    
    # If no person object, skip this file
    if not has_person:
        return False
    
    w = int(root.find('size/width').text)
    h = int(root.find('size/height').text)

    txt_file = xml_file.replace('.xml', '.txt')
    with open(f"{YOLO_ROOT}/labels/train/{txt_file}", 'w') as f:
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in CLASSES:
                continue
            cls_id = CLASSES.index(cls)
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            xmax = float(bbox.find('xmax').text)
            ymin = float(bbox.find('ymin').text)
            ymax = float(bbox.find('ymax').text)
            # Convert to YOLO format (normalized center coordinates and width/height)
            x_center = (xmin + xmax) / 2 / w
            y_center = (ymin + ymax) / 2 / h
            width = (xmax - xmin) / w
            height = (ymax - ymin) / h
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return True


# Copy images and split into train/validation sets
def copy_and_split(converted_files):
    # Read VOC official split files
    def read_split(file):
        with open(f"{VOC_ROOT}/ImageSets/Main/{file}", 'r') as f:
            return [line.strip() for line in f.readlines()]

    train_files = read_split("train.txt")  # Training set file name list (without extension)
    val_files = read_split("val.txt")  # Validation set file name list (without extension)

    # Filter to only include files that have person annotations
    train_files = [f for f in train_files if f in converted_files]
    val_files = [f for f in val_files if f in converted_files]

    # Process training set
    for file_name in train_files:
        # Copy image
        shutil.copy(
            f"{VOC_ROOT}/JPEGImages/{file_name}.jpg",
            f"{YOLO_ROOT}/images/train/{file_name}.jpg"
        )
        # Generate labels (already handled in convert_annotation)

    # Process validation set
    for file_name in val_files:
        shutil.copy(
            f"{VOC_ROOT}/JPEGImages/{file_name}.jpg",
            f"{YOLO_ROOT}/images/val/{file_name}.jpg"
        )
        # Move labels to val directory
        shutil.move(
            f"{YOLO_ROOT}/labels/train/{file_name}.txt",
            f"{YOLO_ROOT}/labels/val/{file_name}.txt"
        )

    # Generate train.txt and val.txt path lists
    with open(f"{YOLO_ROOT}/train.txt", 'w') as f:
        for file_name in train_files:
            f.write(f"{YOLO_ROOT}/images/train/{file_name}.jpg\n")
    with open(f"{YOLO_ROOT}/val.txt", 'w') as f:
        for file_name in val_files:
            f.write(f"{YOLO_ROOT}/images/val/{file_name}.jpg\n")


# Execute conversion and splitting
if __name__ == "__main__":
    # Step 1: Convert all XML labels to YOLO format (temporarily store in labels/train)
    converted_files = []
    for xml_file in os.listdir(f"{VOC_ROOT}/Annotations"):
        if xml_file.endswith('.xml'):
            if convert_annotation(xml_file):
                converted_files.append(xml_file.replace('.xml', ''))

    # Step 2: Split into training/validation sets (only process files with person annotations)
    copy_and_split(converted_files)

    print(f"Conversion and splitting completed! Processed {len(converted_files)} files containing person annotations.")