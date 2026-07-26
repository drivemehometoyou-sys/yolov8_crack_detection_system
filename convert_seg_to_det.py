# convert_seg_to_det.py
import os
import shutil
from glob import glob

def poly_to_bbox(points):
    """
    Convert normalized polygon coordinates to YOLO bbox (xc, yc, w, h).
    points: list of float [x1, y1, x2, y2, ...], length even.
    Returns (xc, yc, w, h) each in [0,1].
    """
    xs = points[0::2]
    ys = points[1::2]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    w = x_max - x_min
    h = y_max - y_min
    xc = x_min + w / 2.0
    yc = y_min + h / 2.0
    # clamp to [0,1] (prevent numerical issues)
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))
    return xc, yc, w, h

def convert_labels_file(src_txt, dst_txt):
    """
    Read polygon labels from src_txt, write YOLO detection labels to dst_txt.
    Returns (ok, error_msg) or (True, None) if successful.
    """
    if not os.path.exists(src_txt):
        return False, f"Source file not found: {src_txt}"
    
    with open(src_txt, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line_no, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            print(f"  Warning: {src_txt} line {line_no} has < 3 tokens, skipped")
            continue
        class_id = parts[0]
        coords = []
        try:
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            print(f"  Warning: {src_txt} line {line_no} contains non-float values, skipped")
            continue
        
        if len(coords) % 2 != 0:
            print(f"  Warning: {src_txt} line {line_no} has odd number of coordinates, skipped")
            continue
        if len(coords) < 4:
            print(f"  Warning: {src_txt} line {line_no} has fewer than 2 points, skipped")
            continue
        
        xc, yc, w, h = poly_to_bbox(coords)
        # Write in YOLO detection format: class_id xc yc w h
        new_lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    
    if not new_lines:
        return False, f"No valid annotations in {src_txt}"
    
    os.makedirs(os.path.dirname(dst_txt), exist_ok=True)
    with open(dst_txt, 'w') as f:
        f.writelines(new_lines)
    return True, None

def copy_images(src_img_dir, dst_img_dir):
    """Copy all image files (common formats) from src to dst."""
    os.makedirs(dst_img_dir, exist_ok=True)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    image_files = []
    for ext in exts:
        image_files.extend(glob(os.path.join(src_img_dir, ext)))
    for img_path in image_files:
        shutil.copy2(img_path, os.path.join(dst_img_dir, os.path.basename(img_path)))
    return len(image_files)

def process_split(src_root, dst_root, split_name):
    """
    Process one split (train/val/test).
    Expected structure:
        src_root / split_name / images / *.jpg
        src_root / split_name / labels / *.txt
    Output structure:
        dst_root / split_name / images / (same images)
        dst_root / split_name / labels / (converted .txt)
    """
    src_img_dir = os.path.join(src_root, split_name, 'images')
    src_lab_dir = os.path.join(src_root, split_name, 'labels')
    dst_img_dir = os.path.join(dst_root, split_name, 'images')
    dst_lab_dir = os.path.join(dst_root, split_name, 'labels')
    
    if not os.path.isdir(src_img_dir) or not os.path.isdir(src_lab_dir):
        print(f"  ⚠️  Split '{split_name}' missing images or labels directory, skipped.")
        return 0, 0
    
    print(f"\n📂 Processing {split_name} ...")
    # Copy images
    num_images = copy_images(src_img_dir, dst_img_dir)
    print(f"   Copied {num_images} images")
    
    # Convert labels
    txt_files = glob(os.path.join(src_lab_dir, '*.txt'))
    converted = 0
    failed = 0
    for txt_path in txt_files:
        rel_name = os.path.basename(txt_path)
        dst_txt = os.path.join(dst_lab_dir, rel_name)
        ok, err = convert_labels_file(txt_path, dst_txt)
        if ok:
            converted += 1
        else:
            failed += 1
            print(f"   ❌ Failed {rel_name}: {err}")
    print(f"   Converted {converted} label files, failed {failed}")
    return num_images, converted

def main():
    # 源数据集路径（已有实例分割数据集）
    src_base = r"C:\Users\zhuchaoyang\Desktop\MyBelovedYOLO\yolov8_crack_detection_system\datasets\my_cracks"
    # 目标数据集路径（新的目标检测数据集）
    dst_base = r"C:\Users\zhuchaoyang\Desktop\MyBelovedYOLO\yolov8_crack_detection_system\datasets\my_real_cracks"
    
    if not os.path.isdir(src_base):
        print(f"❌ Source directory not found:\n   {src_base}")
        return
    
    # 需要处理的子集列表（根据实际存在的情况）
    splits = ['train', 'val', 'test']
    
    print("🚀 Starting conversion from instance segmentation to detection (polygon -> bounding box)")
    print(f"Source: {src_base}")
    print(f"Target: {dst_base}")
    
    total_imgs = 0
    total_labels = 0
    for split in splits:
        imgs, lbls = process_split(src_base, dst_base, split)
        total_imgs += imgs
        total_labels += lbls
    
    print("\n✅ Conversion completed.")
    print(f"   Total images copied: {total_imgs}")
    print(f"   Total label files converted: {total_labels}")
    print("\n📌 Now you can use the dataset in my_real_cracks for YOLO detection training.")
    print("   Remember to update your data.yaml file to point to the new dataset path.")

if __name__ == "__main__":
    main()