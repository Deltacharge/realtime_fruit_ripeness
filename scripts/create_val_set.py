import os
import shutil
import random

# CONFIG
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
VAL_SPLIT = 0.15   # 15% validation

CLASSES = ["fresh", "rotten", "unripe"]

random.seed(42)  # reproducibility

for cls in CLASSES:
    class_train_path = os.path.join(TRAIN_DIR, cls)
    class_val_path = os.path.join(VAL_DIR, cls)

    os.makedirs(class_val_path, exist_ok=True)

    images = os.listdir(class_train_path)
    images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]

    random.shuffle(images)

    val_count = int(len(images) * VAL_SPLIT)
    val_images = images[:val_count]

    for img in val_images:
        src = os.path.join(class_train_path, img)
        dst = os.path.join(class_val_path, img)
        shutil.move(src, dst)

    print(f"{cls}: moved {val_count} images to validation set")

print("\n✅ Validation split created successfully!")
