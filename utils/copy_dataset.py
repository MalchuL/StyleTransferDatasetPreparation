import shutil
import os
from tqdm import tqdm

ORIG_FOLDER = '/home/malchul/work/GAN/DCT_Net/data/finetune_ds/jjbav2/train_real'
TARGET_FOLDER = '/home/malchul/work/GAN/StyleTransferDatasetPreparation/dumped/jojo_256_no_blur/good_samples'
OUTPUT_FOLDER = '/home/malchul/work/GAN/StyleTransferDatasetPreparation/dumped/jojo_256_no_blur/train_real'


os.makedirs(OUTPUT_FOLDER, exist_ok=True)
original_images = os.listdir(ORIG_FOLDER)
target_images = os.listdir(TARGET_FOLDER)
for img in tqdm(original_images):
    if img in target_images:
        shutil.copyfile(os.path.join(TARGET_FOLDER, img), os.path.join(OUTPUT_FOLDER, img))