# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from torchvision.datasets import MNIST
import xml.etree.ElementTree as ET
from zipfile import ZipFile
import argparse
import tarfile
import shutil
import gdown
import uuid
import json
import os
import urllib
import requests

# from wilds.datasets.fmow_dataset import FMoWDataset


# utils #######################################################################

def stage_path(data_dir, name):
    full_path = os.path.join(data_dir, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path


def download_and_extract(url, dst, remove=True):
    gdown.download(url, dst, quiet=False)

    if dst.endswith(".tar.gz"):
        tar = tarfile.open(dst, "r:gz")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".tar"):
        tar = tarfile.open(dst, "r:")
        tar.extractall(os.path.dirname(dst))
        tar.close()

    if dst.endswith(".zip"):
        zf = ZipFile(dst, "r")
        zf.extractall(os.path.dirname(dst))
        zf.close()

    if remove:
        os.remove(dst)

def download_gta5(data_dir):
    # Original URL: http://hemanthdv.org/OfficeHome-Dataset/
    full_path = stage_path(data_dir, "gta5/images")

    for i in range(6, 8):
        if i != 10:
            download_and_extract(f"https://download.visinf.tu-darmstadt.de/data/from_games/data/0{i}_images.zip",
                         os.path.join(data_dir, f"gta5/images/0{i}_images.zip"))
        else:
            download_and_extract("https://download.visinf.tu-darmstadt.de/data/from_games/data/10_images.zip",
                         os.path.join(data_dir, "gta5/images/10_images.zip"))
    
    # for i in range(1, 11):
    #     if i != 10:
    #         download_and_extract(f"https://download.visinf.tu-darmstadt.de/data/from_games/data/0{i}_labels.zip",
    #                      os.path.join(data_dir, f"gta5/labels/0{i}_labels.zip"))
    #     else:
    #         download_and_extract("https://download.visinf.tu-darmstadt.de/data/from_games/data/10_labels.zip",
    #                      os.path.join(data_dir, "gta5/labels/10_labels.zip"))
            
def download_cityscape(data_dir):
    response = requests.get("https://www.cityscapes-dataset.com/file-handling/?packageID=1", stream=True)
    with open(os.path.join(data_dir, f"cityscapes/leftImg8bit_trainvaltest.zip"), 'wb') as f:
        for chunk in response.iter_content(chunk_size=128):
            f.write(chunk)

    print("Extracting the file...")
    with zipfile.ZipFile(os.path.join(data_dir, f"cityscapes/leftImg8bit_trainvaltest.zip"), 'r') as zip_ref:
        extract_path = os.path.join(data_dir, f"cityscapes/leftImg8bit_trainvaltest")
        zip_ref.extractall(extract_path)

    print(f"File extracted to {extract_path}")
    
    # download_and_extract(f"https://www.cityscapes-dataset.com/file-handling/?packageID=1",
    #                 os.path.join(data_dir, f"cityscapes/leftImg8bit_trainvaltest.zip"))
    
    # download_and_extract(f"https://www.cityscapes-dataset.com/file-handling/?packageID=3",
    #                 os.path.join(data_dir, f"cityscapes/gtFine_trainvaltest.zip"))

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('--data_dir', type=str, default='datasets/')
    args = parser.parse_args()
    
    download_gta5(args.data_dir)
    # download_cityscape(args.data_dir)
    
    