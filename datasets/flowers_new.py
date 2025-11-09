###################################################################################################
#
# Copyright (C) 2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
"""
Flowers Dataset
"""
import os
import sys
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from PIL import Image
import ai8x

torch._dynamo.config.suppress_errors = True
torch.manual_seed(0)

def augment_affine_jitter_blur(orig_img):
    """
    Augment with multiple transformations
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), shear=5),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.2),
        transforms.CenterCrop((180, 180)),
        transforms.ColorJitter(brightness=.7),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
        transforms.RandomHorizontalFlip(),
    ])
    return train_transform(orig_img)

def augment_blur(orig_img):
    """
    Augment with center crop and bluring
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((220, 220)),
        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
    ])
    return train_transform(orig_img)

def flowers_get_datasets(data, load_train=True, load_test=True, aug=2):
    """
    Load Flowers dataset
    """
    (data_dir, args) = data
    path = data_dir
    dataset_path = os.path.join(path, "flowers")
    is_dir = os.path.isdir(dataset_path)
    if not is_dir:
        print("******************************************")
        print("Please follow the instructions below:")
        print("Create a 'flowers' directory in the 'data' folder")
        print("Create 'train' and 'test' subdirectories")
        print("Place flower images in class-specific folders under train and test")
        print("Make sure that images are in the following directory structure:")
        print("  'data/flowers/train/class1'")
        print("  'data/flowers/train/class2'")
        print("  'data/flowers/test/class1'")
        print("  'data/flowers/test/class2'")
        print("Re-run the script. The script will create an 'augmented' folder")
        print("with all the original and augmented images. Remove this folder if you want")
        print("to change the augmentation and to recreate the dataset.")
        print("******************************************")
        sys.exit("Dataset not found!")
    else:
        processed_dataset_path = os.path.join(dataset_path, "augmented")

        if os.path.isdir(processed_dataset_path):
            print("augmented folder exists. Remove if you want to regenerate")

        train_path = os.path.join(dataset_path, "train")
        test_path = os.path.join(dataset_path, "test")
        processed_train_path = os.path.join(processed_dataset_path, "train")
        processed_test_path = os.path.join(processed_dataset_path, "test")
        if not os.path.isdir(processed_dataset_path):
            os.makedirs(processed_dataset_path, exist_ok=True)
            os.makedirs(processed_test_path, exist_ok=True)
            os.makedirs(processed_train_path, exist_ok=True)

            # create label folders for train
            for d in os.listdir(train_path):
                mk = os.path.join(processed_train_path, d)
                try:
                    os.mkdir(mk)
                except OSError:
                    pass

            # create label folders for test
            for d in os.listdir(test_path):
                mk = os.path.join(processed_test_path, d)
                try:
                    os.mkdir(mk)
                except OSError:
                    pass

            # copy test folder files
            test_cnt = 0
            # accept common image extensions (case-insensitive)
            IMG_EXTS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
            for (dirpath, _, filenames) in os.walk(test_path):
                print(f'copying {dirpath} -> {processed_test_path}')
                for filename in filenames:
                    if filename.lower().endswith(IMG_EXTS):
                        src = os.path.join(dirpath, filename)
                        dst = os.path.join(processed_test_path, os.path.basename(dirpath), filename)
                        img = Image.open(src).convert('RGB')
                        img = augment_blur(img)
                        img.save(dst)
                        test_cnt += 1

            # copy and augment train folder files
            train_cnt = 0
            for (dirpath, _, filenames) in os.walk(train_path):
                print(f'copying and augmenting {dirpath} -> {processed_train_path}')
                for filename in filenames:
                    if filename.lower().endswith(IMG_EXTS):
                        src = os.path.join(dirpath, filename)
                        dst_base = os.path.join(processed_train_path, 
                                              os.path.basename(dirpath),
                                              os.path.splitext(filename)[0])
                        img = Image.open(src).convert('RGB')
                        
                        # Save original
                        img.save(f"{dst_base}_orig.jpg")
                        train_cnt += 1
                        
                        # Save augmented versions
                        for i in range(aug):
                            img_aug = augment_affine_jitter_blur(img)
                            img_aug.save(f"{dst_base}_aug{i}.jpg")
                            train_cnt += 1

            print(f'Augmented dataset: {test_cnt} test, {train_cnt} train samples')

    # Loading and normalizing train dataset
    if load_train:
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        train_dataset = torchvision.datasets.ImageFolder(root=processed_train_path,
                                                       transform=train_transform)
    else:
        train_dataset = None

    # Loading and normalizing test dataset
    if load_test:
        test_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ai8x.normalize(args=args)
        ])

        test_dataset = torchvision.datasets.ImageFolder(root=processed_test_path,
                                                      transform=test_transform)

        if args.truncate_testset:
            test_dataset = Subset(test_dataset, [0])
    else:
        test_dataset = None

    return train_dataset, test_dataset

# Dataset configuration for AI8X training
datasets = [
    {
        'name': 'flowers',
        'input': (3, 128, 128),
        'output': tuple(['class' + str(i) for i in range(6)]),  # 6 flower classes: daisy, dandelion, iris, rose, sunflower, tulip
        'loader': flowers_get_datasets,
    },
]