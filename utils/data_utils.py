import logging
import os

import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    random_split,
)
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


# def get_loader(args):
#     if args.local_rank not in [-1, 0]:
#         torch.distributed.barrier()

#     transform_train = transforms.Compose(
#         [
#             transforms.RandomResizedCrop(
#                 (args.img_size, args.img_size), scale=(0.05, 1.0)
#             ),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ]
#     )

#     transform_test = transforms.Compose(
#         [
#             transforms.Resize((args.img_size, args.img_size)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
#         ]
#     )

#     # Path to the dataset
#     # data_dir = r"C:\Users\avs20\Documents\GitHub\ViT-EmotionClassificationPytorch\data"  # Update this if needed
#     data_dir = (
#         "/Users/annastuckert/Documents/GitHub/ViT-EmotionClassificationPytorch/data"
#     )
#     # Check if the directory exists
#     assert os.path.exists(data_dir), f"Dataset path {data_dir} does not exist."

#     # Load the dataset using ImageFolder
#     dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

#     # Split the dataset into train and test sets
#     train_size = int(0.8 * len(dataset))  # 80% for training
#     test_size = len(dataset) - train_size  # 20% for testing
#     trainset, testset = random_split(dataset, [train_size, test_size])

#     # Apply test-specific transforms to the test set
#     testset.dataset.transform = transform_test

#     if args.local_rank == 0:
#         torch.distributed.barrier()

#     # Define samplers
#     train_sampler = (
#         RandomSampler(trainset)
#         if args.local_rank == -1
#         else DistributedSampler(trainset)
#     )
#     test_sampler = SequentialSampler(testset)

#     # Define DataLoaders
#     train_loader = DataLoader(
#         trainset,
#         sampler=train_sampler,
#         batch_size=args.train_batch_size,
#         num_workers=4,
#         pin_memory=True,
#     )

#     test_loader = DataLoader(
#         testset,
#         sampler=test_sampler,
#         batch_size=args.eval_batch_size,
#         num_workers=4,
#         pin_memory=True,
#     )

#     return train_loader, test_loader

import logging
import os

import torch
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    random_split,
)
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)


def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                (args.img_size, args.img_size), scale=(0.05, 1.0)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Dataset Path
    data_dir = (
        "/Users/annastuckert/Documents/GitHub/ViT-EmotionClassificationPytorch/data"
    )

    # Debugging output
    print(f"Checking dataset path: {data_dir}")
    assert os.path.exists(data_dir), f"Dataset path {data_dir} does not exist."

    # Load dataset
    dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

    # Print detected classes and class indices
    print("\n📌 Detected Classes in Dataset:", dataset.classes)
    print("📌 Class-to-Index Mapping:", dataset.class_to_idx)

    # Print number of images per class
    class_counts = {class_name: 0 for class_name in dataset.classes}
    for _, label in dataset.samples:
        class_counts[dataset.classes[label]] += 1

    print("\n📊 Image Counts Per Class:")
    for class_name, count in class_counts.items():
        print(f"   {class_name}: {count} images")

    # Ensure dataset is not empty
    assert len(dataset) > 0, "❌ Dataset is empty! No valid images found."

    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))  # 80% training
    test_size = len(dataset) - train_size  # 20% testing
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Apply test-specific transforms
    testset.dataset.transform = transform_test

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Define samplers
    train_sampler = (
        RandomSampler(trainset)
        if args.local_rank == -1
        else DistributedSampler(trainset)
    )
    test_sampler = SequentialSampler(testset)

    # Define DataLoaders
    train_loader = DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    test_loader = DataLoader(
        testset,
        sampler=test_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, test_loader
