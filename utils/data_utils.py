import logging
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler, random_split

logger = logging.getLogger(__name__)

def get_loader(args):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((args.img_size, args.img_size), scale=(0.05, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Path to the dataset
    data_dir = r"C:\Users\avs20\Documents\GitHub\ViT-EmotionClassificationPytorch\data"  # Update this if needed

    # Check if the directory exists
    assert os.path.exists(data_dir), f"Dataset path {data_dir} does not exist."

    # Load the dataset using ImageFolder
    dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

    # Split the dataset into train and test sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    test_size = len(dataset) - train_size  # 20% for testing
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Apply test-specific transforms to the test set
    testset.dataset.transform = transform_test

    if args.local_rank == 0:
        torch.distributed.barrier()

    # Define samplers
    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
    test_sampler = SequentialSampler(testset)

    # Define DataLoaders
    train_loader = DataLoader(trainset,
                              sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              pin_memory=True)
    
    test_loader = DataLoader(testset,
                             sampler=test_sampler,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True)

    return train_loader, test_loader
