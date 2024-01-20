"""

Contains functionality for creating pytorch dataloaders for image classification data

"""

import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

NUM_WORKERS = 0

def create_dataloaders(
    train_dir:str,
    test_dir:str,
    transforms:transforms.Compose,
    batch_size:int,
    num_workers:int = NUM_WORKERS):
    """ 
    Creating training and testing DataLoaders.

    Takes in a training directory and testing directory path and turns them into
    Pytorch dataset and then  Pytorch dataloaders

    Args:
        train_dir: Path to training directory
        test_dir: Path to testing directory
        transforms: Pythorch transfrom function
        batch_size: Number of items in a batch
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        Where class_names is a list of target classes
        Example usage:
            train_dataloader, test_dataloaders, class_names = create_dataloader(
                train_dir=path/to/train/dir,
                transfrom=some_trainsfrom,
                batch_size=32,
                num_workers=4
            )

    """

    # Use ImageFolder to create dataset(s)
    train_data = datasets.ImageFolder(root=train_dir, # target folder of images
                                    transform=transforms, # transforms to perform on data (images)
                                    target_transform=None) # transforms to perform on labels (if necessary)

    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=transforms)
    # Get class names
    class_names = train_data.classes


    # Create a training and testing dataloader 
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size, # how many samples per batch?
                                num_workers=num_workers, # how many subprocesses to use for data loading? (higher = more)
                                shuffle=True,
                                pin_memory=True) # shuffle the data?

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                pin_memory=True) # don't usually need to shuffle testing data

    train_dataloader, test_dataloader
    

    return train_dataloader, test_dataloader, class_names
