
from modular_functions import engine, data_setup
from modular_functions.utils import set_seeds
import torch
import torchvision
from torch import nn

# Get pretrained weights for ViT-Base
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT # requires torchvision >= 0.13, "DEFAULT" means best available

# Setup a ViT model instance with pretrained weights
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

# Freeze the base parameters
for parameter in pretrained_vit.parameters():
    parameter.requires_grad = False

# Transform acccording to the pretrained weights
vit_transform = pretrained_vit_weights.transforms()


def train(train_dir,
        test_dir,
        batch_size:int=32,
        model: torch.nn.Module=pretrained_vit,
        transforms:torchvision.transforms._presets.ImageClassification=vit_transform,
        device: torch.device="gpu"):

        """ 
        
            Trains a Vision Transformer

            Parameters:

            - train_dir (pathlib.PosixPath) - training directory dataset
            - test_dir (pathlib.PosixPath) - testing directory dataset
            - batch_size (int) - batch size DEFAULT = 32
            - model  - DEFAULT = pretrained model of 16 base model from the paper
            - transform (torchvision.transforms._presets.ImageClassification) - DEFAULT vit_transfrom from pretrained model
            - device (torch.device) = DEFAULT = "gpu"


            Returns:

            model - The trained model
            results -  {"train_loss": [],
                                "train_acc": [],
                                "test_loss": [],
                                "test_acc": []
                                }
    
        """

        # Create a dataloader with the new transform
        train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                            test_dir=test_dir,
                                                                                                            transforms=transforms,
                                                                                                            batch_size=batch_size) # Could increase if we had more samples, such as here: https://arxiv.org/abs/2205.01580 (there are other improvements there too...)

        model.heads = nn.Linear(in_features=768, out_features=len(class_names))

        # Create optimizer and loss function according to the paper
        optimizer = torch.optim.Adam(params=model.parameters(),
                                    lr=1e-3)

        loss_fn = torch.nn.CrossEntropyLoss()


        set_seeds(42, device)

        # Train the classifier head of the pretrained ViT feature extractor model
        results = engine.train(model=model.to(device),
                                            train_dataloader=train_dataloader_pretrained,
                                            test_dataloader=test_dataloader_pretrained,
                                            optimizer=optimizer,
                                            loss_fn=loss_fn,
                                            epochs=10,
                                            device=device)
        
        return model, results
