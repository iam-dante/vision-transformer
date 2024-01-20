
""" 

Trains a pytorch image classification model using device-agnotics code

"""

import torch
import os
from torchvision import transforms
import data_setup, utils, engine, model_builder
from timeit import default_timer as timer

import argparse

parser = argparse.ArgumentParser(description='Some Machine Learning Code')
parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
parser.add_argument("--batch_size", type=int,  default=16, help="Size of number of items in a batch")
parser.add_argument("--units_hidden", type=int,  default=10, help="Number of hidden units in the model")
parser.add_argument("--learning_rate",  type=float,  default=0.001, help="The learning rate")
parser.add_argument("--model_name", type=str,  default="model", help="Model saved name")

args = parser.parse_args()

# Hyperparameters
NUM_EPOCHS=args.num_epochs
BATCH_SIZE=args.batch_size
HIDDEN_UNITS=args.units_hidden
LEARNING_RATE = args.learning_rate

# Setup directories
train_dir = "dataset/pizza_steak_sushi/train/"
test_dir = "dataset/pizza_steak_sushi/test/"

# System device agnotic code
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Create transforms
data_transforms = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])

# Create dataset and dataloaders
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                test_dir=test_dir,
                                                                                transforms=data_transforms,
                                                                                batch_size=BATCH_SIZE)

# Create model
model = model_builder.TinyVGG(input_shape=3,
                                hidden_units=HIDDEN_UNITS,
                                output_shape=len(class_names)).to(device)

parser.add_argument("--model", type=str, default=model, help="Model saved name")


# Setup loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                            lr=LEARNING_RATE)

# Start the timer
start_time = timer()

# Training with help of engine.py
engine.train(model=model,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            epochs=NUM_EPOCHS,
            device=device)
# End the timer and print out how long it took
end_time = timer()
print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds")

utils.save_model(model=model,
                target_dir="models",
                model_name=args.model_name,

                )
