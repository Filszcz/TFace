import torch
import random
import pandas as pd
import numpy as np
import os
import torch.utils.data
from timeit import default_timer as timer
from tqdm.auto import tqdm
from torchinfo import summary
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from torchvision import transforms, datasets
import torchvision
from PIL import Image
from torch.utils.data import Dataset
import wandb

import wandb
import random

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="JAN_FACE",

    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.0001,
    "architecture": "TinyVGG",
    "dataset": "KTFE",
    "epochs": 200,
    }
)

# made as an exercise for the ML course
# https://www.learnpytorch.io/
# https://www.youtube.com/watch?v=Z_ikDlimN6A&t=73466s&ab_channel=DanielBourke
# https://www.youtube.com/watch?v=V_xro1bcAuA&ab_channel=freeCodeCamp.org

# data from kaggle
# https://www.kaggle.com/datasets/ngkhang/emotion-visual-thermal

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

BATCH_SIZE = 64
NUM_WORKERS = int(os.cpu_count()/2)
SEED = 42
EPOCHS = 200
IMG_SIZE = 128

random.seed(SEED)
torch.manual_seed(SEED)

path = Path("data")

data_transforms = transforms.Compose([
    transforms.Resize([IMG_SIZE,IMG_SIZE]),
    transforms.RandomHorizontalFlip(p=0.5),
    # transforms.RandomRotation(degrees=(0, 180)),
    transforms.TrivialAugmentWide(num_magnitude_bins=10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
   
])

data = datasets.ImageFolder(root=path,
                            transform=data_transforms)

train_data, test_data = torch.utils.data.random_split(data, [0.8, 0.2])

train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKERS,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             shuffle=False)


classes = data.classes
print(f"This dataset has {len(classes)} classes.")
print(classes)

#tiny VGG

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) 
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*IMG_SIZE*8,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

#train step 

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    
    model_0.train()

    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    wandb.log({"train_loss": train_loss, "train_acc": train_acc})
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():

        for batch, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    wandb.log({"test_loss": test_loss, "test_acc": test_acc})
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []}
    
    for epoch in tqdm(range(epochs)):
        wandb.log({"epoch": epoch})
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        #Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results

model_0 = TinyVGG(input_shape=3, 
                  hidden_units=64,
                  output_shape=len(data.classes)).to(device)

model_1 = torchvision.models.resnet18()
model_1.fc = nn.Linear(512, len(data.classes))

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_0.parameters(),
                             lr=0.0001)

summary(model_0, input_size=[1,3,IMG_SIZE,IMG_SIZE])

start_time = timer()

model_1_results = train(model=model_0,
                        train_dataloader=train_dataloader,
                        test_dataloader=test_dataloader,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=EPOCHS)

end_time = timer()

print(f"Total training time: {end_time-start_time:.3f} seconds")

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "Fruits_classification_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

wandb.finish()