#Installing dependencies we will need while using PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import copy
import argparse
import os
import logging
import sys
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

logger=logging.getLogger(__name__) #means of tracking events
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def test(model, test_loader, criterion, device):

    model.eval() 
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        
        inputs = inputs.to(device) #GPU
        labels = labels.to(device) #GPU
        
        outputs=model(inputs) #first we get predictions
        loss=criterion(outputs, labels)
        
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    #loss
    total_loss = running_loss / len(test_loader.dataset)
    #accuracy
    total_acc = running_corrects / len(test_loader.dataset) 
    
    # we have two different ways to show the result in the logs
    logger.info(f"Testing Loss: {total_loss}")
    logger.info(f"Testing Accuracy: {total_acc}")
    
    print ('Printing Log')
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            total_loss, running_corrects, len(test_loader.dataset), 100.0 * total_acc
        )
    )
    

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    
    # reminder: we did not do hyperparameter optimization for epoch with hyperparameter ranges. 

    epochs=5
    best_loss=1e5
    # loading train and validation datasets
    image_dataset={'train':train_loader, 'validation':validation_loader} 
    loss_counter=0
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'validation']:
            # model would be in two different modes for validation and training
            if phase=='train':
                model.train()
            else:
                model.eval()
            
            
            #definition of loss
            running_loss = 0.0 
            #definition of correct predictions
            running_corrects = 0 
            samples = 0
            
            # Iteration over training and validation datasets with gradients, backward prop and updating the parameters
            for inputs, labels in image_dataset[phase]:
                
                # GPU Usage to boost 
                inputs = inputs.to(device) 
                labels = labels.to(device) 
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train': #only happening with training
                    optimizer.zero_grad() #gradient reset
                    loss.backward() # backward prop
                    optimizer.step() # updating the parameters

                _, preds = torch.max(outputs, 1) #here are predictions
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                samples += len(inputs)
                
                if samples % 100 == 0:
                    print(f'Total sample: {samples}')
                    accuracy = running_corrects / samples
                    print(f'Log Entry: Epoch {epoch}, Phase {phase}.')
                    print(
                        'Images [{}/{} ({:.0f}%)] / Loss: {:.2f} / Accumulated Loss: {:.0f} / '
                        'Accuracy: {}/{} ({:.2f}%)'.format(
                            samples,
                            len(image_dataset[phase].dataset),
                            100 * (samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_loss,
                            running_corrects,
                            samples,
                            100 * accuracy
                        ))

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects // len(image_dataset[phase]) 
            
            if phase=='validation':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc, best_loss))
            
        if loss_counter==1:
            break
#         if epoch==0:
#             break
    return model
    
def net():

    # ResNet50 model is a convolutional neural network (CNN) that is 50 layers deep. Each of the 2 layer blocks in previous version of the model Resnet34 was replaced with 3layer bottleneck blocks. It has higher accuracy than Resnet34. We will use pretrained model
    model = models.resnet50(pretrained=True) 
    # getting number of features which we will provide in the first layer
    num_features = model.fc.in_features

    for param in model.parameters():
        param.requires_grad = False   

    # I would like to enhance my model by adding additional layers. The advantage of Rectified Linear Unit -ReLU- is that it does not activate all the neurons at the same time.
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 128),
                   nn.ReLU(),
                   nn.Linear(128, 64),
                   nn.ReLU(),
                   nn.Linear(64,5)
            )
    return model

def create_data_loaders(data, batch_size):
    train_data_path = os.path.join(data, 'train')
    test_data_path = os.path.join(data, 'test')
    validation_data_path=os.path.join(data, 'validation')

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ])

    train_data = torchvision.datasets.ImageFolder(root=train_data_path, transform=train_transform)
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=test_transform)
    test_data_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

    validation_data = torchvision.datasets.ImageFolder(root=validation_data_path, transform=test_transform)
    validation_data_loader  = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, shuffle=True) 
    
    return train_data_loader, test_data_loader, validation_data_loader

def main(args):
    
    print(f'Log entry: Train batch size:{args.batch_size}')
    print(f'Log entry: Learning rate:{args.learning_rate}')
    
    # Here were go with GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    

    model = net()
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss() #using crossentropyloss which is used for Multiclass Classification
    optimizer = optim.Adagrad(model.parameters(), lr=args.learning_rate) #using adagrad which has adaptive learning rate
    
    logger.info("Starting Model Training")
    train_loader, test_loader, validation_loader=create_data_loaders(args.data, args.batch_size)
#     train_loader, validation_loader, test_loader = create_data_loaders(args.train_dir, args.test_dir, args.validation_dir, args.batch_size, args.test_batch_size)
    train(model, train_loader, validation_loader, criterion, optimizer, device)

    logger.info("Testing Model")
    test(model, test_loader, criterion, device)

    logger.info("Saving Model")
    torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))

if __name__=='__main__':
    
    
    parser=argparse.ArgumentParser()
    
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--data', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
        
    args=parser.parse_args()
    print(args)
    main(args)
