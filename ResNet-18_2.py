#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cProfile import label
import os
import shutil
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl


# In[2]:


class LambdaLayer(nn.Module):
    def __init__(self, lambd):  #run when object of LambdaLayer is initiaed
        super(LambdaLayer, self).__init__() #initialize object
        self.lambd = lambd
        #applies lamdbda function
    
    def forward(self, x):  # x = input tensor
        return self.lambd(x)


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, option='A'):  # option A - default shortcut connection
        super(BasicConvBlock, self).__init__()
        #initialization of Block object


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        #defince components of convolution block

        self.shortcut = nn.Identity()
        #define shortcut connection for ResNet architecture
        


        if stride != 1 or in_channels != out_channels:  #conditions meant --> shortcut needs to be adjusted
            pad_channels = out_channels // 4
            self.shortcut = LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, pad_channels, pad_channels, 0, 0)))
            #alters padding of input tensor
            #dimensions of the tensor align during forwards passing
        
    def forward(self, x):  #actual passing of data and application of block parameters
        out = self.conv1(x)
        out = self.batch_norm1(out)
        out = self.activation1(out)
        out = self.conv2(out)
        out = self.batch_norm2(out)

        shortcut_out = self.shortcut(x)
        out += shortcut_out
        out = F.relu(out)
        return out


# In[3]:

'''
class Network(nn.Module):  #define network
    
    def __init__(self, block_type, block_num):
        super(Network, self).__init__()
        self.in_channels = 16  # no of input channels
        
        self.conv0 = self._conv_block(3, 16, kernel_size=3, stride=1, padding=1, bias=False)#layer 0:performs conv operation, batch, RELU 
                                                                                            
        self.blocks = nn.ModuleList()  #list for block layers
        self.blocks.append(self._layer_block(block_type, 16, block_num[0], starting_stride=1))
        self.blocks.append(self._layer_block(block_type, 32, block_num[1], starting_stride=2))
        self.blocks.append(self._layer_block(block_type, 64, block_num[2], starting_stride=2))
        #adds specific number of blocks for each stage of the network to the list

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10) #64-dimensional input --> 10-dimensional output

    def _conv_block(self, in_channels, out_channels, **kwargs): #helper method to create convolutional block
        return nn.Sequential( #defines the sequential operations of the block
            nn.Conv2d(in_channels, out_channels, **kwargs), #2d convolution
            nn.BatchNorm2d(out_channels), #batch normalization
            nn.ReLU(inplace=True) 
        )

    def _layer_block(self, block_type, out_channels, block_num, starting_stride):  #creates block layer
        strides_list = [starting_stride] + [1] * (block_num - 1)
        layers = []
        
        for stride in strides_list:
            layers.append(block_type(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers) #creates all the block instances

    def forward(self, x): #forward pass of data
        x = self.conv0(x)
        for block in self.blocks: #list of layers of the network 
            x = block(x)
        x = self.avgpool(x) #adaptive average pooling --> fixed-size representation
        x = torch.flatten(x, 1) #collapes tensor dimenion (except for batch)
        x = self.linear(x) #linear transformation to tensor that creates comprehensenable result
        return x

'''

class Network(pl.LightningModule):
    
    def __init__(self, block_type, block_num):
        super(Network, self).__init__()
        self.in_channels = 16  # no of input channels
        
        self.conv0 = self._conv_block(3, 16, kernel_size=3, stride=1, padding=1, bias=False)#layer 0:performs conv operation, batch, RELU 
                                                                                            
        self.blocks = nn.ModuleList()  #list for block layers
        self.blocks.append(self._layer_block(block_type, 16, block_num[0], starting_stride=1))
        self.blocks.append(self._layer_block(block_type, 32, block_num[1], starting_stride=2))
        self.blocks.append(self._layer_block(block_type, 64, block_num[2], starting_stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64, 10) #64-dimensional input --> 10-dimensional output



    def _conv_block(self, in_channels, out_channels, **kwargs): #helper method to create convolutional block
        return nn.Sequential( #defines the sequential operations of the block
            nn.Conv2d(in_channels, out_channels, **kwargs), #2d convolution
            nn.BatchNorm2d(out_channels), #batch normalization
            nn.ReLU(inplace=True) 
        )
    

    def _layer_block(self, block_type, out_channels, block_num, starting_stride):  #creates block layer
            strides_list = [starting_stride] + [1] * (block_num - 1)
            layers = []
            
            for stride in strides_list:
                layers.append(block_type(self.in_channels, out_channels, stride))
                self.in_channels = out_channels

            return nn.Sequential(*layers) #creates all the block instances

    def forward(self, x): #forward pass of data
        x = self.conv0(x)
        for block in self.blocks: #list of layers of the network 
            x = block(x)
        x = self.avgpool(x) #adaptive average pooling --> fixed-size representation
        x = torch.flatten(x, 1) #collapes tensor dimenion (except for batch)
        x = self.linear(x) #linear transformation to tensor that creates comprehensenable result
        return x

    
    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        prediction = self(inputs)
        loss = F.cross_entropy(prediction, labels)
        self.log('train_loss', loss)  # Logging the training loss
        return loss 
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)
        










# In[4]:


def ResNet18_test():
    return Network(block_type = BasicConvBlock , block_num = [2,2,2,2])


# In[5]:


resnet = ResNet18_test()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

resnet.to(device)
summary(resnet, (3,32,32))


# In[6]:


def data():
    transform = transforms.Compose([transforms.Resize((224,2244)),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.5,0.5,0.5], std = [0.5,0.5,0.5])])
    path = "C:/Users/alexc/IB1/Caltech/caltech-101/caltech-101"
    dataset = ImageFolder(path, transform=transform)

    train = int(0.8 * len(dataset)) #training data
    test = len(dataset) - train #testing data
    train , test = random_split(dataset, (train, test)) 

    print("Training Images:  {} ".format(len(train)))
    print("Testing Images: {} ".format(len(test)))

    Batch_size = 32

    trainLoader = DataLoader(train, batch_size = Batch_size, shuffle = True)
    testLoader = DataLoader(test, batch_size = Batch_size, shuffle = True)
    #provide iterables
    
    return trainLoader, testLoader
    


# In[7]:


trainLoader, testLoader = data()


# In[8]:


crit = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=0.01)


# In[9]:


def train_resnet():
    epochs = 15
    train_samples_num = 45000
    val_samples_num = 5000
    train_costs, val_costs = [], [] #to store training and validation losses

    for epoch in range(epochs):
        resnet.train()
        train_running_loss = 0
        correct_train = 0

        for inputs, labels in trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() #optimizer set to 0
            
            #start forwading data
            prediction = resnet(inputs)

            loss = crit(prediction, labels)

            #backpropagation
            loss.backward()
            optimizer.step()

            train_running_loss += loss.item() * inputs.size(0)

            _, predicted_outputs = torch.max(prediction.data, 1)
            correct_train += (predicted_outputs == labels).sum().item()
            #calculates number of correctly predicted trainign samples

        train_epoch_loss = train_running_loss / train_samples_num #avg training loss
        train_costs.append(train_epoch_loss)
        train_acc = correct_train / train_samples_num #accuracy

        resnet.eval() #evaluation mode - dropout and batch normalization are disabled
        val_running_loss = 0
        correct_val = 0

        with torch.no_grad():
            for inputs, labels in testLoader:
                inputs, labels = inputs.to(device), labels.to(device)

            
                prediction = resnet(inputs)

                loss = crit(prediction, labels)

                val_running_loss += loss.item() * inputs.size(0)

                _, predicted_outputs = torch.max(prediction.data, 1)
                correct_val += (predicted_outputs == labels).sum().item()

        val_epoch_loss = val_running_loss / val_samples_num
        val_costs.append(val_epoch_loss)
        val_acc = correct_val / val_samples_num

        info = "[Epoch {}/{}]: train-loss = {:0.6f} | train-acc = {:0.3f} | val-loss = {:0.6f} | val-acc = {:0.3f}" 
        print(info.format(epoch+1, epochs, train_epoch_loss, train_acc, val_epoch_loss, val_acc)) #training process

        torch.save(resnet.state_dict(), '/content/checkpoint_gpu_{}'.format(epoch + 1)) #saves dictionary of trained model

    torch.save(resnet.state_dict(), '/content/resnet-18_weights_gpu') #final trained model dictionary is saved

    return train_costs, val_costs


# In[ ]:




if __name__ == "__main__":
    resnet = Network(block_type=BasicConvBlock, block_num=[2, 2, 2, 2])
    trainLoader, testLoader = data()

    # Create a PyTorch Lightning Trainer
    trainer = pl.Trainer(max_epochs=15) # gpus=1 if torch.cuda.is_available() else None

    # Train the model using the PyTorch Lightning Trainer
    trainer.fit(resnet, trainLoader)

    # Save the final trained model weights
    torch.save(resnet.state_dict(), '/content/resnet-18_weights_gpu')


