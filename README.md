## ﻿ADVANCED MACHINE LEARNING ASSIGNMENT 2: WikiArt peregrinations - Anni Nieminen

This repository holds my code for assignment 2 LT2326 (WikiArt peregrinations).
Below, find description of changes made to the original code and discussion.

## Bonus A - Make the in-class example actually learn something
<p align="center">
  <img width="400" height="400" src="https://github.com/user-attachments/assets/39ed1222-abcc-486e-9834-ea843d321526">
</p>

In general, it seems like the original model doesn't learn much after epoch 2. This is why I changed the n of epochs from 20 to 10, focusing on making changes in the model architecture.

Furthermore, the labels in traindataset and testdataset were in a different order. This was changed by calling the function sorted() when defining self.classes in WikiArtDataset:

```bash
self.classes = list(sorted(classes))
```
Only making this change already improved the multiclass accuracy by 11%.

Change made in the model architecture: 

1. Added more 2 more convolutional layers. Increased the number of output channels.
   
```bash
self.conv2d1 = nn.Conv2d(3, 16, (2,2), padding=2) # n of out channels changed from 1 to 16, but kept kernel size at (2,2)
self.conv2d2 = nn.Conv2d(16, 32, (2,2), padding=2) # added this second convolutional layer
self.conv2d3 = nn.Conv2d(32, 64, (2,2), padding=2) #added this third convolutional layer
```
2. Added more batch normalization layers to go with the added conv layers.

```bash
self.batchnorm2d1 = nn.BatchNorm2d(16) #changed from (105x105) and also changed to BatchNorm2d
self.batchnorm2d2 = nn.BatchNorm2d(32) #added
self.batchnorm2d3 = nn.BatchNorm2d(64) #added
```
3. Increased dropout rate from 0.01 to 0.1.

```bash
self.dropout = nn.Dropout(0.1) #changed droupout rate from 0.01 to 0.1
```
Making these changes has led to the multiclass accuracy improving to 27%.

```bash
Accuracy: 0.28412699699401855
```
