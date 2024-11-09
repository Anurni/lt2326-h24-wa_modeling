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

Changes made in the model architecture: 

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
Making these changes has led to the multiclass accuracy reaching 28%.

```bash
Accuracy: 0.28412699699401855
```
## Part 1 - Fix class imbalance (5 points)

The classes (the art types) did indeed have a very high class imbalance, for instance Analytical_Cubism only had 15 instances whereas Impressionism had 2269 instances (average being ~476 instances per class).
I decided to explore using SMOTE (Synthetic Minority Over-sampling Technique) to adress this issue. (https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
My original plan was to use 'auto' as the resampling strategy (resample all classes but the majority class, thus generating synthetic data samples to match the majority class n of instances (2269).
However, this proved out to be too time- and memory-consuming. That is why I chose the 'dict' as the sampling strategy. All classes with less than 400 data points were to be 'filled' with synthetic data points.
See below how this was done: 
```bash
 sampling_strategy = {}
    label_counts = Counter(y_train)
    
    # we need to check the n of instances per class since oversampling will not work if the n of instances wanted is lower than the existing n of instances  
    for label in label_counts:
        if label_counts[label] > 400:
            sampling_strategy[label] = label_counts[label]  # use existing if n of instances surpasses 400
        else:
            sampling_strategy[label] = 400  # use 400 if not
            
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    # we need to flatten X_train since SMOTE only accepts dimensions of <=2
    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flattened, y_train)
    print("done with resampling")
    print("this is y_train after the SMOTE process", Counter(y_train_resampled))
```
This produced this output: 
```bash
this is y_train after the SMOTE process Counter({np.int64(12): 2269, np.int64(21): 1712, np.int64(23): 1157, np.int64(9): 1127, np.int64(20): 946, np.int64(4): 721, np.int64(3): 688, np.int64(24): 679, np.int64(0): 449, np.int64(17): 413, **np.int64(2): 400**, np.int64(5): 400, np.int64(7): 400, np.int64(26): 400, np.int64(16): 400, np.int64(15): 400, np.int64(22): 400, np.int64(14): 400, np.int64(13): 400, np.int64(10): 400, np.int64(11): 400, np.int64(8): 400, np.int64(18): 400, np.int64(6): 400, np.int64(19): 400, np.int64(25): 400, np.int64(1): 400})
```
Here we can see the indices of the classes inside the np.int64, for instance class n 2 (marked with **) refers to Analytical_Cubism (which had only 15 data instances). After oversampling, the class had 400 instances.


