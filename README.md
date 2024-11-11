## ﻿ADVANCED MACHINE LEARNING ASSIGNMENT 2: WikiArt peregrinations - Anni Nieminen

This repository holds my code for assignment 2 LT2326 (WikiArt peregrinations).
Below, find description of changes made to the original code and discussion.

### Bonus A - Make the in-class example actually learn something --> (wikiart_classification/wikiart.py & train.py)
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
### Part 1 - Fix class imbalance --> (wikiart_classification/train.py)

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
What was interesting after inspecting the testing results was that the multiclass accuracy actually went down by around 6%.
This is probably due to the overfitting to the generated synthetic data. 
```bash
Accuracy: 0.22380952537059784
```
By lowering the threshold for data instances per class to 200 instead of 400 like above (i.e., only generating synthetic data up to 200 data points per class), I managed actually to get the accuracy up to 29%!
```bash
Accuracy: 0.2936508059501648
```
TO TRAIN THE MODEL, SIMPLY RUN THE SCRIPT BY SPECIFYING THE N OF EPOCHS: (note that running this script will take a few minutes due to the upsampling)
```bash
python train.py 10
```

### Part 2 - Autoencode and cluster representations --> (wikiart_encoding/train-encoded_clusters.py & wikiart_encoded_clusters.py)

## 1. Creating an autoencoder that produces compressed representations of the images in the dataset:
   In order to create compressed representations of the images in the dataset, the model architecture had to be changed.
   For the convolution, I kept three convolutional layers, but precising that the stride length will be 2 (leading to n of pixels getting divided by half):
   ```bash
    # we start with image input dimensions of (3, 416*416) 
        self.conv2d1 = nn.Conv2d(3, 16, 2, stride=2, padding=1) # n of in channels = 3, n of out channels=16, 2x2 kernel and stride of 2. 
        #After the first conv layer, the output dimensions will be (16, 209*209) since our stride is 2 and the goal is to compress.
        
        self.conv2d2 = nn.Conv2d(16, 32, 2, stride=2, padding=1) # n of in channels = 16, n of out channels=32, 2x2 kernel and stride of 2.
        #After the second conv layer, the output dimensions will be (32, 105*105) since our stride is 2 and the goal is to compress.
        
        self.conv2d3 = nn.Conv2d(32, 64, 2, stride=2, padding=1) #n of in channels=32, n of out channels=64, 2x2 kerkel and stride of 2.
        #After the third conv layer, the output dimensions will be (64, 53*53) since our stride is 2 and the goal is to compress.
   ```
   In order to create the "latent" representations of the images (the compressed ones), the dimensions were flattened:
   ```bash
    # compressed representation layer, resulting in a vector with dimension of 1x300
        self.compressed_rep = nn.Linear(179776, latent_compressed_rep_dim)
   ```
   Additionally, to feed these compressed representations to the deconvoluting layers (which will allow us to investigate how well the model 'performs'), we had to define these layers:
   ```bash
    # compressed rep into decoder input size
        self.decoder_input = nn.Linear(latent_compressed_rep_dim, 64*53*53)

        # DECONVOLUTING (decoding) LAYERS ARE DEFINED HERE:
        ########################################

        # we start with the input image dimensions from self.decoder_input layer, which has 64 channels and 2809 pixels
        self.deconv2d1 = nn.ConvTranspose2d(64, 32, 2, stride=2, padding=1, output_padding=1)
        #After the first deconvoluting layer, the output dimensions will be (32,105*105) since stride is 2 and the goal is to decompress.
        self.deconv2d2 = nn.ConvTranspose2d(32, 16, 2, stride=2, padding=1, output_padding=1)
        #After the second deconvoluting layer, the output dimensions will be (16,209*209) since stride is 2 and the goal is to decompress.
        self.deconv2d3 = nn.ConvTranspose2d(16, 3, 2, stride=2, padding=1)
        #After the third deconvoluting layer, the output dimensions will be (3, 416*416) since stride is 2 ad the goal is to decompress.
   ```
After the first trials (with applying only the convolutional and deconvolutional layers in addition to the reLU activation function, these training loss were obtained from 10 epochs:
![image](https://github.com/user-attachments/assets/721ac23e-b8e8-4bae-852a-d6479f4bdcec)

TO TRAIN THE ENCODER MODEL, SIMPLY RUN WITH SPECIFYING THE N OF EPOCHS WANTED:
```bash
python train-encoded_clusters.py 10
```

## 2. Saving and clustering the representations using clustering methods from scikit-learn  --> (wikiart_encoding/test-encoded_clusters.py)
   In this part, I will use the latent (compressed) representations of the images to plot a cluster graph in order to see if the model has learnt to cluster different art styles.
   The encoded representations were retrieved by passing the inputs through only the encoder of the model:
   ```bash
    compressed_image_representations = []
    for batch_id, batch in enumerate(tqdm.tqdm(loader)):
        X, y = batch
        #y = y.to(device)
        output = model(X, setting="encode") #we are only passing the input through the encoder now to get the compressed representation
        compressed_image_representations.append(output.detach().cpu().numpy().reshape(output.size(0), -1)) # we need to flatten the representation

    compressed_image_representations = np.vstack(compressed_image_representations) #stacking the batches
   ```
   As the clustering method I chose K-means due to its simplicity, specifying the n of clusters as 27 (n of classes in our dataset).
   The compressed representations were dimensionally reduced with PCA.

   Here is the resulting plotting of the clusters (centroids of the clusters are named). These results were obtained from the model trained with 10 epochs and learning rate 0.001.
   ![image](https://github.com/user-attachments/assets/eb73aac0-b82e-4b7b-91dc-18b83195925e)

We can indeed observe a rather nice clustering of the art styles, for instance Pointillism is a clearly separated cluster on its own on the right of the graph. Cubism and its subcategories Analytical Cubism and Synthetic Cubism are close to each other, and this goes for late and early renaissance too. Pop Art and Colour Field painting are both art styles with bright colours, distinct shapes, which makes sense looking at their neighboring clusters. 

  Here are the class labels of our dataset:
   0 : 'Abstract_Expressionism', 
   1: 'Action_painting', 
   2: 'Analytical_Cubism', 
   3: 'Art_Nouveau_Modern', 
   4: 'Baroque', 
   5: 'Color_Field_Painting', 
   6: 'Contemporary_Realism', 
   7: 'Cubism', 
   8: 'Early_Renaissance', 
   9: 'Expressionism', 
   10: 'Fauvism', 
   11: 'High_Renaissance', 
   12: 'Impressionism',
   13: 'Mannerism_Late_Renaissance', 
   14: 'Minimalism', 
   15: 'Naive_Art_Primitivism', 
   16: 'New_Realism', 
   17: 'Northern_Renaissance', 
   18: 'Pointillism', 
   19: 'Pop_Art', 
   20: 'Post_Impressionism',
   21: 'Realism', 
   22: 'Rococo', 
   23: 'Romanticism',
   24: 'Symbolism',
   25: 'Synthetic_Cubism',
   26: 'Ukiyo_e'
   

   
