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
This already improved the multiclass accuracy by 11%.
