# Road Detection
Detection of roadways within satellite data  
Data provided: 45 training examples, 10 test images  

# Problem Statement
Derive a mask depicting roadways from each 3200 x 4800 satellite image tile.


# Methodology
This is an image segmentation problem requiring pixel-wise binary classification of the input image resulting in a two-class output image. An extension of a fully connected convolutional network (FCN) was selected to solve this problem as it is suitable for use with a small number of training examples. A total of 8 false positives were removed from the dataset during preprocessing.


**Solution Architecture:**  
U-net provides good segementation capability using a symmetrical downsampling and upsampling network of layers which is more efficient than an FCN. A pre-existing open-source implementation (tf_unet) was utilised for this purpose.

**Optimisation:**  
- Input images were scaled from (3200, 4800) down to (400, 600) to circumvent resource constraints.
- The number of layers in the unet was set 4 with 48 features, which provided good performance. 
- Batch normalization was added to the convolutional layer to scale and normalise the input arrays resulting in faster convergence.

# Results  
Training occured over 64 iterations across 50 epochs. The network was trained using an RTX2070 GPU in approximately 90 minutes resulting in an average accuracy of 98%.


**Model training Accuracy:** 
![Alt text](https://user-images.githubusercontent.com/14899131/50678063-cd13a680-1061-11e9-82e9-ec0e4e1e4afd.png "Training Accuracy")


**Mask creation from the training set:** 
![Alt text](https://user-images.githubusercontent.com/14899131/50678099-00eecc00-1062-11e9-8ab5-d2d60b798563.png "Training Result")


**Creating a mask from the test set using the pre-trained model (no ground truth mask)** 
![Alt text](https://user-images.githubusercontent.com/14899131/50678105-077d4380-1062-11e9-9275-719eb4785c68.png "Test Result")


**Improvements:**  
It would be desirable to avoid downsampling the training images in order to capture more details during learning, as well as to output a mask at the same resolution as the training images.

In order to increase the power of the network, it would be beneficial to train on multiple GPUs. Horovod - a C++ library developed by Uber for distributed tensorflow operations (https://github.com/uber/horovod) would be worth investigating to this end. 

An implementation of U-net with iHorovod integration built-in would be a logical choice for testing alongside the proposed solution.
https://github.com/ankurhanda/tf-unet







