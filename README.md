# Road Detection
Detection of roadways from satellite data  
Data provided: 45 training examples, 10 test images  

# Problem Statement
Derive a mask depicting roadways from a 3200 x 4800 satellite image. 


# Methodology
This is an image segmentation problem requiring pixel-wise binary classification of the input image resulting in a two-class output image. A fully connected convolutional network (FCN) was selected to solve this problem as it is suitable for use with a small number of training examples.

**Solution Architecture:**  
U-net is a fast implementation of an FCN which provides good segementation capability using a symettrical downsampling and upsampling network of layers. A pre-existing open-source implemtation (tf_unet) was utilised for this purpose.


**Optimisation:**  
- Input images were scaled from (3200, 4800) down to (400, 600) to circumvent resource constraints.
- The number of layers in the unet was increased from 3 to 5 which was the resource maximum. This resutled in an average 2-3% decrease in average error after 50 epochs.
- Batch normalization was added to the convolutional layer to scale and normalise the input arrays resulting in faster convergence.


# Results  
Training occured over 64 iterations across 50 epochs. The network was trained using an RTX2070 GPU in approximately 90 minutes resulting in an average accruacy of 92%.

![Alt text](https://user-images.githubusercontent.com/14899131/50379633-9928a080-06b3-11e9-8d0b-7f88bc6cacbb.png?raw=true "Title")



**Improvements**
In order to increase the power of the network, it would be beneficial to train on mulitple GPUs. Horovod - a C++ library developed by Uber for distributed tensorflow operations (https://github.com/uber/horovod) - would be worth investigating to this end. 

An imploenmetation of U-net with iHorovod integration built-in would be a logical choice for testing alongside the proposed solution.
https://github.com/ankurhanda/tf-unet







