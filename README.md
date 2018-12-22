# RoadDetection
Detection of roadways from satellite data  
Data provided: 45 training examples, 10 test images  

# Problem Statement
Derive a mask depicting roadways from a 3200 x 4800 satellite image. 


# Methodology
This is an image segmentation problem requiring pixel-wise binary classification of the input image resulting in a two-class output image. A fully connected convolutional network was selected to solve this problem as it is suitable for use with a small number of training examples.




