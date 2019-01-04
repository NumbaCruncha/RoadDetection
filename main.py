from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util
import cv2

def main():

    # input training and test datasets
    train_data = image_util.ImageDataProvider(search_path='RoadDetection_Train_Images', n_class=2)

    # instantiate U-net (best results: layers=5, feature_roots=64, batch_size=2, epochs=50, training_iters=64)
    net = unet.Unet(layers=4,
                    n_class=train_data.n_class,
                    channels=train_data.channels,
                    features_root=48,
                    cost='dice_coefficient',
                    cost_kwargs={'regularizer': 0.01})

    trainer = unet.Trainer(net,
                           batch_size=2,
                           verification_batch_size=4,
                           optimizer="momentum",  opt_kwargs=dict(momentum=0.5))

    # path = trainer.train(data_provider=train_data, output_path="./unet_trained", training_iters=32,  epochs=1, display_step=2)
    trainer.train(data_provider=train_data,
                         output_path="./unet_trained",
                         training_iters=64,
                         epochs=50,
                         dropout=0.75,
                         display_step=2)

    print('Process completed.')

if __name__ == '__main__':
    main()
