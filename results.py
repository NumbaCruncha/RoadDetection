from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from tf_unet import unet
from tf_unet import util
from tf_unet import image_util
import cv2


def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [nx, ny, channels]

    :returns img: the rgb image [nx, ny, 3]
    """
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    img *= 255
    return img

def main():

    # input training and test datasets
    train_data = image_util.ImageDataProvider(search_path='RoadDetection_Train_Images')
    test_data = image_util.ImageDataProvider(search_path='RoadDetection_Test_Images')
    #
    # # train u-net
    # net = unet.Unet(layers=5, n_class=train_data.n_class, channels=train_data.channels, features_root=64, cost='dice_coefficient', cost_kwargs=dict(regularizer=0.01))
    net = unet.Unet(layers=4,
                    n_class=train_data.n_class,
                    channels=train_data.channels,
                    features_root=48,
                    cost='dice_coefficient',
                    cost_kwargs={'regularizer': 0.01,
                                 'class_weights': [0.1777, 0.8222]})

    x_test, y_test = test_data(10)

    # save prediction masks in TIFF format
    data_files = [d for d in test_data.data_files if d.split('.')[-1] == 'jpg']
    data_files.sort()
    for i, name in enumerate(data_files):
        file_name = name.split('\\')[1].split('.')[0]
        prediction = net.predict(model_path="./unet_trained/model.ckpt", x_test=x_test[i].reshape(-1, 600, 400, 3))
        ny = prediction.shape[2]
        img = to_rgb(prediction[..., 1])
        im.fromarray(img[0].round().astype(np.uint8)).save(r'RoadDetection_Test_Predictions\\{}_mask.tif'.format(file_name), 'TIFF', dpi=[300, 300], quality=90)


    # save prediction masks in JPEG format
    data_files = [d for d in test_data.data_files if d.split('.')[-1] == 'jpg']
    data_files.sort()
    for i, name in enumerate(data_files):
        file_name = name.split('\\')[1].split('.')[0]
        prediction = net.predict(model_path="./unet_trained/model.ckpt", x_test=x_test[i].reshape(-1, 600, 400, 3))
        ny = prediction.shape[2]
        img = to_rgb(prediction[..., 1])
        im.fromarray(img[0].round().astype(np.uint8)).save(r'RoadDetection_Test_Predictions\\{}_mask.jpg'.format(file_name), 'JPEG', dpi=[300, 300], quality=90)



    # predict mask from training data for presentation
    x_val, y_val = train_data(1)
    prediction = net.predict(model_path="./unet_trained/model.ckpt", x_test=x_val)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(x_val[0, ..., 0], aspect="auto")
    ax[1].imshow(y_val[0, ..., 1], aspect="auto")
    pred = np.squeeze(prediction[0, ..., 1])
    ax[2].imshow(pred, aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()

    # predict mask from test data for presentation
    x_val, y_val = test_data(1)
    prediction = net.predict(model_path="./unet_trained/model.ckpt", x_test=x_val)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].imshow(x_val[0, ..., 0], aspect="auto")
    ax[1].imshow(y_val[0, ..., 1], aspect="auto")
    pred = np.squeeze(prediction[0, ..., 1])
    ax[2].imshow(pred, aspect="auto")
    ax[0].set_title("Input")
    ax[1].set_title("Ground truth")
    ax[2].set_title("Prediction")
    fig.tight_layout()

    # presentation method 2
    x_test, y_test = test_data(10)
    img = util.combine_img_prediction(x_test, y_test, prediction)
    util.save_image(img, "test_pred_image.jpg")


    util.plot_prediction(x_test=x_train, y_test=x_train, prediction=validation)
    print('Process completed.')
if __name__ == '__main__':
    main()
