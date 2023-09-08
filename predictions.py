import matplotlib.pyplot as plt
import os

from labeldata import id_to_color
import tensorflow as tf
from dataset import DataGenerator
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

CWD = os.path.dirname(__file__)
PRED_IMG_DIR = os.path.join(CWD, 'predictions/')
VAL_IMG_DIR = os.path.join(CWD, 'data/val')

def save_predictions(predictions):

    """
    Saves predicted encoding to a directory predictions.

    Args:
        predictions: Predicted encoding of the UNet.

    """
    for index, prediction in enumerate(predictions):
        temp = np.zeros([prediction.shape[0], prediction.shape[0], 3])
        for row in range(prediction.shape[0]):
            for col in range(prediction.shape[0]):
                temp[row, col, :] = id_to_color[prediction[row, col]]
                temp = temp.astype('uint8')
        plt.imshow(temp)
        plt.axis('off')
        img_path = PRED_IMG_DIR + str(index + 1) + '.jpg'
        plt.savefig(img_path)
        plt.close()
                         
if __name__ == "__main__":

    current_dir = os.path.dirname(__file__)
    val_path = os.path.join(current_dir, VAL_IMG_DIR)
    model = tf.keras.models.load_model(os.path.join(CWD, 'model/'))
    val_data_generator = DataGenerator(val_path, shuffle = False, batch_size = 16, predicting = True)
    predictions = model.predict(val_data_generator)
    predictions = np.argmax(predictions, axis=-1)
    save_predictions(predictions)







