import os
from model import UNet
from dataset import DataGenerator
import tensorflow as tf
from labeldata import labels
import matplotlib.pyplot as plt

CWD = os.path.dirname(__file__)
LEARNING_RATE = 1e-3
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
TRAIN_IMG_DIR = os.path.join(CWD, 'data/train')
VAL_IMG_DIR = os.path.join(CWD, 'data/val')
MODEL_PATH = os.path.join(CWD, 'model/')
N_CLASSES = len(labels)

def plot_model(history):

    """
    Plots model accuracy and loss.
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig(os.path.join(CWD, 'acc.png'))
    plt.close()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc = 'upper left')
    plt.savefig(os.path.join(CWD, 'loss.png'))
    plt.close()

def train(model, train_data, val_data):


    """
    Trains the model, plots model accuracy and loss, and saves model.

    Args:
        model: model to train
        train_data: Training data generator
        val_data: Validation data generator

    """

    callback = tf.keras.callbacks.EarlyStopping("val_accuracy", patience = 15, min_delta = 0.001, mode="max")
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics = ["accuracy"])
    
    history = model.fit(train_data, 
              epochs=EPOCHS, 
              validation_data=val_data,
              callbacks = [callback]
              )
    
    plot_model(history)

    model.save(MODEL_PATH,save_format='tf')

if __name__ == "__main__":
    
    tf.random.set_seed(42)
    train_path = os.path.join(CWD, TRAIN_IMG_DIR)
    val_path = os.path.join(CWD, VAL_IMG_DIR)

    train_data_generator = DataGenerator(train_path, batch_size = BATCH_SIZE)
    val_data_generator = DataGenerator(val_path, shuffle = False, batch_size = BATCH_SIZE)

    model = UNet(N_CLASSES)

    train(model, train_data_generator, val_data_generator)
