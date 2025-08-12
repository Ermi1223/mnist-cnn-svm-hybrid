import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from config.settings import config

def load_mnist():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Plot class distribution
    plot_class_distribution(y_train)
    
    # Preprocess images
    x_train = preprocess_images(x_train)
    x_test = preprocess_images(x_test)
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_images(images):
    """Normalize and reshape images"""
    images = images.astype('float32') / 255.0
    return np.expand_dims(images, axis=-1)  # Add channel dimension

def plot_class_distribution(labels):
    """Visualize class distribution"""
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels.flatten())
    plt.title('MNIST Class Distribution')
    plt.xlabel('Digit Class')
    plt.ylabel('Count')
    plt.tight_layout()
    os.makedirs(config.RESULT_SAVE_PATH, exist_ok=True)
    plt.savefig(
        os.path.join(config.RESULT_SAVE_PATH, 'class_distribution.png'),
        dpi=config.PLOT_DPI,
        format=config.PLOT_FORMAT
    )
    plt.close()

def get_data_loaders():
    """Create train/validation/test data loaders"""
    (x_train, y_train), (x_test, y_test) = load_mnist()
    
    # Create validation split
    val_size = int(len(x_train) * config.VAL_SPLIT)
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_train, y_train = x_train[val_size:], y_train[val_size:]
    
    return {
        'train': (x_train, y_train),
        'val': (x_val, y_val),
        'test': (x_test, y_test)
    }