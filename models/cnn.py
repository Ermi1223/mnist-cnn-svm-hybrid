import tensorflow as tf
from tensorflow.keras import layers, models
from config.settings import config

def build_cnn_model():
    """Build CNN model architecture"""
    model = models.Sequential([
        layers.Conv2D(config.CNN_PARAMS['filters'][0], 
                      config.CNN_PARAMS['kernel_size'], 
                      activation='relu', 
                      input_shape=(*config.IMG_SIZE, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(config.CNN_PARAMS['filters'][1], 
                      config.CNN_PARAMS['kernel_size'], 
                      activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(config.CNN_PARAMS['dropout_rate']),
        layers.Flatten(),
        layers.Dense(config.CNN_PARAMS['dense_units'], activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config.CNN_PARAMS['learning_rate']
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model