import os
import tensorflow as tf
from experiments.base_experiment import BaseExperiment
from models.cnn import build_cnn_model
from utils.visualizer import plot_training_history
from config.settings import config
import numpy as np

class CNNExperiment(BaseExperiment):
    def __init__(self):
        super().__init__("cnn")
        self.model = build_cnn_model()
        
    def train(self, train_data, val_data):
        x_train, y_train = train_data
        x_val, y_val = val_data
        
        history = self.model.fit(
            x_train, y_train,
            epochs=config.CNN_PARAMS['epochs'],
            batch_size=config.BATCH_SIZE,
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Save training history plot
        plot_training_history(
            history, 
            save_path=os.path.join(
                config.RESULT_SAVE_PATH, 
                "cnn_training_history.png"
            )
        )
        
        return history
    
    def evaluate(self, test_data):
        x_test, y_test = test_data
        test_loss, test_acc = self.model.evaluate(x_test, y_test, verbose=0)
        y_pred = np.argmax(self.model.predict(x_test, verbose=0), axis=1)
        
        report, _ = self.generate_report(y_test, y_pred, "cnn_")
        
        self.logger.info(f"Test Accuracy: {test_acc:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        
        return test_acc, report
    
    def save_model(self):
        model_path = os.path.join(self.model_save_path, "cnn_model.h5")
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
        self.logger.info(f"Model loaded from {path}")
    
    def _evaluate_on_noisy_data(self, noisy_x, y_test):
        """Evaluate CNN on noisy data"""
        test_loss, test_acc = self.model.evaluate(noisy_x, y_test, verbose=0)
        return test_acc