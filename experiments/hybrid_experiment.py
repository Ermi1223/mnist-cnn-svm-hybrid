import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from experiments.base_experiment import BaseExperiment
from models.cnn_svm import CNN_SVM
from utils.visualizer import plot_hybrid_performance
from config.settings import config

class HybridExperiment(BaseExperiment):
    def __init__(self, cnn_model_path=None):
        super().__init__("hybrid")
        self.cnn_model_path = cnn_model_path
        self.model = None
        
    def train(self, train_data, val_data):
        x_train, y_train = train_data
        
        # Load or train CNN
        if self.cnn_model_path and os.path.exists(self.cnn_model_path):
            self.model = CNN_SVM(cnn_model=tf.keras.models.load_model(self.cnn_model_path))
        else:
            self.model = CNN_SVM()
        
        # Extract features
        train_features = self.model.extract_features(x_train)
        
        # Train SVM
        self.model.train_svm(train_features, y_train)
        
        return None
    
    def evaluate(self, test_data):
        x_test, y_test = test_data
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report, _ = self.generate_report(y_test, y_pred, "hybrid_")
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        
        # Plot hybrid performance
        plot_hybrid_performance(
            y_test, 
            y_pred, 
            save_path=os.path.join(config.RESULT_SAVE_PATH, "hybrid_performance.png")
        )
        
        return accuracy, report
    
    def save_model(self):
        model_path = os.path.join(self.model_save_path, "cnn_svm_model.joblib")
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, path):
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")
    
    def _evaluate_on_noisy_data(self, noisy_x, y_test):
        """Evaluate CNN-SVM on noisy data"""
        y_pred = self.model.predict(noisy_x)
        return accuracy_score(y_test, y_pred)