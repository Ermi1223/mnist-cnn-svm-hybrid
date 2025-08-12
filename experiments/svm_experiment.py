import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score
from experiments.base_experiment import BaseExperiment
from models.svm import SVM
from utils.visualizer import plot_svm_performance
from config.settings import config

class SVMExperiment(BaseExperiment):
    def __init__(self):
        super().__init__("svm")
        self.model = SVM()
        
    def train(self, train_data, val_data):
        x_train, y_train = train_data
        self.model.train(x_train, y_train)
        return None
    
    def evaluate(self, test_data):
        x_test, y_test = test_data
        y_pred = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        report, _ = self.generate_report(y_test, y_pred, "svm_")
        
        self.logger.info(f"Test Accuracy: {accuracy:.4f}")
        self.logger.info(f"Classification Report:\n{report}")
        
        # Plot SVM performance
        plot_svm_performance(
            x_test, 
            y_test, 
            y_pred,
            save_path=os.path.join(config.RESULT_SAVE_PATH, "svm_performance.png")
        )
        
        return accuracy, report
    
    def save_model(self):
        model_path = os.path.join(self.model_save_path, "svm_model.joblib")
        self.model.save(model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, path):
        self.model.load(path)
        self.logger.info(f"Model loaded from {path}")
    
    def _evaluate_on_noisy_data(self, noisy_x, y_test):
        """Evaluate SVM on noisy data"""
        y_pred = self.model.predict(noisy_x)
        return accuracy_score(y_test, y_pred)