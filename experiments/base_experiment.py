import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from datetime import datetime
from config.settings import config
from utils.logger import setup_logger
from utils.metrics import calculate_metrics
from utils.visualizer import plot_confusion_matrix, plot_training_history
from utils.noise import add_gaussian_noise


class BaseExperiment(ABC):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.logger = setup_logger(experiment_name, config.LOG_PATH)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_path = os.path.join(
            config.MODEL_SAVE_PATH, 
            f"{experiment_name}_{self.timestamp}"
        )
        os.makedirs(self.model_save_path, exist_ok=True)
        os.makedirs(config.RESULT_SAVE_PATH, exist_ok=True)
        
    @abstractmethod
    def train(self, train_data, val_data):
        pass
    
    @abstractmethod
    def evaluate(self, test_data):
        pass
    
    @abstractmethod
    def save_model(self):
        pass
    
    @abstractmethod
    def load_model(self, path):
        pass
    
    def generate_report(self, y_true, y_pred, prefix=""):
        """
        Generate evaluation report and visualizations:
        - Classification report saved to file
        - Confusion matrix plotted and saved
        - Metrics logged and returned
        """
        # Get classification report and metrics dict (includes confusion matrix)
        report, metrics = calculate_metrics(y_true, y_pred)
        
        # Save classification report to file
        report_path = os.path.join(
            config.RESULT_SAVE_PATH, 
            f"{prefix}classification_report.txt"
        )
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Log metrics for info/debug
        self.logger.info(f"Metrics:\n{metrics}")
        
        # Plot and save confusion matrix
        plot_confusion_matrix(
            metrics['confusion_matrix'], 
            save_path=os.path.join(
                config.RESULT_SAVE_PATH, 
                f"{prefix}confusion_matrix.png"
            )
        )
        
        return report, metrics
    
    def test_noise_robustness(self, test_data, sigmas=[0.01, 0.05, 0.1, 0.2, 0.3]):
        """
        Evaluate model performance under varying noise levels.
        Logs accuracy for each noise sigma and plots the robustness curve.
        """
        x_test, y_test = test_data
        results = {}
        
        for sigma in sigmas:
            noisy_x = add_gaussian_noise(x_test, sigma)
            acc = self._evaluate_on_noisy_data(noisy_x, y_test)
            results[sigma] = acc
            self.logger.info(f"Noise σ={sigma}: Accuracy = {acc:.4f}")
        
        # Plot noise robustness results
        self.plot_noise_robustness(results)
        return results
    
    def plot_noise_robustness(self, results):
        """
        Plot noise robustness curve.
        :param results: dict mapping noise sigma to accuracy
        """
        plt.figure(figsize=(10, 6))
        plt.plot(list(results.keys()), list(results.values()), 'bo-')
        plt.title(f"{self.experiment_name} Noise Robustness")
        plt.xlabel("Noise Sigma (σ)")
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig(os.path.join(
            config.RESULT_SAVE_PATH, 
            f"{self.experiment_name}_noise_robustness.png"
        ), dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
        plt.close()
    
    @abstractmethod
    def _evaluate_on_noisy_data(self, noisy_x, y_test):
        """
        Abstract method to evaluate the model on noisy data.
        Must be implemented by subclasses.
        """
        pass
