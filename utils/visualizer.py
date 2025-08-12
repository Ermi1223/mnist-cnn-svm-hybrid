import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from config.settings import config

def plot_training_history(history, save_path):
    """Plot training and validation metrics history"""
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()

def plot_confusion_matrix(cm, save_path, classes=range(10)):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()

def plot_hybrid_performance(y_true, y_pred, save_path):
    """Visualize hybrid model performance"""
    # Create a comparison of correct vs incorrect predictions
    correct = (y_true == y_pred)
    incorrect = ~correct
    
    # Sample visualization
    plt.figure(figsize=(12, 6))
    
    # Plot accuracy per class
    class_acc = []
    for i in range(10):
        class_mask = (y_true == i)
        class_acc.append(np.mean(y_pred[class_mask] == i))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(10), class_acc)
    plt.title('Accuracy per Class')
    plt.xlabel('Digit Class')
    plt.ylabel('Accuracy')
    plt.ylim(0.9, 1.0)
    
    # Plot error distribution
    plt.subplot(1, 2, 2)
    error_counts = [np.sum((y_true == i) & incorrect) for i in range(10)]
    plt.bar(range(10), error_counts)
    plt.title('Error Count per Class')
    plt.xlabel('Digit Class')
    plt.ylabel('Error Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()

def plot_comparative_noise_robustness(results, save_path):
    """Compare noise robustness across models"""
    plt.figure(figsize=(10, 6))
    
    for model_name, model_results in results.items():
        sigmas = list(model_results.keys())
        accuracies = list(model_results.values())
        plt.plot(sigmas, accuracies, 'o-', label=model_name)
    
    plt.title("Comparative Noise Robustness")
    plt.xlabel("Noise Sigma (σ)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()

def plot_svm_performance(x_test, y_true, y_pred, save_path, num_samples=10):
    """Visualize SVM performance with examples"""
    # Get misclassified samples
    incorrect = np.where(y_true != y_pred)[0]
    correct = np.where(y_true == y_pred)[0]
    
    plt.figure(figsize=(15, 10))
    
    # Plot some correct predictions
    plt.subplot(2, 1, 1)
    plt.suptitle("SVM Performance Analysis", fontsize=16)
    plt.title("Correct Predictions")
    for i, idx in enumerate(correct[:num_samples]):
        plt.subplot(2, num_samples, i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap='gray')
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    
    # Plot some incorrect predictions
    plt.subplot(2, 1, 2)
    plt.title("Incorrect Predictions")
    for i, idx in enumerate(incorrect[:num_samples]):
        plt.subplot(2, num_samples, num_samples + i + 1)
        plt.imshow(x_test[idx].squeeze(), cmap='gray')
        plt.title(f"True: {y_true[idx]}\nPred: {y_pred[idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()

def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plot feature importance for SVM"""
    if not hasattr(model, 'coef_'):
        return
        
    # Get feature importances
    importances = np.abs(model.coef_[0])
    indices = np.argsort(importances)[::-1]
    
    # Plot top features
    plt.figure(figsize=(12, 8))
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[indices[:top_n]], align='center')
    plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=90)
    plt.xlim([-1, top_n])
    plt.tight_layout()
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()

def plot_accuracy_comparison(cnn_acc, cnn_svm_acc, svm_acc, save_path=None):
    models = ['CNN', 'CNN-SVM', 'SVM (Pixels)']
    accuracies = [cnn_acc, cnn_svm_acc, svm_acc]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
    plt.ylabel('Accuracy')
    plt.title('Model Comparison on MNIST Test Set')
    plt.ylim(0.9, 1.0)
    
    # Add accuracy labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}', ha='center', va='bottom')
    
    if save_path is None:
        save_path = os.path.join(config.RESULT_SAVE_PATH, 'accuracy_comparison.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=config.PLOT_DPI, format=config.PLOT_FORMAT)
    plt.close()