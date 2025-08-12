
"""Script to evaluate trained models"""
import os
import argparse
import joblib
import numpy as np
from config.settings import config
from data.loader import get_data_loaders
from utils.metrics import calculate_metrics
from utils.visualizer import plot_confusion_matrix

def evaluate_model(model_path, test_data):
    """Evaluate a trained model"""
    # Load model
    if model_path.endswith('.h5'):
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
        model_type = 'cnn'
    else:
        model = joblib.load(model_path)
        model_type = 'svm' if 'svm' in model_path else 'hybrid'
    
    # Prepare test data
    x_test, y_test = test_data
    
    # Make predictions
    if model_type == 'cnn':
        y_pred = np.argmax(model.predict(x_test), axis=1)
    elif model_type == 'hybrid':
        y_pred = model.predict(x_test)
    else:  # SVM
        y_pred = model.predict(x_test)
    
    # Calculate metrics
    report, metrics = calculate_metrics(y_test, y_pred)
    
    # Print results
    print(f"\nEvaluation Results for {os.path.basename(model_path)}")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Plot confusion matrix
    model_name = os.path.basename(model_path).split('.')[0]
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        save_path=os.path.join(
            config.RESULT_SAVE_PATH, 
            f"{model_name}_confusion_matrix.png"
        )
    )
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("model_path", type=str, help="Path to trained model file")
    args = parser.parse_args()
    
    # Load data
    data = get_data_loaders()
    test_data = data['test']
    
    # Evaluate model
    evaluate_model(args.model_path, test_data)