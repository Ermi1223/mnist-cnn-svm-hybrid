import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report
)

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics for classification.
    
    Args:
        y_true (array-like): True class labels.
        y_pred (array-like): Predicted class labels.
        
    Returns:
        report (str): Text classification report (precision, recall, f1 per class).
        metrics (dict): Dictionary with accuracy, macro metrics, per-class metrics, and confusion matrix.
    """
    # Classification report as string
    report = classification_report(
        y_true, 
        y_pred,
        target_names=[str(i) for i in range(10)]
    )
    
    # Confusion matrix (2D array)
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class precision, recall, f1 (arrays of shape [num_classes])
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    # Metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'per_class': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'confusion_matrix': cm
    }
    
    return report, metrics
