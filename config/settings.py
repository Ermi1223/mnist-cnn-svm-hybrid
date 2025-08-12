import os

class Config:
    # Data configuration
    DATA_PATH = "data/"
    IMG_SIZE = (28, 28)
    BATCH_SIZE = 128
    VAL_SPLIT = 0.2
    
    # Model hyperparameters
    CNN_PARAMS = {
        'filters': [32, 64],
        'kernel_size': (3, 3),
        'dense_units': 128,
        'dropout_rate': 0.25,
        'learning_rate': 0.001,
        'epochs': 15
    }
    
    SVM_PARAMS = {
        'C': 10,
        'gamma': 'scale',
        'kernel': 'rbf',
        'max_samples': 10000
    }
    
    # Experiment settings
    SEED = 42
    DEVICE = "cuda"  # Will be set dynamically
    
    # Paths
    ARTIFACT_PATH = "artifacts/"
    MODEL_SAVE_PATH = os.path.join(ARTIFACT_PATH, "models/")
    RESULT_SAVE_PATH = os.path.join(ARTIFACT_PATH, "results/")
    LOG_PATH = os.path.join(ARTIFACT_PATH, "logs/")
    
    # Visualization settings
    PLOT_FORMAT = 'png'
    PLOT_DPI = 300

config = Config()