
"""Script to train individual models"""
import os
import argparse
from config.settings import config
from data.loader import get_data_loaders
from models.cnn import build_cnn_model
from models.svm import SVM
from models.cnn_svm import CNN_SVM

def train_cnn(train_data, val_data):
    """Train CNN model"""
    print("Training CNN model...")
    model = build_cnn_model()
    
    history = model.fit(
        train_data[0], train_data[1],
        epochs=config.CNN_PARAMS['epochs'],
        batch_size=config.BATCH_SIZE,
        validation_data=val_data,
        verbose=1
    )
    
    # Save model
    model_path = os.path.join(config.MODEL_SAVE_PATH, "cnn_model.h5")
    model.save(model_path)
    print(f"CNN model saved to {model_path}")
    
    return model, history

def train_svm(train_data):
    """Train SVM model"""
    print("Training SVM model...")
    model = SVM()
    model.train(train_data[0], train_data[1])
    
    # Save model
    model_path = os.path.join(config.MODEL_SAVE_PATH, "svm_model.joblib")
    model.save(model_path)
    print(f"SVM model saved to {model_path}")
    
    return model

def train_hybrid(train_data, cnn_model_path=None):
    """Train CNN-SVM hybrid model"""
    print("Training CNN-SVM hybrid model...")
    
    # Load or train CNN
    if cnn_model_path and os.path.exists(cnn_model_path):
        import tensorflow as tf
        cnn_model = tf.keras.models.load_model(cnn_model_path)
        model = CNN_SVM(cnn_model=cnn_model)
    else:
        model = CNN_SVM()
    
    # Extract features
    train_features = model.extract_features(train_data[0])
    
    # Train SVM
    model.train_svm(train_features, train_data[1])
    
    # Save model
    model_path = os.path.join(config.MODEL_SAVE_PATH, "hybrid_model.joblib")
    joblib.dump(model, model_path)
    print(f"Hybrid model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train specific models")
    parser.add_argument("model_type", choices=["cnn", "svm", "hybrid"], 
                        help="Type of model to train")
    parser.add_argument("--cnn_model", type=str, default=None,
                        help="Path to pre-trained CNN model for hybrid")
    args = parser.parse_args()
    
    # Load data
    data = get_data_loaders()
    train_data = data['train']
    val_data = data['val']
    
    # Train selected model
    if args.model_type == "cnn":
        train_cnn(train_data, val_data)
    elif args.model_type == "svm":
        train_svm(train_data)
    elif args.model_type == "hybrid":
        train_hybrid(train_data, args.cnn_model)