import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from config.settings import config

class SVM:
    def __init__(self):
        self.model = SVC(
            C=config.SVM_PARAMS['C'],
            gamma=config.SVM_PARAMS['gamma'],
            kernel=config.SVM_PARAMS['kernel'],
            probability=True,
            random_state=config.SEED
        )
        self.scaler = StandardScaler()
        self.trained = False
        
    def train(self, x_train, y_train):
        """Train SVM model"""
        # Flatten images
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        
        # Scale features
        x_train_scaled = self.scaler.fit_transform(x_train_flat)
        
        # Use subset if specified
        if config.SVM_PARAMS['max_samples'] < len(x_train):
            indices = np.random.choice(
                len(x_train), 
                config.SVM_PARAMS['max_samples'], 
                replace=False
            )
            x_train_sampled = x_train_scaled[indices]
            y_train_sampled = y_train[indices]
        else:
            x_train_sampled = x_train_scaled
            y_train_sampled = y_train
            
        # Train model
        self.model.fit(x_train_sampled, y_train_sampled)
        self.trained = True
        return self
    
    def predict(self, images):
        """Make predictions on new images"""
        if not self.trained:
            raise RuntimeError("Model not trained yet")
            
        images_flat = images.reshape(images.shape[0], -1)
        images_scaled = self.scaler.transform(images_flat)
        return self.model.predict(images_scaled)
    
    def predict_proba(self, images):
        """Get prediction probabilities"""
        if not self.trained:
            raise RuntimeError("Model not trained yet")
            
        images_flat = images.reshape(images.shape[0], -1)
        images_scaled = self.scaler.transform(images_flat)
        return self.model.predict_proba(images_scaled)
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'trained': self.trained
        }, path)
    
    def load(self, path):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.trained = data['trained']
        return self