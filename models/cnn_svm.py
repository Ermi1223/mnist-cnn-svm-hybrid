import os
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from models.cnn import build_cnn_model
from config.settings import config

class CNN_SVM:
    def __init__(self, cnn_model=None, plot_architecture=True):
        if cnn_model is None:
            self.cnn = build_cnn_model()
            self.cnn = self._remove_last_layer(self.cnn)
        else:
            self.cnn = self._remove_last_layer(cnn_model)
            
        # Plot architecture if requested
        if plot_architecture:
            self.plot_model_architecture()
            
        self.svm = SVC(
            C=config.SVM_PARAMS['C'],
            gamma=config.SVM_PARAMS['gamma'],
            kernel=config.SVM_PARAMS['kernel'],
            probability=True,
            random_state=config.SEED
        )
        self.scaler = StandardScaler()
        
    def _remove_last_layer(self, model):
        """Remove the last layer of CNN for feature extraction"""
        return tf.keras.Model(
            inputs=model.input, 
            outputs=model.layers[-2].output
        )
    
    def plot_model_architecture(self):
        """Generate and save model architecture diagram"""
        os.makedirs(config.RESULT_SAVE_PATH, exist_ok=True)
        tf.keras.utils.plot_model(
            self.cnn,
            to_file=os.path.join(config.RESULT_SAVE_PATH, 'hybrid_architecture.png'),
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            dpi=config.PLOT_DPI,
            expand_nested=True
        )
    
    def extract_features(self, images):
        """Extract features using CNN"""
        return self.cnn.predict(images, verbose=0)
    
    def train_svm(self, features, labels):
        """Train SVM on extracted features"""
        features_scaled = self.scaler.fit_transform(features)
        self.svm.fit(features_scaled, labels)
    
    def predict(self, images):
        """Make predictions on new images"""
        features = self.extract_features(images)
        features_scaled = self.scaler.transform(features)
        return self.svm.predict(features_scaled)
    
    def predict_proba(self, images):
        """Get prediction probabilities"""
        features = self.extract_features(images)
        features_scaled = self.scaler.transform(features)
        return self.svm.predict_proba(features_scaled)