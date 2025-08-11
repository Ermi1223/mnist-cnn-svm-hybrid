import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, ConfusionMatrixDisplay)
from joblib import dump, load
import time
import os

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Data Preparation ---
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and reshape for CNN
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # One-hot encode labels for CNN
    y_train_cnn = utils.to_categorical(y_train, 10)
    
    return (x_train, y_train, y_train_cnn), (x_test, y_test)

# --- CNN Model ---
def build_and_train_cnn(x_train, y_train_cnn, x_test, y_test):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train model
    history = model.fit(x_train, y_train_cnn,
                        epochs=15,
                        batch_size=128,
                        validation_split=0.2,
                        verbose=1)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, utils.to_categorical(y_test, 10))
    print(f"\nCNN Test Accuracy: {test_acc:.4f}")
    
    # Save model and training history
    model.save('models/cnn_model.h5')
    return model, history, test_acc

# --- CNN-SVM Hybrid ---
def cnn_svm_hybrid(cnn_model, x_train, y_train, x_test, y_test):
    # Create feature extractor (remove last layer)
    feature_extractor = models.Model(
        inputs=cnn_model.input,
        outputs=cnn_model.layers[-2].output  # Last dense layer before softmax
    )
    
    # Extract features
    train_features = feature_extractor.predict(x_train)
    test_features = feature_extractor.predict(x_test)
    
    # Scale features
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # Train SVM with RBF kernel
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    start_time = time.time()
    svm.fit(train_features_scaled, y_train)
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = svm.predict(test_features_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nCNN-SVM Test Accuracy: {accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Save model
    dump(svm, 'models/cnn_svm_model.joblib')
    return svm, accuracy, y_pred

# --- SVM (Direct on Pixels) ---
def train_svm_direct(x_train, y_train, x_test, y_test, sample_size=10000):
    # Flatten images
    x_train_flat = x_train.reshape(len(x_train), -1)
    x_test_flat = x_test.reshape(len(x_test), -1)
    
    # Scale
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train_flat)
    x_test_scaled = scaler.transform(x_test_flat)
    
    # Use subset for training (full dataset is too slow)
    if sample_size < len(x_train):
        indices = np.random.choice(len(x_train), sample_size, replace=False)
        x_train_sampled = x_train_scaled[indices]
        y_train_sampled = y_train[indices]
    else:
        x_train_sampled = x_train_scaled
        y_train_sampled = y_train
    
    # Train SVM
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
    
    start_time = time.time()
    svm.fit(x_train_sampled, y_train_sampled)
    training_time = time.time() - start_time
    
    # Evaluate
    y_pred = svm.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nSVM (Pixels) Test Accuracy: {accuracy:.4f}")
    print(f"Training Time ({sample_size} samples): {training_time:.2f} seconds")
    
    # Save model
    dump(svm, 'models/svm_model.joblib')
    return svm, accuracy, y_pred

# --- Visualization and Reporting ---
def generate_reports(cnn_history, cnn_acc, cnn_svm_acc, svm_acc, 
                     y_test, cnn_pred, cnn_svm_pred, svm_pred):
    # Accuracy comparison
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
    
    plt.savefig('results/accuracy_comparison.png')
    
    # CNN training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
    plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('CNN Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(cnn_history.history['loss'], label='Train Loss')
    plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
    plt.title('CNN Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/cnn_training_history.png')
    
    # Classification reports
    with open('results/classification_report.txt', 'w') as f:
        f.write("CNN Classification Report:\n")
        f.write(classification_report(y_test, cnn_pred))
        
        f.write("\n\nCNN-SVM Classification Report:\n")
        f.write(classification_report(y_test, cnn_svm_pred))
        
        f.write("\n\nSVM (Pixels) Classification Report:\n")
        f.write(classification_report(y_test, svm_pred))
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    cm = confusion_matrix(y_test, cnn_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[0], cmap='Blues')
    axes[0].set_title('CNN Confusion Matrix')
    
    cm = confusion_matrix(y_test, cnn_svm_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[1], cmap='Greens')
    axes[1].set_title('CNN-SVM Confusion Matrix')
    
    cm = confusion_matrix(y_test, svm_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=axes[2], cmap='Oranges')
    axes[2].set_title('SVM (Pixels) Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('results/confusion_matrices.png')

# --- Main Execution ---
def main():
    # Load and preprocess data
    (x_train, y_train, y_train_cnn), (x_test, y_test) = load_and_preprocess_data()
    
    # Train CNN
    print("\n" + "="*50)
    print("Training CNN Model")
    print("="*50)
    cnn_model, cnn_history, cnn_acc = build_and_train_cnn(
        x_train, y_train_cnn, x_test, y_test
    )
    cnn_pred = np.argmax(cnn_model.predict(x_test), axis=1)
    
    # Train CNN-SVM Hybrid
    print("\n" + "="*50)
    print("Training CNN-SVM Hybrid Model")
    print("="*50)
    cnn_svm, cnn_svm_acc, cnn_svm_pred = cnn_svm_hybrid(
        cnn_model, x_train, y_train, x_test, y_test
    )
    
    # Train SVM on Pixels
    print("\n" + "="*50)
    print("Training SVM on Raw Pixels (10,000 sample subset)")
    print("="*50)
    svm_model, svm_acc, svm_pred = train_svm_direct(
        x_train, y_train, x_test, y_test, sample_size=10000
    )
    
    # Generate reports and visualizations
    generate_reports(cnn_history, cnn_acc, cnn_svm_acc, svm_acc,
                     y_test, cnn_pred, cnn_svm_pred, svm_pred)
    
    # Final summary
    print("\n" + "="*50)
    print("Final Results Summary")
    print("="*50)
    print(f"CNN Test Accuracy: {cnn_acc:.4f}")
    print(f"CNN-SVM Test Accuracy: {cnn_svm_acc:.4f}")
    print(f"SVM (Pixels) Test Accuracy: {svm_acc:.4f}")
    print("\nVisualizations and reports saved in 'results' directory")

if __name__ == "__main__":
    main()