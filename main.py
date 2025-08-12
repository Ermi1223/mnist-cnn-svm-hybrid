import os
import sys
import argparse
import joblib
import numpy as np
from config.settings import config
from data.loader import get_data_loaders
from experiments.cnn_experiment import CNNExperiment
from experiments.hybrid_experiment import HybridExperiment
from experiments.svm_experiment import SVMExperiment
from utils.visualizer import plot_comparative_noise_robustness, plot_accuracy_comparison

def setup_directories():
    """Create required artifact directories"""
    os.makedirs(config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(config.RESULT_SAVE_PATH, exist_ok=True)
    os.makedirs(config.LOG_PATH, exist_ok=True)

def main():
    setup_directories()
    
    # Load data
    data = get_data_loaders()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MNIST Classification Experiments")
    parser.add_argument('--model', type=str, choices=['all', 'cnn', 'hybrid', 'svm'], 
                        default='all', help="Model(s) to run")
    parser.add_argument('--train', action='store_true', help="Train models")
    parser.add_argument('--evaluate', action='store_true', help="Evaluate models")
    parser.add_argument('--cnn_model', type=str, default=None, 
                        help="Path to pre-trained CNN model for hybrid")
    args = parser.parse_args()
    
    # Prepare experiments and results containers
    experiments = []
    all_noise_results = {}
    accuracies = {}  # <-- initialize here
    
    if args.model in ['all', 'cnn']:
        experiments.append(CNNExperiment())
        
    if args.model in ['all', 'hybrid']:
        experiments.append(HybridExperiment(cnn_model_path=args.cnn_model))
        
    if args.model in ['all', 'svm']:
        experiments.append(SVMExperiment())
    
    for experiment in experiments:
        print(f"\n{'='*50}")
        print(f"Running {experiment.experiment_name.upper()} experiment")
        print(f"{'='*50}")
        
        if args.train:
            print("Training model...")
            experiment.train(
                train_data=data['train'],
                val_data=data['val']
            )
            experiment.save_model()
            
        if args.evaluate:
            print("Evaluating model...")
            test_acc, _ = experiment.evaluate(test_data=data['test'])  # unpack accuracy, report
            accuracies[experiment.experiment_name] = test_acc  # store accuracy
            
            print("Testing noise robustness...")
            noise_results = experiment.test_noise_robustness(data['test'])
            all_noise_results[experiment.experiment_name] = noise_results
    
    # Compare noise robustness across models
    if args.model == 'all' and args.evaluate:
        plot_comparative_noise_robustness(
            all_noise_results,
            os.path.join(config.RESULT_SAVE_PATH, "comparative_noise_robustness.png")
        )
    
    # Plot accuracy comparison bar chart
    if args.evaluate:
        cnn_acc = accuracies.get('cnn', 0)
        cnn_svm_acc = accuracies.get('hybrid', 0)
        svm_acc = accuracies.get('svm', 0)
        
        plot_accuracy_comparison(cnn_acc, cnn_svm_acc, svm_acc)
        print(f"Saved accuracy comparison plot to {os.path.join(config.RESULT_SAVE_PATH, 'accuracy_comparison.png')}")

if __name__ == "__main__":
    main()
