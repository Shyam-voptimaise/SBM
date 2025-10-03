#!/usr/bin/env python3
"""
YOLOv11 Training Script
This script trains a YOLOv11 model on your custom dataset
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from ultralytics import YOLO
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from configs.config import TRAINING_CONFIG, MODEL_VARIANTS


def load_config(config_path):
    """Load training configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def train_yolo_model(
    model_variant="nano",
    dataset_config="./configs/dataset.yaml",
    training_config="./configs/training_config.yaml",
    project_name="yolo11_training",
    experiment_name="exp"
):
    """
    Train YOLOv11 model
    
    Args:
        model_variant (str): Model variant (nano, small, medium, large, extra_large)
        dataset_config (str): Path to dataset configuration YAML
        training_config (str): Path to training configuration YAML
        project_name (str): Project name for organizing runs
        experiment_name (str): Experiment name
    """
    
    # Check if dataset config exists
    if not os.path.exists(dataset_config):
        raise FileNotFoundError(f"Dataset config not found: {dataset_config}")
    
    # Load training configuration
    if os.path.exists(training_config):
        config = load_config(training_config)
        print(f"Loaded training config from {training_config}")
    else:
        config = TRAINING_CONFIG
        print("Using default training configuration")
    
    # Initialize model
    model_path = MODEL_VARIANTS.get(model_variant, "yolo11n.pt")
    print(f"Initializing YOLOv11 model: {model_path}")
    model = YOLO(model_path)
    
    # Check device
    device = config.get('device', 0)
    if device != 'cpu' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        device = 'cpu'
    
    print(f"Training device: {device}")
    print(f"Dataset config: {dataset_config}")
    print(f"Epochs: {config.get('epochs', 100)}")
    print(f"Batch size: {config.get('batch', 16)}")
    print(f"Image size: {config.get('imgsz', 640)}")
    
    # Train the model
    results = model.train(
        data=dataset_config,
        epochs=config.get('epochs', 100),
        batch=config.get('batch', 16),
        imgsz=config.get('imgsz', 640),
        device=device,
        workers=config.get('workers', 8),
        project=f"./runs/{project_name}",
        name=experiment_name,
        save_period=config.get('save_period', -1),
        patience=config.get('patience', 50),
        resume=config.get('resume', False),
        amp=config.get('amp', True),
        fraction=config.get('fraction', 1.0),
        profile=config.get('profile', False),
        freeze=config.get('freeze', None),
        lr0=config.get('lr0', 0.01),
        lrf=config.get('lrf', 0.01),
        momentum=config.get('momentum', 0.937),
        weight_decay=config.get('weight_decay', 0.0005),
        warmup_epochs=config.get('warmup_epochs', 3.0),
        warmup_momentum=config.get('warmup_momentum', 0.8),
        warmup_bias_lr=config.get('warmup_bias_lr', 0.1),
        box=config.get('box', 7.5),
        cls=config.get('cls', 0.5),
        dfl=config.get('dfl', 1.5),
        plots=config.get('plots', True),
        val=config.get('val', True)
    )
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Results saved to: ./runs/{project_name}/{experiment_name}")
    print(f"Best model saved as: ./runs/{project_name}/{experiment_name}/weights/best.pt")
    print("="*50)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv11 model')
    parser.add_argument('--model', type=str, default='nano', 
                       choices=['nano', 'small', 'medium', 'large', 'extra_large'],
                       help='Model variant to use')
    parser.add_argument('--dataset', type=str, default='./configs/dataset.yaml',
                       help='Path to dataset configuration YAML')
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml',
                       help='Path to training configuration YAML')
    parser.add_argument('--project', type=str, default='yolo11_training',
                       help='Project name for organizing runs')
    parser.add_argument('--name', type=str, default='exp',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('./runs', exist_ok=True)
    
    try:
        results = train_yolo_model(
            model_variant=args.model,
            dataset_config=args.dataset,
            training_config=args.config,
            project_name=args.project,
            experiment_name=args.name
        )
        
        print("\nTraining metrics:")
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                print(f"{key}: {value}")
                
    except Exception as e:
        print(f"Error during training: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()