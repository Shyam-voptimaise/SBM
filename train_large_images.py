#!/usr/bin/env python3
"""
YOLOv11 Training Script with Large Image Support
"""

import os
from ultralytics import YOLO
from pathlib import Path

def train_yolo_large_images():
    """
    Train YOLOv11 model with larger image size to preserve small defects
    """
    # Configuration
    model_name = "yolo11n.pt"
    data_config = "./configs/dataset.yaml"
    
    # Create experiment directory
    experiment_name = "experiment_v3_large_images"
    project_dir = "./runs/coil_defect_detection"
    
    print("="*60)
    print("YOLOv11 Training - Large Images for Small Defects")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Data config: {data_config}")
    print(f"Experiment: {experiment_name}")
    print(f"Image size: 1280x1280 (much larger for small defect detection)")
    
    # Initialize model
    model = YOLO(model_name)
    
    # Training parameters optimized for small defects
    training_params = {
        'data': data_config,
        'epochs': 100,
        'imgsz': 1280,  # Much larger image size
        'batch': 2,     # Smaller batch due to larger images
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,
        'device': 'cpu',
        'workers': 2,
        'project': project_dir,
        'name': experiment_name,
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'SGD',
        'verbose': True,
        'seed': 42,
        'deterministic': True,
        'single_cls': False,
        'rect': False,
        'cos_lr': True,
        'close_mosaic': 10,
        'resume': False,
        'amp': False,
        'fraction': 1.0,
        'profile': False,
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,
        'val': True,
        'lr0': 0.01,
        'lrf': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        'pose': 12.0,
        'kobj': 1.0,
        'label_smoothing': 0.0,
        'nbs': 64,
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 0.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    print("Training Parameters:")
    print(f"Image size: {training_params['imgsz']}")
    print(f"Batch size: {training_params['batch']}")
    print(f"Epochs: {training_params['epochs']}")
    print(f"Patience: {training_params['patience']}")
    
    try:
        # Start training
        print("\nStarting training...")
        results = model.train(**training_params)
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {project_dir}/{experiment_name}")
        
        # Validate the trained model
        print("\nRunning validation...")
        model_path = f"{project_dir}/{experiment_name}/weights/best.pt"
        model = YOLO(model_path)
        val_results = model.val(data=data_config)
        
        print("Validation completed!")
        return results, val_results
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        return None, None

if __name__ == "__main__":
    results, val_results = train_yolo_large_images()
    
    if results:
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        print("Training completed successfully!")
        print("Check the results in the runs/coil_defect_detection/experiment_v3_large_images directory")
        print("\nNext steps:")
        print("1. Test the model with inference")
        print("2. Evaluate detection performance")
        print("3. Adjust parameters if needed")
    else:
        print("Training failed. Please check the error messages above.")