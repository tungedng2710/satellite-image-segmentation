"""
Author: TUNG Ng.
"""
import os, cv2
import numpy as np
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import albumentations as album
import segmentation_models_pytorch as smp
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from utils import *
from configs import *
from dataset import MassachusettsDataset


if __name__ == "__main__":
    configs = Configs()
    x_train_dir = os.path.join(configs.data_dir, 'train')
    y_train_dir = os.path.join(configs.data_dir, 'train_labels')
    x_valid_dir = os.path.join(configs.data_dir, 'val')
    y_valid_dir = os.path.join(configs.data_dir, 'val_labels')
    x_test_dir = os.path.join(configs.data_dir, 'test')
    y_test_dir = os.path.join(configs.data_dir, 'test_labels')

    class_dict = pd.read_csv(configs.class_dict)
    class_names = class_dict['name'].tolist()
    class_rgb_values = class_dict[['r','g','b']].values.tolist()
    # Useful to shortlist specific classes in datasets with large number of classes
    select_classes = ['background', configs.target_objects]
    # Get RGB values of required classes
    select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
    select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

    print(f"Dataset: {configs.data_dir}")
    print(f"Class Names: {class_names}")
    print(f"Class RGB values: {class_rgb_values}")

    # Create segmentation model with pretrained encoder
    model = str_to_class(configs.model)(
            encoder_name=configs.encoder, 
            encoder_weights=configs.encoder_weights, 
            classes=len(class_names), 
            activation=configs.activation,
        )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {configs.model} with {configs.encoder} encoder")
    print(f"Total trainable params: {num_params:,}")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(configs.encoder, configs.encoder_weights)

    train_dataset = MassachusettsDataset(
        x_train_dir, y_train_dir, 
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    valid_dataset = MassachusettsDataset(
        x_valid_dir, y_valid_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    test_dataset = MassachusettsDataset(
        x_test_dir, y_test_dir, 
        augmentation=get_validation_augmentation(), 
        preprocessing=get_preprocessing(preprocessing_fn),
        class_rgb_values=select_class_rgb_values,
    )
    # Get data loaders
    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size_train, shuffle=True, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=configs.batch_size_val, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=configs.batch_size_val, shuffle=False, num_workers=8)

    # Training utilities
    loss = smp.losses.LovaszLoss(mode="binary") # Dice Loss, LovaszLoss, FocalLoss
    loss.__name__ = 'LovaszLoss'
    metrics = [
        smp.utils.metrics.IoU(eps=1., activation=None, threshold=0.5),
        smp.utils.metrics.Fscore(activation=None,),
        smp.utils.metrics.Accuracy()
    ]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
    lr_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=2, eta_min=5e-5,)
    if not os.path.exists("./runs/logs"):
        os.makedirs("./runs/logs")
    best_model_path = f'./runs/{configs.model}_{configs.encoder}_{configs.target_objects}.pth'
    if os.path.exists(best_model_path) and configs.pretrained:
        print(f"Pretrained model found at {best_model_path}. Load it now!")
        model = torch.load(best_model_path, map_location=configs.device)
    
    if configs.training:
        print("-------- Trainer profile --------")
        print(f"Num epochs: {configs.num_epochs}")
        print(f"Loss: {loss.__class__.__name__}")
        print(f"Optimizer: {optimizer.__class__.__name__}")
        print(f"Scheduler: {lr_scheduler.__class__.__name__}")
        print("---------------------------------")
        train_epoch = smp.utils.train.TrainEpoch(
            model, 
            loss=loss,
            metrics=metrics, 
            optimizer=optimizer,
            device=configs.device,
            verbose=True,
        )
        valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=configs.device,
            verbose=True,
        )
        best_iou = 0.0
        best_f1 = 0.0
        train_logs_list, valid_logs_list = [], []
        for epoch in range(0, configs.num_epochs):
            print(f'\nEpoch: {epoch+1}/{configs.num_epochs}')
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)
            train_logs_list.append(train_logs)
            valid_logs_list.append(valid_logs)

            best_result = (best_iou < valid_logs['iou_score']) \
                        and (best_f1 < valid_logs['fscore']) 
            if best_result:
                best_iou = valid_logs['iou_score']
                # best_f1 = valid_logs['fscore']s
                if configs.save:
                    torch.save(model, best_model_path)
                    print(f"Model saved at {best_model_path}")
    

    if configs.testing:
        print("\nEvaluating model on test set:")
        print(f"Loading model: {best_model_path}")
        if not configs.training:
            model = torch.load(best_model_path, map_location=configs.device)
        print("Runing test...")
        tester = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=configs.device,
            verbose=True,
        )
        test_logs = tester.run(test_loader)
        test_results = {
            "target": configs.target_objects,
            "dataset": {
                "images dir": x_test_dir,
                "masks dir": y_test_dir
            },
            "training": {
                "model_name": os.path.basename(best_model_path),
                "num_params": num_params,
                "loss": loss.__class__.__name__,
                "optimizer": optimizer.__class__.__name__,
                "lr_scheduler": lr_scheduler.__class__.__name__
            },
            "test results": {
                "IoU score": test_logs["iou_score"].item(),
                "F1 score": test_logs['fscore'].item(),
                "Accuracy": test_logs['accuracy'].item()
            }
        }
        yml_path = f"runs/logs/{os.path.basename(best_model_path).split('.')[0]}_log.yml"
        with open(yml_path, 'w') as outfile:
            yaml.dump(test_results, outfile)
            print(f"Test results is saved at {yml_path}")


    if configs.inference:
        print(f"Sample data path: {configs.sample_folder}")
        print(f"Segmenting {configs.target_objects}...")
        sample_preds_folder = f"data/{configs.model}_{configs.encoder}_predicts_{configs.target_objects}/"
        select_classes = ['background', configs.target_objects]
        # Get RGB values of required classes
        select_class_indices = [class_names.index(cls.lower()) for cls in select_classes]
        select_class_rgb_values =  np.array(class_rgb_values)[select_class_indices]

        if not os.path.exists(sample_preds_folder):
            os.makedirs(sample_preds_folder)
        print(f"Model: {best_model_path}")
        best_model = torch.load(best_model_path, map_location=configs.device)
        preprocessing_fn = smp.encoders.get_preprocessing_fn(configs.encoder, configs.encoder_weights)
        preprocessing = get_preprocessing(preprocessing_fn)
        augmentation=get_validation_augmentation()
        
        for filename in tqdm.tqdm(os.listdir(configs.sample_folder)):
            image_path = os.path.join(configs.sample_folder, filename)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            image_vis = crop_image(image.astype('uint8'))
            image = augmentation(image=image)["image"]
            image = preprocessing(image=image)["image"]
            image = torch.from_numpy(image).to(configs.device).unsqueeze(0)
            
            pred_mask = best_model(image)
            pred_mask = pred_mask.detach().squeeze().cpu().numpy()
            pred_mask = np.transpose(pred_mask,(1,2,0))
            pred_heatmap = pred_mask[:,:,select_classes.index(configs.target_objects)]
            pred_mask = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))

            save_name = f"predicted_{os.path.basename(image_path).split('.')[0]}.jpg"
            save_path = os.path.join(sample_preds_folder, save_name)
            if configs.save:
                cv2.imwrite(save_path, np.hstack([image_vis, pred_mask])[:,:,::-1])
            
        print(f"Done! Results've been seaved at {sample_preds_folder}")