import os
from datetime import datetime
import argparse
import ipdb.stdout
import pandas as pd
from torch.utils.data import DataLoader
import torch
from torch import optim, nn
from tqdm import tqdm
from torchvision.transforms import v2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import joblib
from torchmetrics.regression import R2Score
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import PlantTraitsDataset
from model import Model

MEAN = [
    0.48145466,
    0.4578275,
    0.40821073]

STD = [
    0.26862954,
    0.26130258,
    0.27577711]

train_transforms = v2.Compose([
    v2.Resize((448, 448)),
    v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    v2.RandomApply([v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)], p=0.5),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=5),
    v2.Normalize(mean=MEAN, std=STD),
])

val_transforms = v2.Compose([
    v2.Resize((448, 448)),
    v2.Normalize(mean=MEAN, std=STD),
])


class cfg:
    num_workers = 16
    batch_size = 16
    num_epochs = 30
    weight_decay = 0.01
    lr = 1e-4


BEST_R2 = -1000000000


def upsample_bins(df, columns, num_bins=10):
    """
    Upsamples the bins of specified continuous columns in the DataFrame.
    """
    binned_df = df.copy()
    for col in columns:
        if isinstance(num_bins, dict):
            bins = num_bins.get(col, 10)  # Use specified bins or default to 10
        else:
            bins = num_bins
        binned_df[col] = pd.cut(df[col], bins=bins, labels=False, retbins=False)

    upsampled_df = None
    max_size = -1
    # Find the maximum bin size
    for col in columns:
        groups = df.groupby(binned_df[col])
        max_size = max(groups.size().max(), max_size)

    for col in columns:
        groups = df.groupby(binned_df[col])
        # Upsample bins to the maximum size
        upsampled_groups = [group.sample(max_size, replace=True) for name, group in groups]
        upsampled_column = pd.concat(upsampled_groups).sort_index()

        if upsampled_df is None:
            upsampled_df = upsampled_column
        else:
            upsampled_df = pd.concat([upsampled_df, upsampled_column], axis=0)

    return upsampled_df


def validate_model(model, criterion, val_dataloader, output_scaler, device,
                   writer, itr):
    print('Running validation')
    model.eval()
    pbar = tqdm(total=len(val_dataloader))
    R2 = R2Score(num_outputs=6, multioutput='uniform_average')
    R2_class_wise = R2Score(num_outputs=6, multioutput='raw_values')

    with torch.no_grad():
        for sample in val_dataloader:
            img = sample['img'].to(device)
            labels = sample['labels'].to(device)
            features = sample['features'].to(device)
            preds = model(img, features)
            loss = criterion(preds, labels)

            labels = 10 ** output_scaler.inverse_transform(labels.detach().cpu())
            preds = 10 ** output_scaler.inverse_transform(preds.detach().cpu())

            R2.update(preds, torch.from_numpy(labels))
            r2 = R2.compute().item()

            R2_class_wise.update(preds, torch.from_numpy(labels))
            r2_cls = R2_class_wise.compute().tolist()

            cls_wise = " ".join([f"{m:.2f}" for m in r2_cls])
            pbar.update()
            pbar.set_postfix(loss=loss.item(), r2_score=r2, r2_cls=cls_wise)

        pbar.close()
        r2 = R2.compute().item()
        print(f'Validation R2 score: {r2:.4f}')
        writer.add_scalar("R2/val", r2, itr)
        writer.flush()
    return r2


def train_epoch(model, ema_model, criterion, optimizer, train_dataloader,
                val_dataloader, lr_scheduler,
                output_scaler, device, writer, itr, log_dir):
    model.train()
    pbar = tqdm(total=len(train_dataloader))
    R2 = R2Score(num_outputs=6, multioutput='uniform_average')

    for sample in train_dataloader:
        img = sample['img'].to(device)
        labels = sample['labels'].to(device)
        features = sample['features'].to(device)

        preds = model(img, features)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        ema_model.update_parameters(model)

        y_true = 10 ** output_scaler.inverse_transform(labels.detach().cpu())
        y_pred = 10 ** output_scaler.inverse_transform(preds.detach().cpu())

        R2.update(y_pred, torch.from_numpy(y_true))
        r2 = R2.compute().item()
        pbar.update()
        pbar.set_postfix(loss=loss.item(), r2_score=r2)

        itr += 1
        writer.add_scalar("Loss/train", loss.item(), itr)
        writer.add_scalar("R2/train", r2, itr)
        writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], itr)
        writer.flush()

        if itr % 2000 == 1999:
            r2 = validate_model(ema_model, criterion, val_dataloader,
                                output_scaler, device, writer, itr)
            global BEST_R2
            if r2 > BEST_R2:
                torch.save(model.state_dict(), f'{log_dir}/model.pt')
                torch.save(model.state_dict(), f'{log_dir}/ema_model.pt')
                BEST_R2 = r2

    pbar.close()
    return itr


def train(model, criterion, optimizer, train_dataloader, val_dataloader,
          num_epochs, output_scaler, device, log_dir, writer):
    curr_itr = 0
    max_itr = 100_000
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=max_itr, eta_min=1e-8)
    ema_model = torch.optim.swa_utils.AveragedModel(model,
                                                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    for epoch in range(num_epochs):
        print(f'Epoch: [{epoch + 1}/{num_epochs}]')
        curr_itr = train_epoch(model, ema_model, criterion, optimizer,
                               train_dataloader, val_dataloader, lr_scheduler,
                               output_scaler, device, writer, curr_itr, log_dir)
        if curr_itr > max_itr:
            break
    print("Training complete!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_ckpt', type=str)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    current_time = "{:%Y_%m_%d_%H_%M}".format(datetime.now())
    if args.debug is True:
        log_dir = f'./logs/debug'
    else:
        log_dir = f'./logs/{current_time}'

    os.makedirs(f'{log_dir}', exist_ok=True)
    print(f'Logging to {log_dir}')

    writer = SummaryWriter(log_dir)

    data_path = "./planttraits2024"
    train_csv = os.path.join(data_path, "train.csv")
    df = pd.read_csv(train_csv)

    # Convert to float
    df = df.astype(float)

    test_csv = os.path.join(data_path, "test.csv")
    test_df = pd.read_csv(test_csv)
    class_names = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    num_classes = len(class_names)
    feature_names = test_df.columns[1:].tolist()
    num_features = len(feature_names)

    print(f'Shape before outlier removal: {df.shape}')

    # Remove outliers
    low = df[class_names].quantile(0.005)
    high = df[class_names].quantile(0.985)
    outliers = ((df[class_names] < low) | (df[class_names] > high))
    df = df[~(outliers.any(axis=1))]

    print(f'Shape after outlier removal: {df.shape}')

    # Log transform.
    df.loc[:, class_names] = np.log10(df[class_names].values)

    # Train val split.
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

    # Normalize targets.
    label_scaler = preprocessing.StandardScaler()
    train_df.loc[:, class_names] = label_scaler.fit_transform(train_df[class_names])
    val_df.loc[:, class_names] = label_scaler.transform(val_df[class_names])
    scaler_filename = f"{log_dir}/label_scaler.save"
    joblib.dump(label_scaler, scaler_filename)

    # Normalize features.
    feature_scaler = preprocessing.StandardScaler()
    train_df.loc[:, feature_names] = feature_scaler.fit_transform(train_df[feature_names])
    val_df.loc[:, feature_names] = feature_scaler.transform(val_df[feature_names])
    scaler_filename = f"{log_dir}/feature_scaler.save"
    joblib.dump(feature_scaler, scaler_filename)

    # Upsample based on bins
    # print(f'Train shape before upsampling: {train_df.shape}')
    # train_df = upsample_bins(train_df, class_names)
    # print(f'Train shape after upsampling: {train_df.shape}')

    # Build dataloader.
    train_dataset = PlantTraitsDataset(train_df, class_names=class_names,
                                       feature_names=feature_names, transforms=train_transforms)
    val_dataset = PlantTraitsDataset(val_df, class_names=class_names,
                                     feature_names=feature_names, transforms=val_transforms)

    print(f'train dataset size: {len(train_dataset)}')
    print(f'val dataset size: {len(val_dataset)}')

    train_dataloader = DataLoader(dataset=train_dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, num_workers=cfg.num_workers, batch_size=cfg.batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'mps'
    model = Model(num_classes=num_classes, num_features=num_features, model_type='swin')

    if args.resume_ckpt is not None:
        print(f'Resuming from {args.resume_ckpt}')
        ckpt = torch.load(args.resume_ckpt)
        model.load_state_dict(ckpt)

    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.SmoothL1Loss()

    train(model, criterion, optimizer, train_dataloader, val_dataloader,
          cfg.num_epochs, label_scaler, device=device, log_dir=log_dir, writer=writer)


if __name__ == '__main__':
    main()
