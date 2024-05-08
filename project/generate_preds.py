import os
from datetime import datetime
import argparse
import pandas as pd
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from torchvision.transforms import v2
import joblib
from torchmetrics.regression import R2Score

from train import cfg, val_transforms
from model import Model
from dataset import PlantTraitsDataset

    
def test_model(model, test_dataloader, output_scaler, device, label_names):
    print('Running validation')
    model.eval()
    R2 = R2Score(num_outputs=len(label_names), multioutput='uniform_average')

    pbar = tqdm(total=len(test_dataloader))
    with torch.no_grad():
        predictions_list = []
        y_true = []
        y_pred = []
        for sample in test_dataloader:
            img = sample['img'].to(device)
            ids = sample['id']
            features = sample['features'].to(device)
            preds = model(img, features)
            labels = sample['labels']

            preds = output_scaler.inverse_transform(preds.detach().cpu())
            preds = 10 ** preds

            R2.update(preds, labels)
            r2 = R2.compute().tolist()

            y_true.extend(labels.detach().cpu().reshape(-1).tolist())
            y_pred.extend(preds.reshape(-1).tolist())

            pbar.update()
            pbar.set_postfix(r2_score=r2)

            preds_list_i = []
            for idx, pred in enumerate(preds):
            # for idx, pred in enumerate(labels):
                d = {}
                d['id'] = ids[idx].long().item()
                for l, val in zip(label_names, pred):
                    d[l.removesuffix("_mean")] = val.item()
                preds_list_i.append(d)
            predictions_list.extend(preds_list_i)
        
        pbar.close()
        r2 = R2.compute().tolist()
        print(f'Validation R2 score: {r2}')
    return predictions_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    args = parser.parse_args()

    data_path = "./planttraits2024"
    test_csv = os.path.join(data_path, "test.csv")
    test_df = pd.read_csv(test_csv)

    test_df = test_df.astype(float)

    class_names = ['X4', 'X11', 'X18', 'X50', 'X26', 'X3112']
    num_classes = len(class_names)
    feature_names = test_df.columns[1:-1].tolist()
    num_features = len(feature_names)

    ref = pd.read_csv('./submission_ref.csv')
    test_df = pd.merge(test_df, ref, on='id')

    label_scaler_path = os.path.join(os.path.dirname(args.ckpt), 'label_scaler.save')
    label_scaler = joblib.load(label_scaler_path)

    feature_scaler_path = os.path.join(os.path.dirname(args.ckpt), 'feature_scaler.save')
    feature_scaler = joblib.load(feature_scaler_path)
    test_df.loc[:, feature_names] = feature_scaler.transform(test_df[feature_names])

    test_dataset = PlantTraitsDataset(test_df, class_names=class_names,
                                    feature_names=feature_names, transforms=val_transforms, split='test')
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=cfg.num_workers, batch_size=32)

    device = 'cuda' if torch.cuda.is_available() else 'mps'
    model = Model(num_classes=num_classes, num_features=num_features)

    print(f'Loading from {args.ckpt}')
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt)
    model.to(device)

    preds = test_model(model, test_dataloader, label_scaler, device, class_names)

    out_path = os.path.join(os.path.dirname(args.ckpt), 'submission.csv')
    df = pd.DataFrame(preds) 
    df.to_csv(out_path, index=False)
    print(f'Saved {out_path}')

    # print(ref.mean())
    print((ref - pd.read_csv(out_path)).abs().max())
    # import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()