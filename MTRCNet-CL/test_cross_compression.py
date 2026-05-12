import os
import time
import pickle
import argparse
import glob
import csv

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn import DataParallel
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import Lambda

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

parser = argparse.ArgumentParser(description='cross-compression lstm testing')

# Defaults set to match your training setup
parser.add_argument('-g', '--gpu', default=[0, 1], nargs='+', type=int, help='index of gpu to use')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length')
parser.add_argument('-t', '--test', default=800, type=int, help='test batch size')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 center, 5 five_crop, 10 ten_crop')
parser.add_argument('--num_classes', default=7, type=int, help='number of phase classes')

parser.add_argument('--data_dir', default='/home/ahallur1/DL/Vani/MTRCNet-CL',
                    type=str, help='directory containing train_val_test_paths_labels_<compression>.pkl')
parser.add_argument('--checkpoint_root', default='checkpoints_phase',
                    type=str, help='root directory containing per-compression trained model folders')
parser.add_argument('--save_dir', default='cross_compression_test_outputs',
                    type=str, help='directory to save outputs')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg

sequence_length = args.seq
test_batch_size = args.test
workers = args.work
crop_type = args.crop
num_classes = args.num_classes
data_dir = args.data_dir
checkpoint_root = args.checkpoint_root
save_dir = args.save_dir

compression_levels = ['original', 'CRF23', 'CRF28', 'CRF35', 'CRF51']

os.makedirs(save_dir, exist_ok=True)

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

print('number of gpu   : {:6d}'.format(num_gpu), flush=True)
print('sequence length : {:6d}'.format(sequence_length), flush=True)
print('test batch size : {:6d}'.format(test_batch_size), flush=True)
print('num of workers  : {:6d}'.format(workers), flush=True)
print('test crop type  : {:6d}'.format(crop_type), flush=True)
print('data dir        : {}'.format(data_dir), flush=True)
print('checkpoint root : {}'.format(checkpoint_root), flush=True)
print('save dir        : {}'.format(save_dir), flush=True)
print('compression lvls: {}'.format(compression_levels), flush=True)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class CholecDataset(Dataset):
    def __init__(self, file_paths, file_labels, transform=None, loader=pil_loader):
        self.file_paths = file_paths
        self.file_labels_1 = file_labels[:, range(7)]
        self.file_labels_2 = file_labels[:, -1]
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.file_paths[index]
        labels_1 = self.file_labels_1[index]
        labels_2 = self.file_labels_2[index]
        img = self.loader(img_name)
        if self.transform is not None:
            img = self.transform(img)
        return img, labels_1, labels_2

    def __len__(self):
        return len(self.file_paths)


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()

        try:
            resnet = models.resnet50(weights=None)
        except TypeError:
            resnet = models.resnet50(pretrained=False)

        self.share = torch.nn.Sequential()
        self.share.add_module("conv1", resnet.conv1)
        self.share.add_module("bn1", resnet.bn1)
        self.share.add_module("relu", resnet.relu)
        self.share.add_module("maxpool", resnet.maxpool)
        self.share.add_module("layer1", resnet.layer1)
        self.share.add_module("layer2", resnet.layer2)
        self.share.add_module("layer3", resnet.layer3)
        self.share.add_module("layer4", resnet.layer4)
        self.share.add_module("avgpool", resnet.avgpool)

        self.lstm = nn.LSTM(2048, 512, batch_first=True)
        self.fc = nn.Linear(512, 7)

        init.xavier_normal_(self.lstm.all_weights[0][0])
        init.xavier_normal_(self.lstm.all_weights[0][1])
        init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        x = self.share(x)
        x = x.view(-1, 2048)
        x = x.view(-1, sequence_length, 2048)
        self.lstm.flatten_parameters()
        y, _ = self.lstm(x)
        y = y.contiguous().view(-1, 512)
        y = self.fc(y)
        return y


def get_useful_start_idx(sequence_length, list_each_length):
    count = 0
    idx = []
    for i in range(len(list_each_length)):
        for j in range(count, count + (list_each_length[i] + 1 - sequence_length)):
            idx.append(j)
        count += list_each_length[i]
    return idx


def get_data(data_path):
    with open(data_path, 'rb') as f:
        train_test_paths_labels = pickle.load(f)

    train_paths = train_test_paths_labels[0]
    val_paths = train_test_paths_labels[1]
    test_paths = train_test_paths_labels[2]
    train_labels = train_test_paths_labels[3]
    val_labels = train_test_paths_labels[4]
    test_labels = train_test_paths_labels[5]
    train_num_each = train_test_paths_labels[6]
    val_num_each = train_test_paths_labels[7]
    test_num_each = train_test_paths_labels[8]

    print('train_paths  : {:6d}'.format(len(train_paths)), flush=True)
    print('train_labels : {:6d}'.format(len(train_labels)), flush=True)
    print('valid_paths  : {:6d}'.format(len(val_paths)), flush=True)
    print('valid_labels : {:6d}'.format(len(val_labels)), flush=True)
    print('test_paths   : {:6d}'.format(len(test_paths)), flush=True)
    print('test_labels  : {:6d}'.format(len(test_labels)), flush=True)

    train_labels = np.asarray(train_labels, dtype=np.int64)
    val_labels = np.asarray(val_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)

    if crop_type == 0:
        test_transforms = transforms.Compose([
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 1:
        test_transforms = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    elif crop_type == 5:
        test_transforms = transforms.Compose([
            transforms.FiveCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(lambda crops: torch.stack([
                transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop)
                for crop in crops
            ]))
        ])
    elif crop_type == 10:
        test_transforms = transforms.Compose([
            transforms.TenCrop(224),
            Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            Lambda(lambda crops: torch.stack([
                transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])(crop)
                for crop in crops
            ]))
        ])
    else:
        raise ValueError("Unsupported crop_type: {}".format(crop_type))

    train_dataset = CholecDataset(train_paths, train_labels, test_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def load_checkpoint(model, checkpoint_path, device):
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    try:
        model.load_state_dict(checkpoint)
        return model
    except RuntimeError:
        new_checkpoint = {}
        has_module_prefix = any(k.startswith('module.') for k in checkpoint.keys())
        model_is_dataparallel = isinstance(model, DataParallel)

        if has_module_prefix and not model_is_dataparallel:
            for k, v in checkpoint.items():
                new_checkpoint[k.replace('module.', '', 1)] = v
        elif (not has_module_prefix) and model_is_dataparallel:
            for k, v in checkpoint.items():
                new_checkpoint['module.' + k] = v
        else:
            raise

        model.load_state_dict(new_checkpoint)
        return model


def find_final_model_for_compression(train_compression):
    compression_dir = os.path.join(checkpoint_root, train_compression)
    if not os.path.isdir(compression_dir):
        return None

    candidates = []
    for p in glob.glob(os.path.join(compression_dir, '*.pth')):
        base = os.path.basename(p)
        if 'checkpoint_epoch_' in base:
            continue
        candidates.append(p)

    if not candidates:
        return None

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def save_confusion_matrix(cm, out_path):
    np.savetxt(out_path, cm, fmt='%d', delimiter=',')


def save_metrics_text(metrics_dict, report_text, out_path):
    with open(out_path, 'w') as f:
        f.write('Overall Metrics\n')
        f.write('===============\n')
        for k, v in metrics_dict.items():
            f.write(f'{k}: {v}\n')
        f.write('\nClassification Report\n')
        f.write('=====================\n')
        f.write(report_text)


def evaluate_model_on_dataset(model_path, test_dataset, test_num_each, train_compression, test_compression):
    num_test = len(test_dataset)
    test_useful_start_idx = get_useful_start_idx(sequence_length, test_num_each)

    num_test_we_use = len(test_useful_start_idx)
    test_we_use_start_idx = test_useful_start_idx[0:num_test_we_use]

    test_idx = []
    for i in range(num_test_we_use):
        for j in range(sequence_length):
            test_idx.append(test_we_use_start_idx[i] + j)

    num_test_all = len(test_idx)

    print('num test start idx : {:6d}'.format(len(test_useful_start_idx)), flush=True)
    print('last idx test start: {:6d}'.format(test_useful_start_idx[-1]), flush=True)
    print('num of test dataset: {:6d}'.format(num_test), flush=True)
    print('num of test we use : {:6d}'.format(num_test_we_use), flush=True)
    print('num of all test use: {:6d}'.format(num_test_all), flush=True)

    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_idx,
        num_workers=workers,
        pin_memory=use_gpu
    )

    model = resnet_lstm().to(device)

    if num_gpu > 1:
        model = DataParallel(model)

    model = load_checkpoint(model, model_path, device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')

    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            inputs, labels_1, labels_2 = data
            labels_2 = labels_2[(sequence_length - 1)::sequence_length]

            inputs = inputs.to(device, non_blocking=use_gpu)
            labels = labels_2.to(device, non_blocking=use_gpu)

            if crop_type == 0 or crop_type == 1:
                outputs = model(inputs)
            elif crop_type == 5:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs = model(inputs)
                outputs = outputs.view(5, -1, num_classes)
                outputs = torch.mean(outputs, 0)
            elif crop_type == 10:
                inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                inputs = inputs.view(-1, 3, 224, 224)
                outputs = model(inputs)
                outputs = outputs.view(10, -1, num_classes)
                outputs = torch.mean(outputs, 0)
            else:
                raise ValueError("Unsupported crop_type: {}".format(crop_type))

            outputs = outputs[sequence_length - 1::sequence_length]
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            test_loss += loss.item()
            test_corrects += torch.sum(preds == labels).item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            if batch_idx % 100 == 0:
                print(f"[train={train_compression} test={test_compression}] batch {batch_idx}/{len(test_loader)}", flush=True)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy = float(test_corrects) / float(num_test_we_use)
    test_average_loss = float(test_loss) / float(num_test_we_use)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    precision_macro = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    precision_weighted = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, digits=4, zero_division=0)

    run_name = f"train_{train_compression}__test_{test_compression}"
    cm_csv_path = os.path.join(save_dir, f'{run_name}_confusion_matrix.csv')
    metrics_txt_path = os.path.join(save_dir, f'{run_name}_metrics.txt')

    save_confusion_matrix(cm, cm_csv_path)

    metrics_dict = {
        'train_compression': train_compression,
        'test_compression': test_compression,
        'checkpoint_path': model_path,
        'loss': test_average_loss,
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'elapsed_seconds': test_elapsed_time
    }

    save_metrics_text(metrics_dict, report, metrics_txt_path)

    print(
        '[train={} test={}] elapsed: {:2.0f}m{:2.0f}s test loss: {:4.4f} test accu: {:.4f}'.format(
            train_compression,
            test_compression,
            test_elapsed_time // 60,
            test_elapsed_time % 60,
            test_average_loss,
            acc
        ),
        flush=True
    )

    return metrics_dict


def write_summary_csv(results, out_path):
    if not results:
        return

    fieldnames = [
        'train_compression',
        'test_compression',
        'checkpoint_path',
        'loss',
        'accuracy',
        'precision_macro',
        'recall_macro',
        'f1_macro',
        'precision_weighted',
        'recall_weighted',
        'f1_weighted',
        'elapsed_seconds'
    ]

    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def write_accuracy_matrix_csv(results, compressions, out_path):
    matrix = {train_c: {test_c: '' for test_c in compressions} for train_c in compressions}

    for row in results:
        matrix[row['train_compression']][row['test_compression']] = row['accuracy']

    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['train\\test'] + compressions)
        for train_c in compressions:
            writer.writerow([train_c] + [matrix[train_c][test_c] for test_c in compressions])


def main():
    all_results = []

    for train_compression in compression_levels:
        print('\n' + '=' * 120, flush=True)
        print('Finding trained model for train compression: {}'.format(train_compression), flush=True)

        model_path = find_final_model_for_compression(train_compression)
        if model_path is None:
            print('Skipping {} because no final model .pth found in {}'.format(
                train_compression,
                os.path.join(checkpoint_root, train_compression)
            ), flush=True)
            continue

        print('Using model: {}'.format(model_path), flush=True)

        for test_compression in compression_levels:
            print('\n' + '-' * 120, flush=True)
            print('Evaluating model trained on {} against test set {}'.format(
                train_compression, test_compression
            ), flush=True)

            data_path = os.path.join(data_dir, f'train_val_test_paths_labels_{test_compression}.pkl')
            if not os.path.exists(data_path):
                print('Skipping test compression {} because file not found: {}'.format(
                    test_compression, data_path
                ), flush=True)
                continue

            _, _, _, _, test_dataset, test_num_each = get_data(data_path)

            result = evaluate_model_on_dataset(
                model_path=model_path,
                test_dataset=test_dataset,
                test_num_each=test_num_each,
                train_compression=train_compression,
                test_compression=test_compression
            )
            all_results.append(result)

    summary_csv = os.path.join(save_dir, 'cross_compression_summary.csv')
    matrix_csv = os.path.join(save_dir, 'cross_compression_accuracy_matrix.csv')

    write_summary_csv(all_results, summary_csv)
    write_accuracy_matrix_csv(all_results, compression_levels, matrix_csv)

    print('\n' + '=' * 120, flush=True)
    print('Finished all cross-compression testing.', flush=True)
    print('Saved summary CSV to {}'.format(summary_csv), flush=True)
    print('Saved accuracy matrix CSV to {}'.format(matrix_csv), flush=True)


if __name__ == "__main__":
    main()

print('Done', flush=True)
print('', flush=True)
