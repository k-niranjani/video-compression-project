import os
import time
import pickle
import argparse

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
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='lstm testing')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length')
parser.add_argument('-t', '--test', default=800, type=int, help='test batch size')
parser.add_argument('-w', '--work', default=2, type=int, help='num of workers to use')
parser.add_argument('-n', '--name', type=str, required=True, help='path to model checkpoint (.pth)')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 center, 5 five_crop, 10 ten_crop')
parser.add_argument('--num_classes', default=7, type=int, help='number of phase classes')
parser.add_argument('--save_dir', default='test_outputs', type=str, help='directory to save outputs')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg

sequence_length = args.seq
test_batch_size = args.test
workers = args.work
model_name = args.name
crop_type = args.crop
num_classes = args.num_classes
save_dir = args.save_dir

os.makedirs(save_dir, exist_ok=True)

model_base_name = os.path.splitext(os.path.basename(model_name))[0]

num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()
device = torch.device('cuda' if use_gpu else 'cpu')

print('number of gpu   : {:6d}'.format(num_gpu), flush=True)
print('sequence length : {:6d}'.format(sequence_length), flush=True)
print('test batch size : {:6d}'.format(test_batch_size), flush=True)
print('num of workers  : {:6d}'.format(workers), flush=True)
print('test crop type  : {:6d}'.format(crop_type), flush=True)
print('name of this model: {:s}'.format(model_name), flush=True)
print('save dir        : {:s}'.format(save_dir), flush=True)


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

        # weights=None because checkpoint is loaded afterward
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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

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


def save_confusion_matrix(cm, out_path):
    np.savetxt(out_path, cm, fmt='%d', delimiter=',')


def plot_multiclass_roc(y_true, y_score, n_classes, out_path):
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.4f})')

    plt.plot(fpr["micro"], tpr["micro"], linestyle='--', label=f'Micro-average (AUC = {roc_auc["micro"]:.4f})')
    plt.plot([0, 1], [0, 1], linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multiclass ROC Curve')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return roc_auc


def save_metrics_text(metrics_dict, report_text, out_path):
    with open(out_path, 'w') as f:
        f.write('Overall Metrics\n')
        f.write('===============\n')
        for k, v in metrics_dict.items():
            f.write(f'{k}: {v}\n')
        f.write('\nClassification Report\n')
        f.write('=====================\n')
        f.write(report_text)


def test_model(test_dataset, test_num_each):
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

    model = load_checkpoint(model, model_name, device)
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction='sum')

    test_loss = 0.0
    test_corrects = 0
    test_start_time = time.time()

    all_preds = []
    all_labels = []
    all_probs = []

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

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            test_loss += loss.item()
            test_corrects += torch.sum(preds == labels).item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.append(probs.cpu().numpy())

            if batch_idx % 100 == 0:
                print(f"test batch {batch_idx}/{len(test_loader)}", flush=True)

    test_elapsed_time = time.time() - test_start_time
    test_accuracy = float(test_corrects) / float(num_test_we_use)
    test_average_loss = float(test_loss) / float(num_test_we_use)

    all_probs = np.concatenate(all_probs, axis=0)
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

    roc_auc_dict = plot_multiclass_roc(
        y_true=all_labels,
        y_score=all_probs,
        n_classes=num_classes,
        out_path=os.path.join(save_dir, f'{model_base_name}_roc_curve.png')
    )

    metrics_dict = {
        'loss': test_average_loss,
        'accuracy': acc,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'f1_weighted': f1_weighted,
        'auc_micro': roc_auc_dict['micro'],
    }

    for i in range(num_classes):
        metrics_dict[f'auc_class_{i}'] = roc_auc_dict[i]

    pred_pkl_path = os.path.join(save_dir, f'{model_base_name}_predictions.pkl')
    probs_npy_path = os.path.join(save_dir, f'{model_base_name}_probabilities.npy')
    labels_npy_path = os.path.join(save_dir, f'{model_base_name}_labels.npy')
    preds_npy_path = os.path.join(save_dir, f'{model_base_name}_preds.npy')
    cm_csv_path = os.path.join(save_dir, f'{model_base_name}_confusion_matrix.csv')
    metrics_txt_path = os.path.join(save_dir, f'{model_base_name}_metrics.txt')

    with open(pred_pkl_path, 'wb') as f:
        pickle.dump(all_preds.tolist(), f)

    np.save(probs_npy_path, all_probs)
    np.save(labels_npy_path, all_labels)
    np.save(preds_npy_path, all_preds)
    save_confusion_matrix(cm, cm_csv_path)
    save_metrics_text(metrics_dict, report, metrics_txt_path)

    print(
        'test elapsed: {:2.0f}m{:2.0f}s'
        ' test loss: {:4.4f}'
        ' test accu: {:.4f}'.format(
            test_elapsed_time // 60,
            test_elapsed_time % 60,
            test_average_loss,
            acc
        ),
        flush=True
    )

    print('precision_macro : {:.4f}'.format(precision_macro), flush=True)
    print('recall_macro    : {:.4f}'.format(recall_macro), flush=True)
    print('f1_macro        : {:.4f}'.format(f1_macro), flush=True)
    print('precision_weighted : {:.4f}'.format(precision_weighted), flush=True)
    print('recall_weighted    : {:.4f}'.format(recall_weighted), flush=True)
    print('f1_weighted        : {:.4f}'.format(f1_weighted), flush=True)
    print('auc_micro          : {:.4f}'.format(roc_auc_dict['micro']), flush=True)

    for i in range(num_classes):
        print(f'auc_class_{i}        : {roc_auc_dict[i]:.4f}', flush=True)

    print(f'saved predictions to {pred_pkl_path}', flush=True)
    print(f'saved probabilities to {probs_npy_path}', flush=True)
    print(f'saved labels to {labels_npy_path}', flush=True)
    print(f'saved preds to {preds_npy_path}', flush=True)
    print(f'saved confusion matrix to {cm_csv_path}', flush=True)
    print(f'saved metrics to {metrics_txt_path}', flush=True)
    print(f'saved ROC plot to {os.path.join(save_dir, f"{model_base_name}_roc_curve.png")}', flush=True)


def main():
    _, _, _, _, test_dataset, test_num_each = get_data('train_val_test_paths_labels.pkl')
    test_model(test_dataset, test_num_each)


if __name__ == "__main__":
    main()

print('Done', flush=True)
print('', flush=True)