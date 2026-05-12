import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.init as init
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn import DataParallel
import os
from PIL import Image, ImageOps
import time
import pickle
import numpy as np
from torchvision.transforms import Lambda
import argparse
import copy
import random
import numbers

parser = argparse.ArgumentParser(description='lstm training')
parser.add_argument('-g', '--gpu', default=[0], nargs='+', type=int, help='index of gpu to use')
parser.add_argument('-s', '--seq', default=4, type=int, help='sequence length')
parser.add_argument('-t', '--train', default=100, type=int, help='train batch size')
parser.add_argument('-v', '--val', default=8, type=int, help='valid batch size')
parser.add_argument('-o', '--opt', default=1, type=int, help='0 for sgd 1 for adam')
parser.add_argument('-m', '--multi', default=1, type=int, help='0 for single opt, 1 for multi opt')
parser.add_argument('-e', '--epo', default=25, type=int, help='epochs')
parser.add_argument('-w', '--work', default=2, type=int, help='num workers')
parser.add_argument('-f', '--flip', default=0, type=int, help='0 no flip, 1 flip')
parser.add_argument('-c', '--crop', default=1, type=int, help='0 rand, 1 center, 5 five_crop, 10 ten_crop')
parser.add_argument('-l', '--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum for sgd')
parser.add_argument('--weightdecay', default=0, type=float, help='weight decay for sgd')
parser.add_argument('--dampening', default=0, type=float, help='dampening for sgd')
parser.add_argument('--nesterov', default=False, type=bool, help='nesterov momentum')
parser.add_argument('--sgdadjust', default=1, type=int, help='sgd lr schedule method')
parser.add_argument('--sgdstep', default=5, type=int, help='step size for StepLR')
parser.add_argument('--sgdgamma', default=0.1, type=float, help='gamma for StepLR')
parser.add_argument('--save_dir', default='checkpoints_phase', type=str, help='directory to save checkpoints')
parser.add_argument('--pretrained', default=0, type=int, help='1 to use ImageNet pretrained ResNet50, 0 otherwise')

args = parser.parse_args()

gpu_usg = ",".join(list(map(str, args.gpu)))
sequence_length = args.seq
train_batch_size = args.train
val_batch_size = args.val
optimizer_choice = args.opt
multi_optim = args.multi
epochs = args.epo
workers = args.work
use_flip = args.flip
crop_type = args.crop
learning_rate = args.lr
momentum = args.momentum
weight_decay = args.weightdecay
dampening = args.dampening
use_nesterov = args.nesterov
sgd_adjust_lr = args.sgdadjust
sgd_step = args.sgdstep
sgd_gamma = args.sgdgamma
save_dir = args.save_dir
use_pretrained = bool(args.pretrained)

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_usg
num_gpu = torch.cuda.device_count()
use_gpu = torch.cuda.is_available()

os.makedirs(save_dir, exist_ok=True)

print('number of gpu   : {:6d}'.format(num_gpu), flush=True)
print('sequence length : {:6d}'.format(sequence_length), flush=True)
print('train batch size: {:6d}'.format(train_batch_size), flush=True)
print('valid batch size: {:6d}'.format(val_batch_size), flush=True)
print('optimizer choice: {:6d}'.format(optimizer_choice), flush=True)
print('multiple optim  : {:6d}'.format(multi_optim), flush=True)
print('num of epochs   : {:6d}'.format(epochs), flush=True)
print('num of workers  : {:6d}'.format(workers), flush=True)
print('test crop type  : {:6d}'.format(crop_type), flush=True)
print('whether to flip : {:6d}'.format(use_flip), flush=True)
print('learning rate   : {:.4f}'.format(learning_rate), flush=True)
print('momentum for sgd: {:.4f}'.format(momentum), flush=True)
print('weight decay    : {:.4f}'.format(weight_decay), flush=True)
print('dampening       : {:.4f}'.format(dampening), flush=True)
print('use nesterov    : {:6d}'.format(int(use_nesterov)), flush=True)
print('method for sgd  : {:6d}'.format(sgd_adjust_lr), flush=True)
print('step for sgd    : {:6d}'.format(sgd_step), flush=True)
print('gamma for sgd   : {:.4f}'.format(sgd_gamma), flush=True)
print('save dir        : {}'.format(save_dir), flush=True)
print('use pretrained  : {}'.format(use_pretrained), flush=True)


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.count = 0

    def __call__(self, img):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)

        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img

        random.seed(self.count // sequence_length)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        self.count += 1
        return img.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontalFlip(object):
    def __init__(self):
        self.count = 0

    def __call__(self, img):
        seed = self.count // sequence_length
        random.seed(seed)
        prob = random.random()
        self.count += 1
        if prob < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


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


class SeqSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class resnet_lstm(torch.nn.Module):
    def __init__(self):
        super(resnet_lstm, self).__init__()
        resnet = models.resnet50(pretrained=use_pretrained)
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

    if use_flip == 0:
        train_transforms = transforms.Compose([
            RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])
    else:
        train_transforms = transforms.Compose([
            RandomCrop(224),
            RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.3456, 0.2281, 0.2233], [0.2528, 0.2135, 0.2104])
        ])

    if crop_type == 0:
        test_transforms = transforms.Compose([
            RandomCrop(224),
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

    train_dataset = CholecDataset(train_paths, train_labels, train_transforms)
    val_dataset = CholecDataset(val_paths, val_labels, test_transforms)
    test_dataset = CholecDataset(test_paths, test_labels, test_transforms)

    return train_dataset, train_num_each, val_dataset, val_num_each, test_dataset, test_num_each


def build_optimizer(model):
    model_for_optim = model.module if isinstance(model, DataParallel) else model

    if multi_optim == 0:
        if optimizer_choice == 0:
            optimizer = optim.SGD(
                model_for_optim.parameters(),
                lr=learning_rate,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=use_nesterov
            )
            if sgd_adjust_lr == 0:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_step, gamma=sgd_gamma)
            else:
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
            optimizer = optim.Adam(model_for_optim.parameters(), lr=learning_rate)
            scheduler = None
    else:
        if optimizer_choice == 0:
            optimizer = optim.SGD([
                {'params': model_for_optim.share.parameters()},
                {'params': model_for_optim.lstm.parameters(), 'lr': learning_rate},
                {'params': model_for_optim.fc.parameters(), 'lr': learning_rate},
            ],
                lr=learning_rate / 10,
                momentum=momentum,
                dampening=dampening,
                weight_decay=weight_decay,
                nesterov=use_nesterov
            )
            if sgd_adjust_lr == 0:
                scheduler = lr_scheduler.StepLR(optimizer, step_size=sgd_step, gamma=sgd_gamma)
            else:
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        else:
            optimizer = optim.Adam([
                {'params': model_for_optim.share.parameters()},
                {'params': model_for_optim.lstm.parameters(), 'lr': learning_rate},
                {'params': model_for_optim.fc.parameters(), 'lr': learning_rate},
            ], lr=learning_rate / 10)
            scheduler = None

    return optimizer, scheduler


def train_model(train_dataset, train_num_each, val_dataset, val_num_each):
    num_train = len(train_dataset)
    num_val = len(val_dataset)

    train_useful_start_idx = get_useful_start_idx(sequence_length, train_num_each)
    val_useful_start_idx = get_useful_start_idx(sequence_length, val_num_each)

    num_train_we_use = len(train_useful_start_idx) // max(num_gpu, 1) * max(num_gpu, 1)
    num_val_we_use = len(val_useful_start_idx) // max(num_gpu, 1) * max(num_gpu, 1)

    train_we_use_start_idx = train_useful_start_idx[0:num_train_we_use]
    val_we_use_start_idx = val_useful_start_idx[0:num_val_we_use]

    train_idx = []
    for i in range(num_train_we_use):
        for j in range(sequence_length):
            train_idx.append(train_we_use_start_idx[i] + j)

    val_idx = []
    for i in range(num_val_we_use):
        for j in range(sequence_length):
            val_idx.append(val_we_use_start_idx[i] + j)

    num_train_all = len(train_idx)
    num_val_all = len(val_idx)

    print('num of train dataset: {:6d}'.format(num_train), flush=True)
    print('num train start idx : {:6d}'.format(len(train_useful_start_idx)), flush=True)
    print('last idx train start: {:6d}'.format(train_useful_start_idx[-1]), flush=True)
    print('num of train we use : {:6d}'.format(num_train_we_use), flush=True)
    print('num of all train use: {:6d}'.format(num_train_all), flush=True)
    print('num of valid dataset: {:6d}'.format(num_val), flush=True)
    print('num valid start idx : {:6d}'.format(len(val_useful_start_idx)), flush=True)
    print('last idx valid start: {:6d}'.format(val_useful_start_idx[-1]), flush=True)
    print('num of valid we use : {:6d}'.format(num_val_we_use), flush=True)
    print('num of all valid use: {:6d}'.format(num_val_all), flush=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=SeqSampler(val_idx),
        num_workers=workers,
        pin_memory=False
    )

    model = resnet_lstm()
    if use_gpu:
        model = model.cuda()

    if num_gpu > 1:
        model = DataParallel(model)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer, scheduler = build_optimizer(model)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0.0
    correspond_train_acc = 0.0
    record_np = np.zeros([epochs, 4])

    for epoch in range(epochs):
        np.random.shuffle(train_we_use_start_idx)
        train_idx = []
        for i in range(num_train_we_use):
            for j in range(sequence_length):
                train_idx.append(train_we_use_start_idx[i] + j)

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            sampler=SeqSampler(train_idx),
            num_workers=workers,
            pin_memory=False
        )

        model.train()
        train_loss = 0.0
        train_corrects = 0
        train_start_time = time.time()

        for batch_idx, data in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"epoch {epoch} train batch {batch_idx}/{len(train_loader)}", flush=True)

            inputs, labels_1, labels_2 = data

            if use_gpu:
                inputs = inputs.cuda(non_blocking=True)
                labels = labels_2.cuda(non_blocking=True)
            else:
                labels = labels_2

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_corrects += torch.sum(preds == labels.data).item()

        train_elapsed_time = time.time() - train_start_time
        train_accuracy = float(train_corrects) / float(num_train_all)
        train_average_loss = float(train_loss) / float(num_train_all)

        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_start_time = time.time()

        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                if batch_idx % 100 == 0:
                    print(f"epoch {epoch} val batch {batch_idx}/{len(val_loader)}", flush=True)

                inputs, labels_1, labels_2 = data
                labels_2 = labels_2[(sequence_length - 1)::sequence_length]

                if use_gpu:
                    inputs = inputs.cuda(non_blocking=True)
                    labels = labels_2.cuda(non_blocking=True)
                else:
                    labels = labels_2

                if crop_type == 0 or crop_type == 1:
                    outputs = model(inputs)
                elif crop_type == 5:
                    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                    inputs = inputs.view(-1, 3, 224, 224)
                    outputs = model(inputs)
                    outputs = outputs.view(5, -1, 7)
                    outputs = torch.mean(outputs, 0)
                elif crop_type == 10:
                    inputs = inputs.permute(1, 0, 2, 3, 4).contiguous()
                    inputs = inputs.view(-1, 3, 224, 224)
                    outputs = model(inputs)
                    outputs = outputs.view(10, -1, 7)
                    outputs = torch.mean(outputs, 0)
                else:
                    raise ValueError("Unsupported crop_type: {}".format(crop_type))

                outputs = outputs[sequence_length - 1::sequence_length]
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs.data, 1)

                val_loss += loss.item()
                val_corrects += torch.sum(preds == labels.data).item()

        val_elapsed_time = time.time() - val_start_time
        val_accuracy = float(val_corrects) / float(num_val_we_use)
        val_average_loss = float(val_loss) / float(num_val_we_use)

        print(
            'epoch: {:4d}'
            ' train in: {:2.0f}m{:2.0f}s'
            ' train loss: {:4.4f}'
            ' train accu: {:.4f}'
            ' valid in: {:2.0f}m{:2.0f}s'
            ' valid loss: {:4.4f}'
            ' valid accu: {:.4f}'.format(
                epoch,
                train_elapsed_time // 60,
                train_elapsed_time % 60,
                train_average_loss,
                train_accuracy,
                val_elapsed_time // 60,
                val_elapsed_time % 60,
                val_average_loss,
                val_accuracy
            ),
            flush=True
        )

        if scheduler is not None:
            if optimizer_choice == 0:
                if sgd_adjust_lr == 0:
                    scheduler.step()
                else:
                    scheduler.step(val_average_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
        elif val_accuracy == best_val_accuracy and train_accuracy > correspond_train_acc:
            correspond_train_acc = train_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

        record_np[epoch, 0] = train_accuracy
        record_np[epoch, 1] = train_average_loss
        record_np[epoch, 2] = val_accuracy
        record_np[epoch, 3] = val_average_loss

        ckpt_epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        npy_epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.npy')
        torch.save(model.state_dict(), ckpt_epoch_path)
        np.save(npy_epoch_path, record_np[:epoch + 1])
        print(f"saved {ckpt_epoch_path}", flush=True)
        print(f"saved {npy_epoch_path}", flush=True)

    print('best accuracy: {:.4f} cor train accu: {:.4f}'.format(best_val_accuracy, correspond_train_acc), flush=True)

    save_val = int("{:4.0f}".format(best_val_accuracy * 10000))
    save_train = int("{:4.0f}".format(correspond_train_acc * 10000))

    model_name = os.path.join(
        save_dir,
        "lstm"
        + "_epoch_" + str(epochs)
        + "_length_" + str(sequence_length)
        + "_opt_" + str(optimizer_choice)
        + "_mulopt_" + str(multi_optim)
        + "_flip_" + str(use_flip)
        + "_crop_" + str(crop_type)
        + "_batch_" + str(train_batch_size)
        + "_train_" + str(save_train)
        + "_val_" + str(save_val)
        + ".pth"
    )

    record_name = os.path.join(
        save_dir,
        "lstm"
        + "_epoch_" + str(epochs)
        + "_length_" + str(sequence_length)
        + "_opt_" + str(optimizer_choice)
        + "_mulopt_" + str(multi_optim)
        + "_flip_" + str(use_flip)
        + "_crop_" + str(crop_type)
        + "_batch_" + str(train_batch_size)
        + "_train_" + str(save_train)
        + "_val_" + str(save_val)
        + ".npy"
    )

    torch.save(best_model_wts, model_name)
    np.save(record_name, record_np)

    print(f"saved final model to {model_name}", flush=True)
    print(f"saved final record to {record_name}", flush=True)


def main():
    train_dataset, train_num_each, val_dataset, val_num_each, _, _ = get_data('train_val_test_paths_labels.pkl')
    train_model(train_dataset, train_num_each, val_dataset, val_num_each)


if __name__ == "__main__":
    main()

print('Done', flush=True)
print('', flush=True)