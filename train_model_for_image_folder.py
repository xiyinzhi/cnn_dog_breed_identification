# -*- coding:utf-8 -*-
__author__ = 'Yi'

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(['inceptionresnetv2', 'nasnetalarge'] + [name for name in models.__dict__
                                                              if name.islower() and not name.startswith("__")
                                                              and callable(models.__dict__[name])])

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')

parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='fine tune pre-trained model')
parser.add_argument('--image_size', dest='image_size', default=224,
                    type=int, help='Image size for training input(default: 224)')
parser.add_argument('--test_image_size', dest='test_image_size', default=256,
                    type=int, help='Image size for testing input(default: 256)')
# parser.add_argument('--data_balance', dest='data_balance', action='store_true',
#                     help='use weighted sampler to balance training data.')
parser.add_argument('--step_size', dest='step_size', default=30,
                    type=int, help='Lr stepsize(default: 30)')

best_prec1 = 0
special_arch_mean = {
    "inceptionresnetv2": [0.5, 0.5, 0.5],
    "nasnetalarge": [0.5, 0.5, 0.5]
}
special_arch_std = {
    "inceptionresnetv2": [0.5, 0.5, 0.5],
    "nasnetalarge": [0.5, 0.5, 0.5]
}
train_image_size = {
    "inceptionresnetv2": 299,
    "nasnetalarge": 331
}
test_image_size = {
    "inceptionresnetv2": 331,
    "nasnetalarge": 354
}


class FineTuneModel(nn.Module):
    def __init__(self, original_model, arch, num_classes):
        super(FineTuneModel, self).__init__()

        if arch.startswith('alexnet'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'alexnet'
        elif arch.startswith('resnet'):
            # Everything except the last linear layer
            self.features = nn.Sequential(*list(original_model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Linear(original_model.fc.in_features, num_classes)
            )
            self.modelName = 'resnet'
        elif arch.startswith('vgg16'):
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(25088, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )
            self.modelName = 'vgg16'
        else:
            raise ("Finetuning not supported on this architecture yet")

        # Freeze those weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        f = self.features(x)
        if self.modelName == 'alexnet':
            f = f.view(f.size(0), 256 * 6 * 6)
        elif self.modelName == 'vgg16':
            f = f.view(f.size(0), -1)
        elif self.modelName == 'resnet':
            f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y


def my_pil_loader(path):
    """
    Default loader can't load images correctly. Please pass this function as the loader of Dataset.
    :param path:
    :return:
    """
    from PIL import Image
    img = Image.open(path)
    return img.convert('RGB')


def load_inception_resnet_v2(num_classes, finetune=True):
    from pretrainedmodels.inceptionresnetv2 import InceptionResNetV2
    print("Use InceptionResNetV2.")

    if not finetune:
        return InceptionResNetV2(num_classes)

    pretrained_model_path = '/Users/dingyi/.torch/models/inceptionresnetv2-d579a627.pth'
    model = InceptionResNetV2()
    model.load_state_dict(torch.load(pretrained_model_path))

    new_classif = nn.Linear(1536, num_classes)
    new_classif.weight.data = model.classif.weight.data[:num_classes]
    new_classif.bias.data = model.classif.bias.data[:num_classes]
    model.classif = new_classif

    frozen_layers = nn.Sequential(*list(model.children())[:-1])
    for p in frozen_layers.parameters():
        p.requires_grad = False

    print("Classif:", model.classif)

    model.input_space = 'RGB'
    model.input_size = [3, 299, 299]
    model.input_range = [0, 1]

    model.mean = [0.5, 0.5, 0.5]
    model.std = [0.5, 0.5, 0.5]

    return model


def data_balance_weights(images, shuffle=True):
    from collections import defaultdict
    if shuffle:
        import random
        random.shuffle(images)

    counter = defaultdict(int)
    for img, cls in images:
        counter[cls] += 1
    n = len(images)
    cls_weight = {cls: n * 1.0 / n_imgs for cls, n_imgs in counter.items()}
    return [cls_weight[cls] for img, cls in images]


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    with open('args', 'w') as f:
        f.write(str(args))

    args.distributed = args.world_size > 1
    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    traindir = os.path.join(args.data, 'train')
    # Get number of classes from train directory
    num_classes = len([name for name in os.listdir(traindir)])
    print("num_classes = {}".format(num_classes))

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch == "inceptionresnetv2":
            model = load_inception_resnet_v2(num_classes, args.finetune)
        else:
            if args.finetune:
                print("Finetune.")
                original_model = models.__dict__[args.arch](pretrained=True)
                model = FineTuneModel(original_model, args.arch, num_classes)
            else:
                model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # gpus = map(int, args.gpus.split(','))
    gpus = range(torch.cuda.device_count())
    # torch.cuda.set_device(gpus[0])
    print("GPUs:", gpus, "current device", torch.cuda.current_device())

    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features, device_ids=gpus)
            model.cuda(gpus[0])
        else:
            model = torch.nn.DataParallel(model, device_ids=gpus).cuda(gpus[0])
    else:
        model.cuda(gpus[0])
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=gpus)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpus[0])

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()) if args.finetune else model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    # traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    mean_val = special_arch_mean[args.arch] if special_arch_mean.has_key(args.arch) else [0.485, 0.456, 0.406]
    std_val = special_arch_std[args.arch] if special_arch_std.has_key(args.arch) else [0.229, 0.224, 0.225]

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406] if args.arch != "inceptionresnetv2" else [0.5, 0.5, 0.5],
    #                                  std=[0.229, 0.224, 0.225] if args.arch != "inceptionresnetv2" else [0.5, 0.5, 0.5])
    normalize = transforms.Normalize(mean=mean_val, std=std_val)

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomSizedCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]), loader=my_pil_loader)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # elif args.data_balance:
    #     balance_weights = data_balance_weights(train_dataset.imgs)
    #     train_sampler = torch.utils.data.sampler.WeightedRandomSampler(balance_weights, args.batch_size,
    #                                                                    replacement=True)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Scale(args.test_image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ]), loader=my_pil_loader),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, stepsize=args.step_size)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, gpus)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion, gpus)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, model)


def train(train_loader, model, criterion, optimizer, epoch, gpus):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)  # .cuda(gpus[0])
        target_var = torch.autograd.Variable(target)  # .cuda(gpus[0])

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion, gpus):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)  # .cuda(gpus[0])
        target_var = torch.autograd.Variable(target, volatile=True)  # .cuda(gpus[0])

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, model, filename='checkpoint.pth.dict'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.dict')
        torch.save(model, "best.model")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, stepsize=30):
    """Sets the learning rate to the initial LR decayed by 10 every #stepsize epochs"""
    lr = args.lr * (0.1 ** (epoch // stepsize))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
