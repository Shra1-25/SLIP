# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
import math
from multiprocessing import context
import os
import random
import shutil
import time
import timm
import warnings
from torch import inf


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from models import LanguageAndVisionConcat
import models
from collections import OrderedDict
from transformers import DistilBertTokenizer
from sklearn.metrics import recall_score
from pytorch_lightning import seed_everything
from torchmetrics import AUROC

import datasets
import utils


def get_args_parser():
    parser = argparse.ArgumentParser(description='Linear probe evaluation', add_help=False)
    parser.add_argument('--root', default='/scratch/ssc10020/IndependentStudy/SLIP/dataset/ISIC/', type=str,
                        help='path to dataset root')
    parser.add_argument('--dataset', default='isic', help='dataset name')
    parser.add_argument('--output-dir', default='./', type=str)
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base_patch16_224',
                        help='model architecture: (default: ViT-B/16)')
    parser.add_argument('-j', '--workers', default=64, type=int, metavar='N',
                        help='number of data loading workers (default: 64)')
    parser.add_argument('--epochs', default=25, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N',
                        help='number of samples per-device/per-gpu ')
    parser.add_argument('--num-classes', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial (base) learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--model', default='SLIP_VITB16', type=str)
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                        metavar='W', help='weight decay (default: 0.)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', default=10, type=int)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=77, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--pretrained', default='', type=str,
                        help='path to CLIP pretrained checkpoint')
    parser.add_argument('--context-length', default=26, type=int, help='maximum length of the token embeddings for text data')
    parser.add_argument('--save-model-name-tag', default='linear_eval', type=str)
    return parser

best_acc1 = torch.tensor(0)


def main(args):
    # utils.init_distributed_mode(args)
    args.distributed=False

    global best_acc1

    if args.seed is not None:
        # random.seed(args.seed)
        # torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        # warnings.warn('You have chosen to seed training. '
        #               'This will turn on the CUDNN deterministic setting, '
        #               'which can slow down your training considerably! '
        #               'You may see unexpected behavior when restarting '
        #               'from checkpoints.')
        seed_everything(args.seed)

    if args.arch.startswith('vit'):
        linear_keyword = 'head'
    else:
        linear_keyword = 'fc' # 'fc' for resnet50 and 'head' for vit
    if os.path.isfile(args.pretrained):
        print("=> loading checkpoint '{}'".format(args.pretrained))

        if args.gpu is None:
            checkpoint = torch.load(args.pretrained)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(args.pretrained, map_location=loc)

        visual_keyword = 'visual.'

        # rename CLIP pre-trained keys
        ckpt = torch.load(args.pretrained, map_location='cpu')
        state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.', '')] = v
        
        # state_dict = checkpoint['state_dict']
        # for k in list(state_dict.keys()):
        #     # retain only base_encoder up to before the embedding layer
        #     if k.startswith(visual_keyword) and not k.startswith(visual_keyword + linear_keyword):
        #         # remove prefix
        #         state_dict[k[len(visual_keyword):]] = state_dict[k]
        #     # delete renamed or unused k
        #     del state_dict[k]
    else:
        raise Exception('Missing pretrained model checkpoint: {}'.format(args.pretrained))

    # create model
    print("=> creating model '{}'".format(args.arch))
    # slip_model = timm.models.create_model(args.arch, num_classes=args.num_classes)
    slip_model = getattr(models, args.model)(ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim, context_length=args.context_length)
    slip_model.cuda()
    slip_model.load_state_dict(state_dict, strict=True)

    for name,param in slip_model.named_parameters():
        param.requires_grad = False

    model = LanguageAndVisionConcat(embed_module=slip_model, num_classes=args.num_classes)
    
    # import pdb; pdb.set_trace()
    args.start_epoch = 0
    # msg = model.load_state_dict(state_dict, strict=False)
    
    # assert set(msg.missing_keys) == {"%s.weight" % linear_keyword, "%s.bias" % linear_keyword}

    # # freeze all layers but the last fc
    # for name, param in model.named_parameters():
    #     if name not in ['%s.weight' % linear_keyword, '%s.bias' % linear_keyword]:
    #         param.requires_grad = False
    # # init the fc layer
    # getattr(model, linear_keyword).weight.data.normal_(mean=0.0, std=0.01)
    # getattr(model, linear_keyword).bias.data.zero_()

    init_lr = args.lr * int(args.batch_size / utils.get_world_size()) / 256
    args.workers = int((args.workers + utils.get_world_size() - 1) / utils.get_world_size())

    model.cuda(args.gpu)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    # assert len(parameters) == 2  # weight, bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            # best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    cwd = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(cwd, 'dataset_catalog.json')) as f:
        catalog = json.load(f)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(384),
        transforms.RandomHorizontalFlip(),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        normalize,
    ])
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    if args.dataset=='isic':
        train_dataset = datasets.ISICE2ETrainDataset(train_transform, args.root, os.path.join(args.root, 'train_split_metadata.csv'), tokenizer=tokenizer, context_length=args.context_length)# datasets.get_downstream_dataset(catalog, args.dataset, is_train=True, transform=train_transform)
        val_dataset = datasets.ISICE2ETrainDataset(val_transform, args.root, os.path.join(args.root, 'val_split_metadata.csv'), tokenizer=tokenizer, context_length=args.context_length) # datasets.get_downstream_dataset(catalog, args.dataset, is_train=False, transform=val_transform)
        test_dataset = datasets.ISICE2ETrainDataset(val_transform, args.root, os.path.join(args.root, 'test_data.csv'), tokenizer=tokenizer, context_length=args.context_length)
    elif args.dataset=='cbis':
        train_dataset = datasets.CBISE2ETrainDataset(train_transform, args.root, os.path.join(args.root, 'train_split_metadata.csv'), tokenizer=tokenizer, context_length=args.context_length)# datasets.get_downstream_dataset(catalog, args.dataset, is_train=True, transform=train_transform)
        val_dataset = datasets.CBISE2ETrainDataset(val_transform, args.root, os.path.join(args.root, 'val_split_metadata.csv'), tokenizer=tokenizer, context_length=args.context_length) # datasets.get_downstream_dataset(catalog, args.dataset, is_train=False, transform=val_transform)
        test_dataset = datasets.CBISE2ETrainDataset(val_transform, args.root, os.path.join(args.root, 'test_data.csv'), tokenizer=tokenizer, context_length=args.context_length)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print(args)
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train_stats = train(train_loader, model, criterion, optimizer, epoch, args)
        
        if (epoch + 1) % args.eval_freq != 0:
            continue

        # evaluate on validation set
        val_stats = validate(val_loader, model, criterion, args)
        acc1 = val_stats['acc1']

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if utils.is_main_process(): # only the first GPU saves checkpoint
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.output_dir, args)
            if epoch == args.start_epoch:
                sanity_check(model.state_dict(), args.pretrained, linear_keyword, visual_keyword)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'val_{k}': v for k, v in val_stats.items()},
                     'epoch': epoch}

        if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'linear_{}_lr={}_log.txt'.format(args.dataset, args.lr)), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')
    
    print("Evaluating on test dataset:")
    ckpt = torch.load(f'{args.output_dir}/'+args.save_model_name_tag+'_checkpoint_best_seed_'+str(args.seed)+'_context_len_'+str(args.context_length)+'.pt', map_location=loc)
    model_best = LanguageAndVisionConcat(embed_module=slip_model, num_classes=args.num_classes)
    model_best.cuda(args.gpu)
    model_best.load_state_dict(ckpt['state_dict'])
    test_stats = validate(test_loader, model_best, criterion, args)
    log_stats = {**{f'test_{k}': v for k, v in test_stats.items()}}
    if utils.is_main_process():
            with open(os.path.join(args.output_dir, 'linear_test_{}_lr={}_log.txt'.format(args.dataset, args.lr)), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (images, caption, target, aug1, aug2) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            for k,v in caption.items():
                caption[k] = v.cuda()
            aug1 = aug1.cuda()
            aug2 = aug2.cuda()
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        inputs = [images, caption, aug1, aug2]

        # compute output
        output = model(inputs)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, min(5,args.num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, caption, target, aug1, aug2) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
                for k,v in caption.items():
                    caption[k] = v.cuda()
                aug1 = aug1.cuda()
                aug2 = aug2.cuda()
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            inputs = [images, caption, aug1, aug2]
            # compute output
            output = model(inputs)
            
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5,args.num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            rec_score = recall_score(target.cpu(), output.argmax(dim=1).cpu(), labels=range(args.num_classes), average='micro')
            auc_score = AUROC(task='multiclass', num_classes=args.num_classes)(output, target)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.4f} Acc@5 {top5.avg:.4f} Recall score {rec_score:.4f} AUROC multiclass score {auc_score:.4f}'
              .format(top1=top1, top5=top5, rec_score=rec_score, auc_score=auc_score))

    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


def save_checkpoint(state, is_best, output_dir, args):
    ckpt_path = f'{output_dir}/'+args.save_model_name_tag+'_checkpoint_seed_'+str(args.seed)+'_context_len_'+str(args.context_length)+'.pt'
    best_path = f'{output_dir}/'+args.save_model_name_tag+'_checkpoint_best_seed_'+str(args.seed)+'_context_len_'+str(args.context_length)+'.pt'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, best_path)


def sanity_check(state_dict, pretrained_weights, linear_keyword, visual_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = visual_keyword + k[len('module.'):] \
            if k.startswith('module.') else visual_keyword + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear probe evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
