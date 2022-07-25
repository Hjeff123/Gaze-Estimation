#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
from collections import OrderedDict
import importlib
import logging
import argparse
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
import torchvision.utils
from torchvision import models
from torchsummary import summary
try:
    from tensorboardX import SummaryWriter
    is_tensorboard_available = True
except Exception:
    is_tensorboard_available = False

from image_loader import get_loader

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0

# train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# train_id = [0, 1, 2, 3]

mark=[]


def str2bool(s):
    if s.lower() == 'true':
        return True
    elif s.lower() == 'false':
        return False
    else:
        raise RuntimeError('Boolean value expected')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, choices=['Lenet_AiA', 'Lenet_Cap', 'Resnet', 'Resnet_AiA','SqueezeNet'], default='Lenet_AiA') # required=True,
    parser.add_argument('--dataset', type=str, default="../dataset/UTMultiview/") # required=True, MPIIGaze, UTMultiview, EyeDiap
    parser.add_argument('--test_id', type=int, default=test_id) # required=True, 0~14
    parser.add_argument('--outdir', type=str, default="./result/lenet/00") # required=True,
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--num_workers', type=int, default=7)
    # optimizer
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--base_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=str2bool, default=True)
    parser.add_argument('--milestones', type=str, default='[20, 30]')#
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', default=True)
    parser.add_argument('--no-tensorboard', dest='tensorboard', action='store_false')
    parser.add_argument('--tensorboard_images', action='store_true')
    parser.add_argument('--tensorboard_parameters', action='store_true')

    args = parser.parse_args()
    if not is_tensorboard_available:
        args.tensorboard = False
        args.tensorboard_images = False
        args.tensorboard_parameters = False

    assert os.path.exists(args.dataset)
    args.milestones = json.loads(args.milestones)

    return args


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count


def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    norm = torch.sqrt(x**2 + y**2 + z**2)
    x /= norm
    y /= norm
    z /= norm
    return x, y, z


def compute_angle_error(preds, labels):
    pred_x, pred_y, pred_z = convert_to_unit_vector(preds)
    label_x, label_y, label_z = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y + pred_z * label_z
    return torch.acos(angles) * 180 / np.pi


def train(epoch, model, optimizer, criterion, train_loader, config, writer):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, gazes, poses) in enumerate(train_loader):
        global_step += 1

        if config['tensorboard_images'] and step == 0:
            image = torchvision.utils.make_grid(
                images, normalize=True, scale_each=True)
            writer.add_image('Train/Image', image, epoch)

        images = images.cuda().float()
        gazes = gazes.cuda().float()
        poses = poses.cuda().float()

        optimizer.zero_grad()

        outputs, A, D = model(images, poses)
        # outputs = model(images)
        loss = criterion(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        if config['tensorboard']:
            writer.add_scalar('Train/RunningLoss', loss_meter.val, global_step)

        if step % 1 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'AngleError {:.2f} ({:.2f})'.format(
                            epoch,
                            step,
                            len(train_loader),
                            loss_meter.val,
                            loss_meter.avg,
                            angle_error_meter.val,
                            angle_error_meter.avg,)
                        )

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
        writer.add_scalar('Train/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, test_loader, config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, gazes, poses) in enumerate(test_loader):
        if config['tensorboard_images'] and epoch == 0 and step == 0:
            image = torchvision.utils.make_grid(images, normalize=True, scale_each=True)
            writer.add_image('Test/Image', image, epoch)

        images = images.cuda().float()
        gazes = gazes.cuda().float()
        poses = poses.cuda().float()

        with torch.no_grad():
            outputs, A, D = model(images, poses)
        #print(outputs.shape,gazes.shape)
        loss = criterion(outputs, gazes)

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

    logger.info('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(epoch, loss_meter.avg, angle_error_meter.avg))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    if config['tensorboard']:
        if epoch > 0:
            writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
            writer.add_scalar('Test/AngleError', angle_error_meter.avg, epoch)
        writer.add_scalar('Test/Time', elapsed, epoch)

    if config['tensorboard_parameters']:
        for name, param in model.named_parameters():
            writer.add_histogram(name, param, global_step)

    return angle_error_meter.avg, A, D


def main():
    min_error=1000
    epoch_lable=0
    args = parse_args()
    logger.info(json.dumps(vars(args), indent=2))

    # TensorBoard SummaryWriter
    writer = SummaryWriter() if args.tensorboard else None

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data loaders
    # train_loader, test_loader = get_loader(args.dataset, args.test_id, args.batch_size, args.num_workers, True)
    train_loader, test_loader = get_loader(args.dataset, train_id, args.test_id, args.batch_size, args.num_workers, True) # train [0,4,5,6,7,8], test [1,2,3]
    # model
    module = importlib.import_module('models.{}'.format(args.arch))
    model = module.Model()
    model.cuda()
    #summary(model,(1,36,60))

    criterion = nn.MSELoss(reduction='mean')

## optimizer
    # optimizer = torch.optim.SGD( #LeNet
    #     model.parameters(),
    #     lr=args.base_lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    #     nesterov=args.nesterov)
    optimizer = torch.optim.Adam( # ResNet
        model.parameters(),
        lr=args.base_lr,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=args.lr_decay)

    config = {
        'tensorboard': args.tensorboard,
        'tensorboard_images': args.tensorboard_images,
        'tensorboard_parameters': args.tensorboard_parameters,
    }

    # run test before start training
    #test(0, model, criterion, test_loader, config, writer)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        train(epoch, model, optimizer, criterion, train_loader, config, writer)
        angle_error, A, D = test(epoch, model, criterion, test_loader, config, writer)
        if angle_error<min_error:
            min_error=angle_error
            epoch_lable=epoch
            np.save('A.npy', A.cpu().numpy())
            np.save('D.npy', D.cpu().numpy())

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
        ])
        model_path = os.path.join(outdir, 'model_state.pth')
        torch.save(state, model_path)
        # torch.cuda.empty_cache()
        # sleep(100)
        # del(model)

    txtFile = "../dataset/prediction.txt"
    with open(txtFile, 'a') as f:
        f.write(str(test_id))
        f.write(": ")
        f.write(str(min_error))
        f.write('\n')
    f.close()

    dict={}
    dict["test_id"]=test_id
    dict["min_error"]=min_error
    dict["epoch"]=epoch_lable
    mark.append(dict)
    for dic in mark:
        print(dic)
    if args.tensorboard:
        outpath = os.path.join(outdir, 'all_scalars.json')
        writer.export_scalars_to_json(outpath)
    del (model)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    # testset = [1, 3, 5, 7, 8, 9, 12, 14]
    testset = [1]
    for i in testset:
        # train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] # EyeDiap
        # train_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]  # MPIIGaze
        train_id = [11, 12, 13, 14 ,15 ,16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45] # UTMultiview
    # i=2
        test_id = i
        train_id=[i for i in train_id if i not in [test_id]]
        print("test_id", i, "train", train_id)
        main()