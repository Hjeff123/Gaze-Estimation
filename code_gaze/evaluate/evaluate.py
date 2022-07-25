import os
import time
import logging
from models.lenet_cap import Model
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn
from main import AverageMeter,compute_angle_error
from dataloader import MPIIGazeDataset

def get_test_loader(dataset_dir, test_subject_id, batch_size=64, num_workers=7, use_gpu=True):
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)
    subject_ids = ['p{:02}'.format(index) for index in range(15)]
    test_subject_id = subject_ids[test_subject_id]

    test_dataset = MPIIGazeDataset(test_subject_id, dataset_dir)

    assert len(test_dataset) == 3000

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return test_loader

def test(model, criterion, test_loader):
    logging.basicConfig(
        format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.info('========evaluating========')
    model.eval()
    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, poses, gazes) in enumerate(test_loader):

        images = images.cuda()
        poses = poses.cuda()
        gazes = gazes.cuda()
        with torch.no_grad():
            outputs = model(images, poses)
        loss = criterion(outputs, gazes)

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

    logger.info(' Loss {:.4f} AngleError {:.2f}'.format(loss_meter.avg, angle_error_meter.avg))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    return angle_error_meter.avg

if __name__ == '__main__':
    model_path="E:\DeepLearning\Gaze_Tracking\Code_GazeTracking\\result\lenet\\00\model_state.pth"
    test_id = 3
    dataset_path="../../data"
    model = Model()
    model.load_state_dict(torch.load(model_path)["state_dict"])
    model.cuda()
    test_loader = get_test_loader(dataset_path,test_id)
    criterion = nn.MSELoss(size_average=True)
    test(model, criterion, test_loader)