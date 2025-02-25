import os
import torch
import wandb
from datasets import SHAPENET

def test_interpolation_baseline() -> None:

    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = SHAPENET(data_dir, split='train', sliced=False, dim_window=[1,1,1], dim_slice=[1,1,1])
    testset = dataset.data.to(device)
    testset = testset.to(torch.float)

    testset = testset[32:34]

    b, c, x, y, z = testset.shape

    testset_lr = torch.nn.functional.interpolate(testset, size=[x//4, y//4, z//4], mode='trilinear')
    testset_lr = (testset_lr > 0.05)
    testset_lr = testset_lr.to(torch.float)

    trilinear = torch.nn.functional.interpolate(testset_lr.clone(), size=[x, y, z], mode='trilinear')
    trilinear = (trilinear > 0.05)
    nearest = torch.nn.functional.interpolate(testset_lr.clone(), size=[x, y, z],mode='nearest')
    nearest = (nearest > 0.05)


    iou_trilinear = dataset.iou(testset.clone(), trilinear.clone())
    iou_nearest = dataset.iou(testset.clone(), nearest.clone())
    f1_trilinear, pre_trilinear, rec_trilinear = dataset.f1(testset.clone(), trilinear.clone())
    f1_nearest, pre_nearest, rec_nearest = dataset.f1(testset.clone(), nearest.clone())
    
    print('simple interpolation baseline for 3D shapes testset')
    print(f'iou trilinear: {iou_trilinear:.4f}, nearest: {iou_nearest:.4f}')
    print(f'f1 trilinear: {f1_trilinear:.4f}, nearest: {f1_nearest:.4f}')
    print(f'precision trilinear: {pre_trilinear:.4f}, nearest: {pre_nearest:.4f}')
    print(f'recall trilinear: {rec_trilinear:.4f}, nearest: {rec_nearest:.4f}')

wandb.init(project=f'shape')
test_interpolation_baseline()