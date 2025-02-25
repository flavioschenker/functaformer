import torch
from datasets import KITTY360

def test_interpolation_baseline() -> None:
    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = KITTY360(data_dir, split='test', sliced=False, dim_window=[1,1], dim_slice=[1,1])
    testset = dataset.data.to(device)
    testset[testset > 1.0] = 0.0

    b, c, x, y = testset.shape


    testset_lr = torch.zeros(b,c,x//4,y)
    testset_lr = testset[:,:,::4]

    print(testset_lr.shape, testset.shape)

    bilinear = torch.nn.functional.interpolate(testset_lr, size=[x, y], mode='bilinear')
    nearest = torch.nn.functional.interpolate(testset_lr, size=[x, y],mode='nearest-exact')
    print(bilinear.shape, nearest.shape)

    mae_bilinear = dataset.mean_abs_error(bilinear.clone(), testset.clone())
    psnr_bilinear = dataset.psnr(bilinear.clone(), testset.clone())
    mae_nearest = dataset.mean_abs_error(nearest.clone(), testset.clone())
    psnr_nearest = dataset.psnr(nearest.clone(), testset.clone())


    iou_bilinear = 0
    iou_nearest = 0
    for i in range(len(testset)):
            print(i)
            pc_hr = dataset.grid_to_pointcloud(testset[i,0])
            pc_bilinear = dataset.grid_to_pointcloud(bilinear[i,0])
            pc_nearest = dataset.grid_to_pointcloud(nearest[i,0])
            print(pc_hr.shape, pc_bilinear.shape, pc_nearest.shape)
            iou_bilinear += dataset.iou(pc_hr.clone(), pc_bilinear.clone())
            iou_nearest += dataset.iou(pc_hr.clone(), pc_nearest.clone())

    iou_bilinear /= len(testset)
    iou_nearest /= len(testset)
    
    print('simple interpolation baseline for lidar testset')
    print(f'iou trilinear: {iou_bilinear:.4f}, nearest: {iou_nearest:.4f}')
    print(f'mae trilinear: {mae_bilinear:.4f}, nearest: {mae_nearest:.4f}')
    print(f'psnr: {psnr_bilinear:.4f}, nearest: {psnr_nearest:.4f}')

test_interpolation_baseline()