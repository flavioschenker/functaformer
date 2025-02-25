import torch
from datasets import ADOBE240

def test_interpolation_baseline() -> None:

    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = ADOBE240(data_dir, split='test', sliced=False, dim_window=[1,1,1], dim_slice=[1,1,1])
    psnrs_trilinear = 0
    psnrs_nearest = 0
    ssims_trilinear = 0
    ssims_nearest = 0
    for i in range(len(dataset.paths)):
        print(i, end='\r')
        video_hr = torch.load(dataset.paths[i], weights_only=True)
        video_hr = video_hr.to(device)

        c, t, x, y = video_hr.shape

        video_hr = torch.permute(video_hr, (1,0,2,3))       # (t, c, x, y)
        video_hr = video_hr[:225]
        video_hr = torch.reshape(video_hr, (9,25, c, x, y))   # (9, 25, c, x, y)
        video_hr = torch.permute(video_hr, (0,2,1,3,4))       # (9, c, 25, x, y)

        b, c, t, x, y = video_hr.shape

        video_hr = video_hr / 255.

        video_lr = video_hr.clone()
        video_lr = video_lr[:,:,[0,8,16,24]]
        video_lr = torch.permute(video_lr, (0,2,1,3,4))       # (b, t, c, x, y)
        video_lr = torch.reshape(video_lr, (b*4, c, x, y))    # (b*t, c, x, y)
        video_lr = torch.nn.functional.interpolate(video_lr, size=[x//4, y//4], mode='bilinear')
        video_lr = torch.reshape(video_lr, (b, 4, c, x//4, y//4))   # (b, t, c, x, y)
        video_lr = torch.permute(video_lr, (0,2,1,3,4))       # (b, c, t, x, y)

        trilinear = torch.nn.functional.interpolate(video_lr, size=[t, x, y], mode='trilinear')
        nearest = torch.nn.functional.interpolate(video_lr, size=[t, x, y],mode='nearest-exact')

        psnr_trilinear = dataset.psnr(video_hr, trilinear)
        psnr_nearest = dataset.psnr(video_hr, nearest)
        ssim_trilinear = dataset.ssim(video_hr, trilinear)
        ssim_nearest = dataset.ssim(video_hr, nearest)

        psnrs_trilinear += psnr_trilinear
        psnrs_nearest += psnr_nearest
        ssims_trilinear += ssim_trilinear
        ssims_nearest += ssim_nearest

    psnrs_trilinear /= len(dataset.paths)
    psnrs_nearest /= len(dataset.paths)
    ssims_trilinear /= len(dataset.paths)
    ssims_nearest /= len(dataset.paths)

    print('simple interpolation baseline for lidar testset')
    print(f'psnr: {psnrs_trilinear:.4f}, nearest: {psnrs_nearest:.4f}')
    print(f'ssim trilinear: {ssims_trilinear:.4f}, nearest: {ssims_nearest:.4f}')

with torch.no_grad():
    test_interpolation_baseline()