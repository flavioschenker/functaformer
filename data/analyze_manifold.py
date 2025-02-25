import os
import torch
import wandb
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cartopy
import cartopy.crs as ccrs
from datasets import ERA5
from models import SuperSiren, FunctaTransformer
from metrics import manifold_psnr

def test_interpolation_baseline() -> None:

    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = ERA5(data_dir, train=False, windowed=False, dim_window=[1,1], dim_slice=[1,1])
    testset = dataset.data.to(device)
    b, c, lat, lon = testset.shape

    data_x2 = torch.nn.functional.interpolate(testset, size=[lat//2, lon//2], mode='bilinear')
    data_x3 = torch.nn.functional.interpolate(testset, size=[lat//3, lon//3], mode='bilinear')
    data_x4 = torch.nn.functional.interpolate(testset, size=[lat//4, lon//4], mode='bilinear')

    print(testset.shape, data_x2.shape, data_x3.shape, data_x4.shape)

    data_x2_trilinear = torch.nn.functional.interpolate(data_x2, size=[lat, lon], mode='bilinear')
    data_x2_nearest = torch.nn.functional.interpolate(data_x2, size=[lat, lon],mode='nearest-exact')
    data_x3_trilinear = torch.nn.functional.interpolate(data_x3, size=[lat, lon], mode='bilinear')
    data_x3_nearest = torch.nn.functional.interpolate(data_x3, size=[lat, lon],mode='nearest-exact')
    data_x4_trilinear = torch.nn.functional.interpolate(data_x4, size=[lat, lon], mode='bilinear')
    data_x4_nearest = torch.nn.functional.interpolate(data_x4, size=[lat, lon],mode='nearest-exact')

    accuracy_x2_trilinear = manifold_psnr(testset, data_x2_trilinear, 0)
    accuracy_x2_nearest = manifold_psnr(testset, data_x2_nearest, 0)
    accuracy_x3_trilinear = manifold_psnr(testset, data_x3_trilinear, 0)
    accuracy_x3_nearest = manifold_psnr(testset, data_x3_nearest, 0)
    accuracy_x4_trilinear = manifold_psnr(testset, data_x4_trilinear, 0)
    accuracy_x4_nearest = manifold_psnr(testset, data_x4_nearest, 0)
    
    print('simple interpolation baseline for Manifold testset')
    print(f'x2 bilinear: {accuracy_x2_trilinear:.3f}, nearest: {accuracy_x2_nearest:.3f}')
    print(f'x3 bilinear: {accuracy_x3_trilinear:.3f}, nearest: {accuracy_x3_nearest:.3f}')
    print(f'x4 bilinear: {accuracy_x4_trilinear:.3f}, nearest: {accuracy_x4_nearest:.3f}')


def generate_samples():
    scale = 4
    device = torch.device('cuda:0')
    data_dir = '/your/path/to/the/data/folder/'
    dataset = ERA5(data_dir, train=False, windowed=False, dim_window=[2,2], dim_slice=[6,6])
    functa_model = SuperSiren(
        device=device,
        dim_input=2,
        dim_output=1,
        dim_window=[6,6],
        dim_hidden=64,
        dim_layers=3,
        dim_functa=36,
        inner_steps=3,
        omega=30,
        activation='siren',
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'functa_manifold_1012975.pt'))
    functa_model.load_state_dict(state_dict)
    upscale_model = FunctaTransformer(
        dim_data=[30,60],
        dim_window=[2,2],
        dim_functa=36,
        dim_embedding=180,
        dim_blocks=[6,6,6,6,6,6],
        dim_hidden=360,
        num_head=6,
        drop_conn=0.1,
        drop_attn=0,
        drop_ffn=0,
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'upscale_manifold_1015988.pt'))
    upscale_model.load_state_dict(state_dict)
    upscale_model.eval()

    os.environ['WANDB_DIR'] = os.path.join(data_dir)
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=f'ma_samples', config={'modality':'manifold','scale':scale})
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    latitude = torch.linspace(90, -90, steps=180)
    longitude = torch.linspace(0, 359, steps=360)


    for i, (cords, data_hr) in enumerate(loader):
        print(f'{i+1:>4}/{len(loader)}')
        cords, data_hr = cords.to(device), data_hr.to(device)
        data_lr = dataset.downsample(data_hr, scale)
        pred_hr, loss_hr, functa_hr = functa_model(cords, data_hr)
        pred_lr, loss_lr, functa_lr = functa_model(cords, data_lr)
        with torch.no_grad():
            functa_hr_mean = torch.mean(functa_hr, dim=1, keepdim=True)
            functa_hr_std = torch.std(functa_hr, dim=1, keepdim=True)
            functa_lr_mean = torch.mean(functa_lr, dim=1, keepdim=True)
            functa_lr_std = torch.std(functa_lr, dim=1, keepdim=True)
            functa_hr = (functa_hr - functa_hr_mean) / functa_hr_std
            functa_lr = (functa_lr - functa_lr_mean) / functa_lr_std
            functa_pr = upscale_model(functa_lr)
            functa_pr = functa_pr * functa_hr_std + functa_hr_mean
            pred_pr = functa_model.decode_functa(cords, functa_pr)

            pred_hr = torch.clamp(pred_hr, 0, 1)
            pred_lr = torch.clamp(pred_lr, 0, 1)
            pred_pr = torch.clamp(pred_pr, 0, 1)
            pred_hr = pred_hr[:,-1]
            pred_lr = pred_lr[:,-1]
            pred_pr = pred_pr[:,-1]
            data_hr = dataset.merge(data_hr)
            data_lr = dataset.merge(data_lr)
            pred_hr = dataset.merge(pred_hr)
            pred_lr = dataset.merge(pred_lr)
            pred_pr = dataset.merge(pred_pr)
            data_hr = data_hr[0,0]
            data_lr = data_lr[0,0]
            pred_hr = pred_hr[0,0]
            pred_lr = pred_lr[0,0]
            pred_pr = pred_pr[0,0]
            manifold = torch.stack([data_hr,data_lr,pred_hr,pred_lr,pred_pr])
            data_hr, data_lr, pred_hr, pred_lr, pred_pr = dataset.manifold_to_image(latitude, longitude, manifold)

            wandb.log({
                'data_hr': data_hr,
                'data_lr': data_lr,
                'pred_hr': pred_hr,
                'pred_lr': pred_lr,
                'pred_pr': pred_pr,
            })

        del cords, data_hr, data_lr
        del pred_hr, pred_lr, pred_pr
        del loss_hr, loss_lr
        del functa_hr, functa_lr, functa_hr_mean, functa_hr_std, functa_lr_mean, functa_lr_std, functa_pr
    wandb.finish()

def test_naive_upscale(
    scale:int
) -> None:
    dim_slice_lr = [3, 6]
    dim_slice_hr = [scale*3, scale*6]
    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = ERA5(data_dir, train=False, windowed=False, dim_window=[scale,scale], dim_slice=dim_slice_hr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    model = SuperSiren(
        device=device,
        dim_input=2,
        dim_output=1,
        dim_window=dim_slice_lr,
        dim_hidden=64,
        dim_layers=3,
        dim_functa=18,
        inner_steps=3,
        omega=30,
        activation='siren',
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'functa_manifold_1012032.pt'))
    model.load_state_dict(state_dict)

    psnrs = 0

    coordinates_hr = dataset.get_coordinates().to(device).unsqueeze(0)
    dataset.dim_slice = dim_slice_lr
    coordinates_lr = dataset.get_coordinates().to(device).unsqueeze(0)
    os.environ['WANDB_DIR'] = os.path.join(data_dir)
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=f'ma_samples', config={'modality':'manifold','scale':scale})
    latitude_hr = torch.linspace(90, -90, steps=180)
    longitude_hr = torch.linspace(0, 359, steps=360)
    latitude_lr = torch.linspace(90, -90, steps=180//scale)
    longitude_lr = torch.linspace(0, 359, steps=360//scale)

    for i, (_, data_hr) in enumerate(loader):
        print(f'{i:>4}/{len(loader)}', end='\r')
        data_hr = data_hr.to(device)
        data_hr = dataset.merge(data_hr)
        b, c, x, y = data_hr.shape
        data_lr = torch.nn.functional.interpolate(data_hr, size=[x//scale, y//scale], mode='bilinear')
        data_lr = dataset.partition_batch(data_lr)
        model.dim_window = dim_slice_lr
        pred_lr, loss_lr, functa_lr = model(coordinates_lr, data_lr)
        model.dim_window = dim_slice_hr
        pred_hr = model.decode_functa(coordinates_hr, functa_lr)
        pred_hr = torch.clamp(pred_hr, 0, 1)
        pred_lr = torch.clamp(pred_lr, 0, 1)
        pred_lr = dataset.merge(pred_lr[:,-1])
        pred_hr = dataset.merge(pred_hr[:,-1])
        data_lr = dataset.merge(data_lr)
        pred_hr = dataset.cut(pred_hr)
        data_hr = dataset.cut(data_hr)
        dataset.dim_data = [180//scale, 360//scale]
        pred_lr = dataset.cut(pred_lr)
        data_lr = dataset.cut(data_lr)
        dataset.dim_data = [180,360]
        psnr = manifold_psnr(data_hr, pred_hr, 0)
        psnrs += psnr

        if i < 20:

            manifold_hr = torch.stack([data_hr[0,0], pred_hr[0,0]])
            wandb_data_hr, wandb_pred_hr = dataset.manifold_to_image(latitude_hr, longitude_hr, manifold_hr)
            manifold_lr = torch.stack([data_lr[0,0], pred_lr[0,0]])
            wandb_data_lr, wandb_pred_lr = dataset.manifold_to_image(latitude_lr, longitude_lr, manifold_lr)

            wandb.log({
                'data_hr': wandb_data_hr,
                'data_lr': wandb_data_lr,
                'functa_hr': wandb_pred_hr,
                'functa_lr': wandb_pred_lr,
            })

    psnrs /= len(loader)
    wandb.finish()
    print(psnrs)

def sphere_samples():

    wandb.init(project=f'manifold')

    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = ERA5(data_dir, split='test', sliced=False, dim_window=[1,1], dim_slice=[1,1])
    hr = dataset.data.to(device)
    b, c, lat, lon = hr.shape

    lr = torch.nn.functional.interpolate(hr, size=[lat//4, lon//4], mode='bilinear')

    lat_hr = torch.linspace(90, -90, steps=180)
    lon_hr = torch.linspace(0, 359, steps=360)
    lat_lr = torch.linspace(90, -90, steps=180//4)
    lon_lr = torch.linspace(0, 359, steps=360//4)
    print(lr.shape, hr.shape)

    hr = hr.to('cpu')
    lr = lr.to('cpu')

    n = 20

    lon = torch.linspace(-180, 180, n, dtype=int)
    lat = torch.linspace(-90, 90, n, dtype=int)

    for i in range(n):
        print(i)
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.NearsidePerspective(lon[i].item(), lat[i].item()))
        ax.pcolormesh(lon_hr, lat_hr, hr[i,0], transform=ccrs.PlateCarree(), cmap='plasma', shading='auto')
        ax.coastlines()
        plt.tight_layout()
        hr_wandb = wandb.Image(plt)
        plt.close(fig)

        fig = plt.figure()
        ax = plt.axes(projection=ccrs.NearsidePerspective(lon[i].item(), lat[i].item()))
        ax.pcolormesh(lon_lr, lat_lr, lr[i,0], transform=ccrs.PlateCarree(), cmap='plasma', shading='auto')
        # ax.gridlines()
        ax.coastlines()
        plt.tight_layout()
        lr_wandb = wandb.Image(plt)
        plt.close(fig)

        wandb.log({
            'pr': hr_wandb,
            'lr': lr_wandb,
        })

sphere_samples()