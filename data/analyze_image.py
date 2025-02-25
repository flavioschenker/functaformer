import os
import torch
import wandb
from torch.utils.data import DataLoader
from models import SuperSiren, FunctaTransformer
from datasets import DF2K, SET5, SET14, URBAN100, BSD100, MANGA109
from metrics import image_psnr, image_ssim

def test_naive_upscale(
    dataset,
    scale:int
) -> None:
    dim_slice_lr = [2, 2]
    dim_slice_hr = [scale*2, scale*2]
    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset.dim_slice = dim_slice_hr
    dataset.dim_window = [scale, scale]
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    model = SuperSiren(
        device=device,
        dim_input=2,
        dim_output=3,
        dim_window=[2,2],
        dim_hidden=64,
        dim_layers=3,
        dim_functa=12,
        inner_steps=3,
        omega=30,
        activation='siren',
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'functa_image_1012039.pt'))
    model.load_state_dict(state_dict)

    psnrs = 0

    coordinates_hr = dataset.get_coordinates().to(device).unsqueeze(0)
    dataset.dim_slice = dim_slice_lr
    coordinates_lr = dataset.get_coordinates().to(device).unsqueeze(0)
    os.environ['WANDB_DIR'] = os.path.join(data_dir)
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=f'ma_samples', config={'modality':'shape','scale':scale})

    for i, (cords, data_hr) in enumerate(loader):
        print(f'{i:>4}/{len(loader)}', end='\r')
        coordinates_lr = coordinates_lr.detach()
        coordinates_hr = coordinates_hr.detach()
        data_hr = data_hr.to(device)
        data_hr = dataset.merge(data_hr)
        b, c, x, y = data_hr.shape
        data_lr = torch.nn.functional.interpolate(data_hr, size=[x//scale, y//scale], mode='bilinear')
        print(data_hr.shape, data_lr.shape)
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

        psnr = image_psnr(data_hr, pred_hr, 0)
        psnr_lr = image_psnr(data_lr, pred_lr, 0)
        psnrs += psnr
        print(pred_hr.shape, psnr, psnr_lr)
        wandb.log({
            'data_hr': wandb.Image(data_hr),
            'data_lr': wandb.Image(data_lr),
            'functa_hr': wandb.Image(pred_hr),
            'functa_lr': wandb.Image(pred_lr),
            'psnr': psnr,
        })

        del cords, data_hr, data_lr
        del pred_hr, pred_lr
        del loss_lr
        del functa_lr


    psnrs /= len(loader)
    print(psnrs)
    wandb.finish()

def test_interpolation_baseline(dataset) -> None:
    device = torch.device('cuda:0')
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    psnr_x2_bilinear = 0
    psnr_x2_nearest = 0
    psnr_x3_bilinear = 0
    psnr_x3_nearest = 0
    psnr_x4_bilinear = 0
    psnr_x4_nearest = 0
    ssim_x2_bilinear = 0
    ssim_x2_nearest = 0
    ssim_x3_bilinear = 0
    ssim_x3_nearest = 0
    ssim_x4_bilinear = 0
    ssim_x4_nearest = 0

    for i, (cords, data_hr) in enumerate(loader):
        data_hr = data_hr.to(device)
        data_hr = dataset.merge(data_hr)
        b, c, x, y = data_hr.shape
        data_x2 = torch.nn.functional.interpolate(data_hr, size=[x//2, y//2], mode='bilinear')
        data_x3 = torch.nn.functional.interpolate(data_hr, size=[x//3, y//3], mode='bilinear')
        data_x4 = torch.nn.functional.interpolate(data_hr, size=[x//4, y//4], mode='bilinear')
        print(data_hr.shape, data_x2.shape, data_x3.shape, data_x4.shape)
        data_x2_bilinear = torch.nn.functional.interpolate(data_x2, size=[x, y], mode='bilinear')
        data_x2_nearest = torch.nn.functional.interpolate(data_x2, size=[x, y],mode='nearest-exact')
        data_x3_bilinear = torch.nn.functional.interpolate(data_x3, size=[x, y], mode='bilinear')
        data_x3_nearest = torch.nn.functional.interpolate(data_x3, size=[x, y],mode='nearest-exact')
        data_x4_bilinear = torch.nn.functional.interpolate(data_x4, size=[x, y], mode='bilinear')
        data_x4_nearest = torch.nn.functional.interpolate(data_x4, size=[x, y],mode='nearest-exact')

        psnr_x2_bilinear += image_psnr(data_hr, data_x2_bilinear, 0)
        psnr_x2_nearest += image_psnr(data_hr, data_x2_nearest, 0)
        psnr_x3_bilinear += image_psnr(data_hr, data_x3_bilinear, 0)
        psnr_x3_nearest += image_psnr(data_hr, data_x3_nearest, 0)
        psnr_x4_bilinear += image_psnr(data_hr, data_x4_bilinear, 0)
        psnr_x4_nearest += image_psnr(data_hr, data_x4_nearest, 0)
        ssim_x2_bilinear += image_ssim(data_hr, data_x2_bilinear, 0)
        ssim_x2_nearest += image_ssim(data_hr, data_x2_nearest, 0)
        ssim_x3_bilinear += image_ssim(data_hr, data_x3_bilinear, 0)
        ssim_x3_nearest += image_ssim(data_hr, data_x3_nearest, 0)
        ssim_x4_bilinear += image_ssim(data_hr, data_x4_bilinear, 0)
        ssim_x4_nearest += image_ssim(data_hr, data_x4_nearest, 0)

    psnr_x2_bilinear /= len(loader)
    psnr_x2_nearest /= len(loader)
    psnr_x3_bilinear /= len(loader)
    psnr_x3_nearest /= len(loader)
    psnr_x4_bilinear /= len(loader)
    psnr_x4_nearest /= len(loader)
    ssim_x2_bilinear /= len(loader)
    ssim_x2_nearest /= len(loader)
    ssim_x3_bilinear /= len(loader)
    ssim_x3_nearest /= len(loader)
    ssim_x4_bilinear /= len(loader)
    ssim_x4_nearest /= len(loader)


    print('simple interpolation baseline for image testset')
    print(f'x2: bl psnr {psnr_x2_bilinear:.4f} bl ssim {ssim_x2_bilinear:.4f}, nn psnr: {psnr_x2_nearest:.4f} nn ssim {ssim_x2_nearest:.4f}')
    print(f'x3: bl psnr {psnr_x3_bilinear:.4f} bl ssim {ssim_x3_bilinear:.4f}, nn psnr: {psnr_x3_nearest:.4f} nn ssim {ssim_x3_nearest:.4f}')
    print(f'x4: bl psnr {psnr_x4_bilinear:.4f} bl ssim {ssim_x4_bilinear:.4f}, nn psnr: {psnr_x4_nearest:.4f} nn ssim {ssim_x4_nearest:.4f}')

def analyze_testset(dataset, scale:int, functa_model, upscale_model) -> None:
    device = torch.device('cuda:0')
    data_dir = '/your/path/to/the/data/folder/'

    psnrs = 0
    ssims = 0
    os.environ['WANDB_DIR'] = os.path.join(data_dir)
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=f'ma_samples', config={'modality':'image','scale':scale})
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)


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

            pred_hr = pred_hr[:,-1]
            pred_lr = pred_lr[:,-1]
            pred_pr = pred_pr[:,-1]
            pred_pr = torch.clamp(pred_pr, 0, 1)
            pred_hr = dataset.merge(pred_hr)
            pred_lr = dataset.merge(pred_lr)
            pred_pr = dataset.merge(pred_pr)
            data_hr = dataset.merge(data_hr)
            data_lr = dataset.merge(data_lr)
            psnrs += image_psnr(data_hr, pred_pr, 0)
            ssims += image_ssim(data_hr, pred_pr, 0)

            if i < 150:
                wandb.log({
                    'data_hr': wandb.Image(data_hr),
                    'data_lr': wandb.Image(data_lr),
                    'pred_hr': wandb.Image(pred_hr),
                    'pred_lr': wandb.Image(pred_lr),
                    'pred_pr': wandb.Image(pred_pr),
                })

        del cords, data_hr, data_lr
        del pred_hr, pred_lr, pred_pr
        del loss_hr, loss_lr
        del functa_hr, functa_lr, functa_hr_mean, functa_hr_std, functa_lr_mean, functa_lr_std, functa_pr

    psnrs /= len(loader)
    ssims /= len(loader)
    print(f'{testset}: psnr {psnrs:.3f}, ssim {ssims:.4f}')
    wandb.finish()



data_dir = '/your/path/to/the/data/folder/'
device = torch.device('cuda:0')
functa_id = '1012976'
upscale_id = '1015966'
dim_slice = [3,3]
dim_window = [12,12]

testset = 'df2k'
if testset == 'df2k':
    dataset = DF2K(data_dir, train=False, windowed=False, dim_window=dim_window, dim_slice=dim_slice)
elif testset == 'set5':
    dataset = SET5(data_dir, dim_window=dim_window, dim_slice=dim_slice)
elif testset == 'set14':
    dataset = SET14(data_dir, dim_window=dim_window, dim_slice=dim_slice)
elif testset == 'bsd100':
    dataset = BSD100(data_dir, dim_window=dim_window, dim_slice=dim_slice)
elif testset == 'urban100':
    dataset = URBAN100(data_dir, dim_window=dim_window, dim_slice=dim_slice)
elif testset == 'manga109':
    dataset = MANGA109(data_dir, dim_window=dim_window, dim_slice=dim_slice)

functa_model = SuperSiren(
    device=device,
    dim_input=2,
    dim_output=3,
    dim_window=dim_slice,
    dim_hidden=64,
    dim_layers=3,
    dim_functa=27,
    inner_steps=3,
    omega=30,
    activation='siren',
).to(device)
state_dict = torch.load(os.path.join(data_dir,'models',f'functa_image_{functa_id}.pt'))
functa_model.load_state_dict(state_dict)

upscale_model = FunctaTransformer(
    dim_data=[48,48],
    dim_window=dim_window,
    dim_functa=27,
    dim_embedding=180,
    dim_blocks=[6,6,6,6,6,6],
    dim_hidden=360,
    num_head=6,
    drop_conn=0.1,
    drop_attn=0,
    drop_ffn=0,
).to(device)
state_dict = torch.load(os.path.join(data_dir,'models',f'upscale_image_{upscale_id}.pt'))
upscale_model.load_state_dict(state_dict)
upscale_model.eval()


test_interpolation_baseline(dataset)
# test_naive_upscale(dataset, 2)
# analyze_testset(dataset, 4, functa_model, upscale_model)
