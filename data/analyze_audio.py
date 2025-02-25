import os
import torch
import wandb
from torch.utils.data import DataLoader
from datasets import LIBS
from models import SuperSiren, FunctaTransformer
from metrics import audio_psnr
import torchaudio
import matplotlib.pyplot as plt

def test_interpolation_baseline() -> None:

    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = LIBS(data_dir, train=False, windowed=False, dim_window=[1], dim_slice=[1])
    testset = dataset.data.to(device)
    b, c, t = testset.shape

    data_x2 = torch.nn.functional.interpolate(testset, size=[t//2], mode='linear')
    data_x3 = torch.nn.functional.interpolate(testset, size=[t//3], mode='linear')
    data_x4 = torch.nn.functional.interpolate(testset, size=[t//4], mode='linear')

    print(testset.shape, data_x2.shape, data_x3.shape, data_x4.shape)

    data_x2_trilinear = torch.nn.functional.interpolate(data_x2, size=[t], mode='linear')
    data_x2_nearest = torch.nn.functional.interpolate(data_x2, size=[t],mode='nearest-exact')
    data_x3_trilinear = torch.nn.functional.interpolate(data_x3, size=[t], mode='linear')
    data_x3_nearest = torch.nn.functional.interpolate(data_x3, size=[t],mode='nearest-exact')
    data_x4_trilinear = torch.nn.functional.interpolate(data_x4, size=[t], mode='linear')
    data_x4_nearest = torch.nn.functional.interpolate(data_x4, size=[t],mode='nearest-exact')

    accuracy_x2_trilinear = audio_psnr(testset, data_x2_trilinear, 0)
    accuracy_x2_nearest = audio_psnr(testset, data_x2_nearest, 0)
    accuracy_x3_trilinear = audio_psnr(testset, data_x3_trilinear, 0)
    accuracy_x3_nearest = audio_psnr(testset, data_x3_nearest, 0)
    accuracy_x4_trilinear = audio_psnr(testset, data_x4_trilinear, 0)
    accuracy_x4_nearest = audio_psnr(testset, data_x4_nearest, 0)
    
    print('simple interpolation baseline for Audio testset')
    print(f'x2 linear: {accuracy_x2_trilinear:.3f}, nearest: {accuracy_x2_nearest:.3f}')
    print(f'x3 linear: {accuracy_x3_trilinear:.3f}, nearest: {accuracy_x3_nearest:.3f}')
    print(f'x4 linear: {accuracy_x4_trilinear:.3f}, nearest: {accuracy_x4_nearest:.3f}')



def test_naive_upscale(
    scale:int
) -> None:
    dim_slice_lr = [1000]
    dim_slice_hr = [scale*1000]
    data_dir = '/your/path/to/the/data/folder/'
    device = torch.device('cuda:0')
    dataset = LIBS(data_dir, train=False, windowed=False, dim_window=[scale], dim_slice=dim_slice_hr)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)
    model = SuperSiren(
        device=device,
        dim_input=1,
        dim_output=1,
        dim_window=dim_slice_lr,
        dim_hidden=64,
        dim_layers=3,
        dim_functa=1000,
        inner_steps=3,
        omega=50,
        activation='siren',
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'functa_audio_1011256.pt'))
    model.load_state_dict(state_dict)

    psnrs = 0

    coordinates_hr = dataset.get_coordinates().to(device).unsqueeze(0)
    dataset.dim_slice = dim_slice_lr
    coordinates_lr = dataset.get_coordinates().to(device).unsqueeze(0)
    os.environ['WANDB_DIR'] = os.path.join(data_dir)
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=f'ma_samples', config={'modality':'audio','scale':scale})

    for i, (_, data_hr) in enumerate(loader):
        print(f'{i:>4}/{len(loader)}', end='\r')
        data_hr = data_hr.to(device)
        data_hr = dataset.merge(data_hr)
        b, c, t = data_hr.shape
        data_lr = torch.nn.functional.interpolate(data_hr, size=[t//scale], mode='linear')
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
        dataset.dim_data = [60000//scale]
        pred_lr = dataset.cut(pred_lr)
        data_lr = dataset.cut(data_lr)
        dataset.dim_data = [60000]
        psnr = audio_psnr(data_hr, pred_hr, 0)
        psnrs += psnr

        if i < 20:

            data_hr = data_hr*2 - 1
            data_lr = data_lr*2 - 1
            pred_hr = pred_hr*2 - 1
            pred_lr = pred_lr*2 - 1

            wandb.log({
                'data_hr': wandb.Audio(data_hr[0,0].cpu(), 12000),
                'data_lr': wandb.Audio(data_lr[0,0].cpu(), 12000),
                'functa_hr': wandb.Audio(pred_hr[0,0].cpu(), 12000),
                'functa_lr': wandb.Audio(pred_lr[0,0].cpu(), 12000),
            })

    psnrs /= len(loader)
    wandb.finish()
    print(psnrs)

def mel_to_image(mel:torch.Tensor):
    mel = mel.numpy(force=True)
    plt.figure(figsize=(10, 4),dpi=200)
    plt.imshow(mel, aspect='auto', origin='lower', cmap='inferno')
    plt.tight_layout()
    plt.axis('off')
    result = wandb.Image(plt)
    plt.close()
    return result

def generate_samples():
    scale = 4
    device = torch.device('cuda:0')
    data_dir = '/your/path/to/the/data/folder/'
    dataset = LIBS(data_dir, train=False, windowed=False, dim_window=[2], dim_slice=[3])
    functa_model = SuperSiren(
        device=device,
        dim_input=1,
        dim_output=1,
        dim_window=[3],
        dim_hidden=64,
        dim_layers=3,
        dim_functa=3,
        inner_steps=3,
        omega=50,
        activation='siren',
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'functa_audio_1013267.pt'))
    functa_model.load_state_dict(state_dict)
    upscale_model = FunctaTransformer(
        dim_data=[20000],
        dim_window=[2],
        dim_functa=3,
        dim_embedding=180,
        dim_blocks=[6,6,6,6,6,6],
        dim_hidden=360,
        num_head=6,
        drop_conn=0.1,
        drop_attn=0,
        drop_ffn=0,
    ).to(device)
    state_dict = torch.load(os.path.join(data_dir,'models',f'upscale_audio_1015991.pt'))
    upscale_model.load_state_dict(state_dict)
    upscale_model.eval()

    os.environ['WANDB_DIR'] = os.path.join(data_dir)
    os.environ['WANDB_SILENT'] = 'true'
    wandb.init(project=f'ma_samples', config={'modality':'audio','scale':scale})
    loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=12000,
        n_fft=512,
    ).to(device)
    db_transform = torchaudio.transforms.AmplitudeToDB().to(device)

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
            data_hr = data_hr*2 - 1
            data_lr = data_lr*2 - 1
            pred_hr = pred_hr*2 - 1
            pred_lr = pred_lr*2 - 1
            pred_pr = pred_pr*2 - 1

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
            data_hr = db_transform(mel_spectrogram_transform(data_hr))
            data_lr = db_transform(mel_spectrogram_transform(data_lr))
            pred_hr = db_transform(mel_spectrogram_transform(pred_hr))
            pred_lr = db_transform(mel_spectrogram_transform(pred_lr))
            pred_pr = db_transform(mel_spectrogram_transform(pred_pr))
            data_hr = mel_to_image(data_hr)
            data_lr = mel_to_image(data_lr)
            pred_hr = mel_to_image(pred_hr)
            pred_lr = mel_to_image(pred_lr)
            pred_pr = mel_to_image(pred_pr)

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


generate_samples()