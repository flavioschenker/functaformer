import os
import torch
import wandb
from metrics import audio_psnr

class LIBS(torch.utils.data.Dataset):
    def __init__(self,
        data_dir:str,
        train:bool,
        windowed:bool,
        dim_slice:list[int],
        dim_window:list[int],
    ) -> None:
        super().__init__()

        self.train = train
        self.windowed = windowed
        self.dim_slice = dim_slice
        self.dim_window = dim_window
        self.dim_data = [60000]
        self.dim_data_pad = self.calculate_pad()
        self.coordinates = self.get_coordinates()

        if train:
            self.path = os.path.join(data_dir,'datasets','LIBRISPEECH','tensor','libs_train.pt')
            self.data = torch.load(self.path)
            print(f'Libris Speech train loaded: {self.data.shape}')
        else:
            self.path = os.path.join(data_dir,'datasets','LIBRISPEECH','tensor','libs_test.pt')
            self.data = torch.load(self.path)
            print(f'Libris Speech test loaded: {self.data.shape}')


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, i:int) -> torch.Tensor:
        data = self.data[i]                                                 # (c, t)
        if self.windowed:
            data = self.random_crop(data)                                   # (c, wt)
        else:
            data = self.pad(data)                                           # (c, t)
        data = self.partition(data)                                         # (nt, c, wt)
        return self.coordinates, data


    def get_coordinates(self) -> torch.Tensor:
        t, = self.dim_slice
        st = torch.linspace(-50, 50, steps=t)
        coordinates = torch.meshgrid([st], indexing='ij')
        coordinates = torch.stack(coordinates, dim=-1)
        coordinates = torch.flatten(coordinates, end_dim=-2)
        coordinates = coordinates.detach().clone()
        return coordinates.requires_grad_(True)
  

    def get_metric(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last inner step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        psnr = audio_psnr(data_true, data_pred, 0)
        return psnr


    def get_wandb(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> tuple[wandb.Image, wandb.Image]:
        data_pred = data_pred[0]                                            # first in batch
        data_true = data_true[0]                                            # first in batch
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_true = torch.unsqueeze(data_true, 0)                           # (1, nt, c, wt)
        data = torch.cat((data_pred, data_true))                            # (s+2, nt, c, wt)
        data = self.merge(data)                                             # (s+2, c, t)
        data = self.cut(data)                                               # (s+2, c, t)
        data = data*2 - 1                                                   # denormalize to [-1, 1]
        data = data.cpu()
        return wandb.Audio(data[-2,0], 12000), wandb.Audio(data[-1,0], 12000)
    

    def partition(self, data:torch.Tensor) -> torch.Tensor:
        c, t = data.shape
        wt, = self.dim_slice
        nt = t//wt
        data = torch.reshape(data, (c, nt, wt))                             # (c, nt, wt)
        data = torch.permute(data, (1, 0, 2))                               # (nt, c, wt)
        return data.detach()
    

    def partition_batch(self, data:torch.Tensor) -> torch.Tensor:
        b, c, t = data.shape
        wt, = self.dim_slice
        nt = t//wt
        data = torch.reshape(data, (b, c, nt, wt))                          # (b, c, nt, wt)
        data = torch.permute(data, (0, 2, 1, 3))                            # (b, nt, c, wt)
        return data.detach()
    

    def merge(self, data:torch.Tensor) -> torch.Tensor:
        b, nt, c, wt = data.shape
        data = torch.permute(data, (0, 2, 1, 3))                            # (b, c, nt, wt)
        data = torch.reshape(data, (b, c, nt*wt))                           # (b, c, t)
        return data.detach()


    def cut(self, data:torch.Tensor) -> torch.Tensor:
        t, = self.dim_data
        assert data.ndim == 3
        return data[:,:,:t].detach()


    def pad(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2
        c, t = data.shape
        st, = self.dim_slice
        wt, = self.dim_window
        pt = st*wt
        pt = (pt - (t%pt))%pt
        result = torch.zeros((c, t+pt)).to(data)
        result[:,:t] = data
        return result.detach()
    

    def calculate_pad(self) -> tuple[int, int ,int]:
        st, = self.dim_slice
        wt, = self.dim_window
        pt = st*wt
        t, = self.dim_data
        return [(((t+pt-1)//pt)*pt)//st]


    def random_crop(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 2
        c, t = data.shape
        ct, = self.dim_slice
        st = torch.randint(0, max(0, t - ct), (1,))
        data = data[:, st:st+ct]
        return data.detach()
    

    def downsample(self, data_hr:torch.Tensor, scale:int) -> torch.Tensor:
        assert data_hr.ndim == 4
        assert scale in [2,3,4]
        b, nt, c, wt = data_hr.shape
        data_hr = self.merge(data_hr)
        b, c, t = data_hr.shape

        if scale==2:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[t//2], mode='linear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[t], mode='linear')
        elif scale==3:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[t//3], mode='linear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[t], mode='linear')
        elif scale==4:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[t//4], mode='linear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[t], mode='linear')
            
        data_lr = self.partition_batch(data_lr)
        return data_lr.detach()