import os
import torch
import wandb
from metrics import manifold_psnr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


class ERA5(torch.utils.data.Dataset):
    def __init__(self,
        data_dir:str,
        split:str,
        sliced:bool,
        dim_slice:list[int],
        dim_window:list[int],
    ) -> None:
        super().__init__()

        self.sliced = sliced
        self.dim_slice = dim_slice
        self.dim_window = dim_window
        self.dim_data = [180, 360]
        self.dim_data_pad = self.calculate_pad()
        self.coordinates = self.get_coordinates()

        if split == 'train':
            self.path = os.path.join(data_dir,'datasets','era5','tensor','era5_train.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'ERA5 train loaded: {self.data.shape}')
        elif split == 'valid':
            self.path = os.path.join(data_dir,'datasets','era5','tensor','era5_valid.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'ERA5 test loaded: {self.data.shape}')
        else:
            self.path = os.path.join(data_dir,'datasets','era5','tensor','era5_test.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'ERA5 test loaded: {self.data.shape}')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i:int) -> torch.Tensor:
        data = self.data[i]                                         # (c, lat, lon)
        if self.sliced:
            data = self.random_crop(data)                           # (c, wa, wo)
        else:
            data = self.pad(data)                                   # (c, lat, lon)
        data = self.partition(data)                                 # (na, no, c, wa, wo)
        return self.coordinates, data
    

    def get_coordinates(self) -> torch.Tensor:
        a, o = self.dim_slice
        sa = torch.linspace(0. + 0.5/a, 1. - 0.5/a, steps=a)
        so = torch.linspace(0. + 0.5/o, 1. - 0.5/o, steps=o)
        coordinates = torch.meshgrid([sa, so], indexing='ij')
        coordinates = torch.stack(coordinates, dim=-1)
        coordinates = torch.flatten(coordinates, end_dim=-2)
        # x = torch.cos(coordinates[:,0]) * torch.cos(coordinates[:,1])
        # y = torch.cos(coordinates[:,0]) * torch.sin(coordinates[:,1])
        # z = torch.sin(coordinates[:,0])
        # coordinates = torch.stack([x,y,z], dim=1)
        coordinates = coordinates.detach().clone()
        return coordinates.requires_grad_(True)       


    def get_metric(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        psnr = manifold_psnr(data_true, data_pred, 0)
        return psnr


    def get_wandb(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> tuple[wandb.Image, wandb.Image]:
        latitude = torch.linspace(90, -90, steps=180)
        longitude = torch.linspace(0, 359, steps=360)
        data_pred = data_pred[0]                                            # first in batch
        data_true = data_true[0]                                            # first in batch
        data_pred = torch.clamp(data_pred, 0, 1)
        data_true = torch.unsqueeze(data_true, 0)                           # (1, na, no, c, wa, wo)
        data = torch.cat((data_pred, data_true))                            # (s+2, na, no, c, wa, wo)
        data = self.merge(data)                                             # (s+2, c, lat, lon)
        data = self.cut(data)                                               # (s+2, c, lat, lon)
        images = self.manifold_to_image(latitude, longitude, data[:,0])
        return images[:-1], images[-1]


    def partition(self, data:torch.Tensor) -> torch.Tensor:
        c, lat, lon = data.shape
        wa, wo = self.dim_slice
        na, no = lat//wa, lon//wo
        data = torch.reshape(data, (c, na, wa, no, wo))                     # (c, na, wa, no, wo)
        data = torch.permute(data, (1, 3, 0, 2, 4))                         # (na, no, c, wa, wo)
        return data.detach()


    def partition_batch(self, data:torch.Tensor) -> torch.Tensor:
        b, c, lat, lon = data.shape
        wa, wo = self.dim_slice
        na, no = lat//wa, lon//wo
        data = torch.reshape(data, (b, c, na, wa, no, wo))                  # (b, c, na, wa, no, wo)
        data = torch.permute(data, (0, 2, 4, 1, 3, 5))                      # (b, na, no, c, wa, wo)
        return data.detach()


    def merge(self, data:torch.Tensor) -> torch.Tensor:
        b, na, no, c, wa, wo = data.shape
        data = torch.permute(data, (0, 3, 1, 4, 2, 5))                      # (b, c, na, wa, no, wo)
        data = torch.reshape(data, (b, c, na*wa, no*wo))                    # (b, c, lat, lon)
        return data


    def cut(self, data:torch.Tensor) -> torch.Tensor:
        lat, lon = self.dim_data
        assert data.ndim == 4
        return data[:,:,:lat,:lon].detach()


    def pad(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 3
        c, lat, lon = data.shape
        slat, slon = self.dim_slice
        wlat, wlon = self.dim_window
        plat, plon = slat*wlat, slon*wlon
        plat = (plat-(lat%plat))%plat
        plon = (plon-(lon%plon))%plon
        result = torch.zeros((c, lat+plat, lon+plon)).to(data)
        result[:,:lat,:lon] = data
        return result.detach()


    def calculate_pad(self) -> tuple[int, int ,int]:
        slat, slon = self.dim_slice
        wlat, wlon = self.dim_window
        plat, plon = slat*wlat, slon*wlon
        lat, lon = self.dim_data
        return [(((lat+plat-1)//plat)*plat)//slat, (((lon+plon-1)//plon)*plon)//slon]


    def random_crop(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 3
        c, lat, lon = data.shape
        ca, co = self.dim_slice
        sa = torch.randint(0, max(0, lat - ca), (1,))
        so = torch.randint(0, max(0, lon - co), (1,))
        data = data[:, sa:sa+ca, so:so+co]
        return data.detach()


    def manifold_to_image(self, latitude:torch.Tensor, longitude:torch.Tensor, manifold:torch.Tensor) -> torch.Tensor:
        images = []
        manifold = manifold.cpu()
        for i in range(len(manifold)):
            fig = plt.figure()
            #NearsidePerspective
            ax = plt.axes(projection=ccrs.NearsidePerspective())
            ax.coastlines()
            ax.pcolormesh(longitude, latitude, manifold[i], cmap='plasma')
            plt.tight_layout()
            images.append(wandb.Image(plt))
            plt.close(fig)
        return images
    

    def downsample(self, data_hr:torch.Tensor, scale:int) -> torch.Tensor:
        assert data_hr.ndim == 6
        assert scale in [2,3,4]
        b, na, no, c, wa, wo = data_hr.shape
        data_hr = self.merge(data_hr)
        b, c, lat, lon = data_hr.shape

        if scale==2:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[lat//2, lon//2], mode='bilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[lat, lon], mode='bilinear')
        elif scale==3:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[lat//3, lon//3], mode='bilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[lat, lon], mode='bilinear')
        elif scale==4:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[lat//4, lon//4], mode='bilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[lat, lon], mode='bilinear')
            
        data_lr = self.partition_batch(data_lr)
        return data_lr.detach()