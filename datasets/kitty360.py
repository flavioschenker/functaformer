import os
import torch
import wandb
from metrics import manifold_psnr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


class KITTY360(torch.utils.data.Dataset):
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
        self.dim_data = [64, 1024]
        self.dim_data_pad = self.calculate_pad()
        self.coordinates = self.get_coordinates()
        self.elevation_min = -24.8
        self.elevation_max = 2
        self.azimuth_min = -180
        self.azimuth_max = 180
        self.laser_max_range = 80
        self.laser_min_range = 1

        if split == 'train':
            self.path = os.path.join(data_dir,'datasets','kitty360','tensor','kitty360_train.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'KITTY360 train loaded: {self.data.shape}')
        elif split == 'valid':
            self.path = os.path.join(data_dir,'datasets','kitty360','tensor','kitty360_valid.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'KITTY360 valid loaded: {self.data.shape}')
        else:
            self.path = os.path.join(data_dir,'datasets','kitty360','tensor','kitty360_test.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'KITTY360 test loaded: {self.data.shape}')


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, i:int) -> torch.Tensor:
        data = self.data[i]                                         # (c, lat, lon)
        data[data > 1.0] = 0.0
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
        coordinates = coordinates.detach().clone()
        return coordinates.requires_grad_(True)


    def get_sample(self) -> torch.Tensor:
        id = torch.randint(0, len(self.data), (1,)).item()
        data = self.data[id]
        data[data > 1.0] = 0.0
        data = self.pad(data)
        data = self.partition(data)
        return self.coordinates, data


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
        na, no, c, wa, wo = data.shape
        data = torch.permute(data, (2, 0, 3, 1, 4))                         # (c, na, wa, no, wo)
        data = torch.reshape(data, (c, na*wa, no*wo))                       # (c, lat, lon)
        return data


    def merge_batch(self, data:torch.Tensor) -> torch.Tensor:
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
  

    def downsample(self, data_hr:torch.Tensor, scale:int) -> torch.Tensor:
        assert data_hr.ndim == 6
        assert scale in [2,3,4]
        b, na, no, c, wa, wo = data_hr.shape
        data_hr = self.merge_batch(data_hr)
        b, c, lat, lon = data_hr.shape

        data_lr = torch.zeros_like(data_hr)
        data_lr[:,:,::4] = data_hr[:,:,::4]
        data_lr = self.partition_batch(data_lr)
        return data_lr.detach()


    def get_wandb(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> tuple[wandb.Object3D, wandb.Object3D]:
        data_pred = data_pred[0,-1]                                         # first in batch, last step
        data_true = data_true[0]                                            # first in batch
              
        data_pred = torch.clamp(data_pred, 0, 1)

        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)

        pc_pred = self.grid_to_pointcloud(data_pred[0])
        pc_true = self.grid_to_pointcloud(data_true[0])


        return wandb.Object3D(pc_pred.numpy(force=True)), wandb.Object3D(pc_true.numpy(force=True))
    

    def get_metric_train(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge_batch(data_pred)
        data_true = self.merge_batch(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)

        mae = self.mean_abs_error(data_pred, data_true)
        psnr = self.psnr(data_pred, data_true)
        return {
            'mae': mae,
            'psnr': psnr
        }


    def get_metric_valid(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        with torch.no_grad():
            b = data_true.shape[0]
            data_pred = data_pred[:,-1]                                         # last step
            data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
            data_pred = self.merge_batch(data_pred)
            data_true = self.merge_batch(data_true)
            data_pred = self.cut(data_pred)
            data_true = self.cut(data_true)

            mae = self.mean_abs_error(data_pred, data_true)
            psnr = self.psnr(data_pred, data_true)
            iou = 0
            cd = 0
            for i in range(b):
                pc_true = self.grid_to_pointcloud(data_true[i,0])
                pc_pred = self.grid_to_pointcloud(data_pred[i,0])
                iou += self.iou(pc_true, pc_pred)
                # cd += self.chamfer_distance(pc_true, pc_pred)
                del pc_true, pc_pred
            iou /= b
            cd /= b
            return {
                'mae': mae,
                'psnr': psnr,
                'iou': iou,
                'cd': cd
            }
        
    def get_metric_test(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        return self.get_metric_valid(data_pred, data_true)

    def grid_to_pointcloud(self, data:torch.Tensor) -> torch.Tensor:
        elevation_bins = data.shape[0]
        elevation_id = torch.arange(elevation_bins, device=data.device)
        elevation_resolution = 0
        if elevation_bins > 1:
            elevation_resolution = (self.elevation_max - self.elevation_min) / (elevation_bins - 1)

        azimuth_bins = data.shape[1]
        azimuth_id = torch.arange(azimuth_bins, device=data.device)
        azimuth_resolution = 0
        if azimuth_bins > 1:
            azimuth_resolution = (self.azimuth_max - self.azimuth_min) / (azimuth_bins - 1)
        
        elevation_grid, azimuth_grid = torch.meshgrid(elevation_id, azimuth_id, indexing='ij')
        elevation_grid = elevation_grid.flatten()
        azimuth_grid = azimuth_grid.flatten()

        phi = self.elevation_min + elevation_grid * elevation_resolution
        phi = torch.deg2rad(phi)
        theta = self.azimuth_min + azimuth_grid * azimuth_resolution
        theta = torch.deg2rad(theta)
        r = data.flatten()
        i = torch.floor(r * 13).to(torch.int32) + 1
        r = r * self.laser_max_range

        filter = (r<=self.laser_max_range) & (r>=self.laser_min_range)

        phi = phi[filter]
        theta = theta[filter]
        i = i[filter]
        r = r[filter]
        x = r * torch.cos(phi) * torch.cos(theta)
        y = r * torch.cos(phi) * torch.sin(theta)
        z = r * torch.sin(phi)

        result = torch.stack((x, y, z, i), axis=-1)
        return result
    
    def iou(self, true:torch.Tensor, pred:torch.Tensor, grid_size:float=0.1) -> float:
        assert true.dim() == 2
        assert pred.dim() == 2
        assert true.shape[-1] == 4
        assert pred.shape[-1] == 4
        true_id = (true[:,:3] / grid_size).to(torch.int)
        pred_id = (pred[:,:3] / grid_size).to(torch.int)

        true_unique = torch.unique(true_id, dim=0)
        pred_unique = torch.unique(pred_id, dim=0)

        union = len(true_unique) + len(pred_unique)

        true_unique = torch.unsqueeze(true_unique, 0)
        pred_unique = torch.unsqueeze(pred_unique, 1)

        matches = torch.all(pred_unique == true_unique, dim=2)
        mask = torch.any(matches, dim=1)

        intersection = len(pred_unique[mask])
        union -= intersection

        if union == 0:
            return 0.0
        else:
            iou = intersection / union
            return iou
    

    def mean_abs_error(self, true:torch.Tensor, pred:torch.Tensor):
        error = torch.abs(pred - true)
        error = torch.mean(error)
        error *= self.laser_max_range
        return error.item()

    def psnr(self, pred:torch.Tensor, true:torch.Tensor) -> torch.Tensor:
        assert true.dim() == 4 # batch of lidar images with shape (b, c, lat, lon)
        assert pred.shape == true.shape
        assert true.dtype == torch.float32
        assert pred.dtype == true.dtype
        assert true.max() <= 1.
        assert pred.max() <= 1.

        mse = torch.square(true - pred)
        mse = torch.mean(mse)
        if mse == 0:
            return 100.0
        else:
            psnr = -10*torch.log10(mse)
            return psnr.item()

    def chamfer_distance(self, true:torch.Tensor, pred:torch.Tensor):
        assert true.dim() == 2
        assert pred.dim() == 2
        assert true.shape[-1] == 4
        assert pred.shape[-1] == 4


        with torch.no_grad():
            true = torch.unsqueeze(true[:,:3], axis=0)  # (1, n, 3)
            pred = torch.unsqueeze(pred[:,:3], axis=1)  # (m, 1, 3)
            dist = torch.sum((true - pred)**2, axis=2)  # (m, n)
            min_dist_true_pred, _ = torch.min(dist, axis=0) # (n,)
            min_dist_pred_true, _ = torch.min(dist, axis=1) # (m,)
                
            cd = torch.mean(min_dist_true_pred) + torch.mean(min_dist_pred_true)
            del dist, min_dist_pred_true, min_dist_true_pred
            return cd.item()        