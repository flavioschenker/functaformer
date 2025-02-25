import os
import numpy
import torch
import wandb
from metrics import image_psnr
from PIL import Image

class SET14(torch.utils.data.Dataset):
    def __init__(self,
        data_dir:str,
        dim_slice:list[int],
        dim_window:list[int],
    ) -> None:
        super().__init__()      

        self.dim_slice = dim_slice
        self.dim_window = dim_window
        self.dir = os.path.join(data_dir,'datasets','SET14')
        self.paths = [os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir)) if f.endswith(('jpg', 'png', 'jpeg'))]
        self.cache = {}
        self.coordinates = self.get_coordinates()

        print(f'SET14 loaded: {len(self.paths)}')


    def __len__(self) -> int:
        return len(self.paths)


    def __getitem__(self, i) -> torch.Tensor:
        if i in self.cache:
            data = self.cache[i]
        else:
            data = Image.open(self.paths[i])
            self.cache[i] = data

        data = self.pil_to_tensor(data)
        data = self.pad(data)
        data = self.partition(data)
        return self.coordinates, data


    def get_coordinates(self) -> torch.Tensor:
        x, y = self.dim_slice
        sx = torch.linspace(0. + 0.5/x, 1. - 0.5/x, steps=x)
        sy = torch.linspace(0. + 0.5/y, 1. - 0.5/y, steps=y)
        coordinates = torch.meshgrid([sx, sy], indexing='ij')
        coordinates = torch.stack(coordinates, dim=-1)
        coordinates = torch.flatten(coordinates, end_dim=-2)
        coordinates = coordinates.detach().clone()
        return coordinates.requires_grad_(True)   


    def get_metric(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        psnr = image_psnr(data_true, data_pred, 0)
        return psnr


    def get_wandb(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> tuple[wandb.Image, wandb.Image]:
        data_pred = data_pred[0]                                            # first in batch
        data_true = data_true[0]                                            # first in batch
        data_pred = torch.clamp(data_pred, 0, 1)
        data_true = torch.unsqueeze(data_true, 0)                           # (1, nx, ny, c, wx, wy)
        data = torch.cat((data_pred, data_true))                            # (s+2, nx, ny, c, wx, wy)
        data = self.merge(data)                                             # (s+2, c, x, y)
        return wandb.Image(data[:-1]), wandb.Image(data[-1])


    def partition(self, data:torch.Tensor) -> torch.Tensor:
        c, x, y = data.shape
        wx, wy = self.dim_slice
        nx, ny = x//wx, y//wy
        data = torch.reshape(data, (c, nx, wx, ny, wy))                     # (c, nx, wx, ny, wy)
        data = torch.permute(data, (1, 3, 0, 2, 4))                         # (nx, ny, c, wx, wy)
        return data.detach()


    def partition_batch(self, data:torch.Tensor) -> torch.Tensor:
        b, c, x, y = data.shape
        wx, wy = self.dim_slice
        nx, ny = x//wx, y//wy
        data = torch.reshape(data, (b, c, nx, wx, ny, wy))                  # (b, c, nx, wx, ny, wy)
        data = torch.permute(data, (0, 2, 4, 1, 3, 5))                      # (b, nx, ny, c, wx, wy)
        return data.detach()


    def merge(self, data:torch.Tensor) -> torch.Tensor:
        b, nx, ny, c, wx, wy = data.shape
        data = torch.permute(data, (0, 3, 1, 4, 2, 5))                      # (b, c, nx, wx, ny, wy)
        data = torch.reshape(data, (b, c, nx*wx, ny*wy))                    # (b, c, x, y)
        return data.detach()


    def pad(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 3
        c, x, y = data.shape
        sx, sy = self.dim_slice
        wx, wy = self.dim_window
        px, py = sx*wx, sy*wy
        px = (px - (x%px))%px
        py = (py - (y%py))%py
        result = torch.zeros((c, x+px, y+py)).to(data)
        result[:,:x,:y] = data
        return result.detach()


    def pil_to_tensor(self, image:Image) -> torch.Tensor:
        image = numpy.array(image)
        image = torch.from_numpy(image)
        image = torch.permute(image, (2, 0, 1))
        image = image.to(torch.float32)
        image = image / 255.
        return image


    def downsample(self, data_hr:torch.Tensor, scale:int) -> torch.Tensor:
        assert data_hr.ndim == 6
        assert scale in [2,3,4]
        b, nx, ny, c, wx, wy = data_hr.shape
        data_hr = self.merge(data_hr)
        b, c, x, y = data_hr.shape

        if scale==2:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[x//2, y//2], mode='bilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y], mode='bilinear')
        elif scale==3:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[x//3, y//3], mode='bilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y], mode='bilinear')
        elif scale==4:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[x//4, y//4], mode='bilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y], mode='bilinear')
            
        data_lr = self.partition_batch(data_lr)
        return data_lr.detach()