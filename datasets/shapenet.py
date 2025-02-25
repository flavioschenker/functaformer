import os
import torch
import wandb
import matplotlib.pyplot as plt
from metrics import voxel_accuracy

class SHAPENET(torch.utils.data.Dataset):
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
        self.dim_data = [64, 64, 64]
        self.dim_data_pad = self.calculate_pad()
        self.coordinates = self.get_coordinates()

        if split == 'train':
            self.path = os.path.join(data_dir,'datasets','shapenet','tensor','shapenet_train.pt')
            self.data = torch.load(self.path, weights_only=True)
            print(f'SHAPENET train loaded: {self.data.shape}')
        elif split == 'valid':
            self.path = os.path.join(data_dir,'datasets','shapenet','tensor','shapenet_valid.pt')
            self.data = torch.load(self.path, weights_only=True)
            self.data = self.data
            print(f'SHAPENET valid loaded: {self.data.shape}')
        else:
            self.path = os.path.join(data_dir,'datasets','shapenet','tensor','shapenet_test.pt')
            self.data = torch.load(self.path, weights_only=True)
            self.data = self.data
            print(f'SHAPENET test loaded: {self.data.shape}')


    def __len__(self) -> int:
        return len(self.data)


    def __getitem__(self, i:int) -> torch.Tensor:
        data = self.data[i]                                                 # (c, x, y, z)
        data = data.to(torch.float32)
        if self.sliced:
            data = self.random_crop(data)                                   # (c, wx, wy, wz)
        else:
            data = self.pad(data)                                           # (c, x, y, z)
        data = self.partition(data)                                         # (nx, ny, nz, c, wx, wy, wz)
        return self.coordinates, data
    

    def get_coordinates(self) -> torch.Tensor:
        x, y, z = self.dim_slice
        sx = torch.linspace(0. + 0.5/x, 1. - 0.5/x, steps=x)
        sy = torch.linspace(0. + 0.5/y, 1. - 0.5/y, steps=y)
        sz = torch.linspace(0. + 0.5/z, 1. - 0.5/z, steps=z)
        coordinates = torch.meshgrid([sx, sy, sz], indexing='ij')
        coordinates = torch.stack(coordinates, dim=-1)
        coordinates = torch.flatten(coordinates, end_dim=-2)
        coordinates = coordinates.detach().clone()
        return coordinates.requires_grad_(True)       


    def get_sample(self) -> torch.Tensor:
        id = torch.randint(0, len(self.data), (1,)).item()
        data = self.data[id]
        data = data.to(torch.float32)
        data = self.pad(data)
        data = self.partition(data)
        return self.coordinates, data


    def get_metric_train(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        accuracy = voxel_accuracy(data_true, data_pred)
        return {
            'acc': accuracy,
        }


    def get_metric_valid(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        accuracy = voxel_accuracy(data_true, data_pred)
        iou = self.iou(data_true, data_pred)
        precision, recall, f1 = self.f1(data_true, data_pred)
        return {
            'acc': accuracy,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


    def get_metric_test(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        return self.get_metric_valid(data_pred, data_true)

    def get_wandb(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> tuple[wandb.Image, wandb.Image]:
        data_pred = data_pred[0]                                            # first in batch
        data_true = data_true[0]                                            # first in batch
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_true = torch.unsqueeze(data_true, 0)                           # (1, nx, ny, nz, c, wx, wy, wz)
        data = torch.cat((data_pred, data_true))                            # (s+2, nx, ny, nz, c, wx, wy, wz)
        data = self.merge(data)                                             # (s+2, c, x, y, z)
        data = self.cut(data)                                               # (s+2, c, x, y, z)
        wandb_true = self.render_model(data[-1,0])
        wandb_pred = self.render_model(data[-2,0])
        return wandb.Image(wandb_pred), wandb.Image(wandb_true)


    def partition(self, data:torch.Tensor) -> torch.Tensor:
        c, x, y, z = data.shape
        wx, wy, wz = self.dim_slice
        nx, ny, nz = x//wx, y//wy, z//wz
        data = torch.reshape(data, (c, nx, wx, ny, wy, nz, wz))             # (c, nx, wx, ny, wy, nz, wz)
        data = torch.permute(data, (1, 3, 5, 0, 2, 4, 6))                   # (nx, ny, nz, c, wx, wy, wz)
        return data.detach()
    

    def partition_batch(self, data:torch.Tensor) -> torch.Tensor:
        b, c, x, y, z = data.shape
        wx, wy, wz = self.dim_slice
        nx, ny, nz = x//wx, y//wy, z//wz
        data = torch.reshape(data, (b, c, nx, wx, ny, wy, nz, wz))          # (b, c, nx, wx, ny, wy, nz, wz)
        data = torch.permute(data, (0, 2, 4, 6, 1, 3, 5, 7))                # (b, nx, ny, nz, c, wx, wy, wz)
        return data.detach()
    

    def merge(self, data:torch.Tensor) -> torch.Tensor:
        b, nx, ny, nz, c, wx, wy, wz = data.shape
        data = torch.permute(data, (0, 4, 1, 5, 2, 6, 3, 7))                # (b, c, nx, wx, ny, wy, nz, wz)
        data = torch.reshape(data, (b, c, nx*wx, ny*wy, nz*wz))             # (b, c, x, y, z)
        return data.detach()


    def cut(self, data:torch.Tensor) -> torch.Tensor:
        x, y, z = self.dim_data
        assert data.ndim == 5
        return data[:,:,:x,:y,:z].detach()


    def pad(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 4
        c, x, y, z = data.shape
        sx, sy, sz = self.dim_slice
        wx, wy, wz = self.dim_window
        px, py, pz = sx*wx, sy*wy, sz*wz
        px = (px - (x%px))%px
        py = (py - (y%py))%py
        pz = (pz - (z%pz))%pz
        result = torch.zeros((c, x+px, y+py, z+pz), dtype=torch.float32).to(data)
        result[:,:x,:y,:z] = data
        return result.detach()


    def calculate_pad(self) -> tuple[int, int ,int]:
        sx, sy, sz = self.dim_slice
        wx, wy, wz = self.dim_window
        px, py, pz = sx*wx, sy*wy, sz*wz
        x, y, z = self.dim_data
        return [(((x+px-1)//px)*px)//sx, (((y+py-1)//py)*py)//sy, (((z+pz-1)//pz)*pz)//sz]


    def random_crop(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 4
        c, x, y, z = data.shape
        cx, cy, cz = self.dim_slice
        sx = torch.randint(0, max(0, x - cx), (1,))
        sy = torch.randint(0, max(0, y - cy), (1,))
        sz = torch.randint(0, max(0, z - cz), (1,))
        data = data[:, sx:sx+cx, sy:sy+cy, sz:sz+cz]
        return data.detach()

    def iou(self, true, pred) -> torch.Tensor:      
        intersection = (true & pred).to(int).sum()
        union = (true | pred).to(int).sum()
        
        if union == 0:
            return 1.0
        else:
            iou = intersection / union
        return iou.item()


    def f1(self, true, pred):
        tp = (true & pred).to(int).sum()
        
        pp = pred.to(int).sum()
        ap = true.to(int).sum()

        if (pp + ap + tp).item() == 0:
            return 1.0, 1.0, 1.0

        precision = (tp / pp).item() if pp > 0 else 0.0
        recall = (tp / ap).item() if ap > 0 else 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0


        return f1, precision, recall


    def downsample(self, data_hr:torch.Tensor, scale:int) -> torch.Tensor:
        assert data_hr.ndim == 8
        assert scale in [2,3,4]
        b, nx, ny, nz, c, wx, wy, wz = data_hr.shape
        data_hr = self.merge(data_hr)
        b, c, x, y, z = data_hr.shape

        if scale==2:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[x//2, y//2, z//2], mode='trilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y, z], mode='trilinear')
        elif scale==3:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[x//3, y//3, z//3], mode='trilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y, z], mode='trilinear')
        elif scale==4:
            data_lr = torch.nn.functional.interpolate(data_hr, size=[x//4, y//4, z//4], mode='trilinear')
            data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y, z], mode='trilinear')
            
        data_lr = (data_lr > 0.05)
        data_lr = data_lr.to(torch.float32)
        data_lr = self.partition_batch(data_lr)
        return data_lr.detach()


    def render_model(self,
        voxel:torch.Tensor
    ):
        voxel = voxel.permute(0, 2, 1)  # x, y, z -> x, z, y
        voxel = voxel >= 0.5
        voxel = voxel.numpy(force=True)

        fig = plt.figure(dpi=200)
        plt.tight_layout()
        ax = fig.add_subplot(111, projection='3d')
        ax.axis('off')
        ax.view_init(elev=30, azim=225)
        ax.voxels(voxel, facecolors='lightgray')
        img = wandb.Image(plt)
        plt.close()
        return img