import os
import torch
import wandb
from metrics import video_psnr

class ADOBE240(torch.utils.data.Dataset):
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
        self.dim_data = [25,256,256]
        self.dim_data_pad = self.calculate_pad()
        self.coordinates = self.get_coordinates()

        if split == 'train':
            self.dir = os.path.join(data_dir,'datasets','adobe240','tensors','train')
            self.paths = [os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir)) if f.endswith(('pt'))]
            print(f'Adobe240 train loaded: {len(self.paths)}')
        elif split == 'valid':
            self.dir = os.path.join(data_dir,'datasets','adobe240','tensors','valid')
            self.paths = [os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir)) if f.endswith(('pt'))]
            print(f'Adobe240 valid loaded: {len(self.paths)}')
        else:
            self.dir = os.path.join(data_dir,'datasets','adobe240','tensors','test')
            self.paths = [os.path.join(self.dir, f) for f in sorted(os.listdir(self.dir)) if f.endswith(('pt'))]
            print(f'Adobe240 test loaded: {len(self.paths)}')

        self.cache = {}


    def __len__(self) -> int:
        return len(self.paths)


    def __getitem__(self, i) -> torch.Tensor:
        if i in self.cache:
            data = self.cache[i]
        else:
            data = torch.load(self.paths[i], weights_only=True)
            self.cache[i] = data

        if self.sliced:
            data = self.random_crop(data)
        else:
            data = self.random_crop(data, [25,256,256])
            data = self.pad(data)            
        data = self.partition(data)
        return self.coordinates, data / 255.
    

    def get_sample(self) -> torch.Tensor:
        id = torch.randint(0, len(self), (1,)).item()
        if id in self.cache:
            data = self.cache[id]
        else:
            data = torch.load(self.paths[id], weights_only=True)
            self.cache[id] = data

        data = self.random_crop(data, [25,256,256])
        data = self.pad(data)
        data = self.partition(data)
        return self.coordinates, data / 255.

    def get_coordinates(self) -> torch.Tensor:
        t, x, y = self.dim_slice
        st = torch.linspace(0. + 0.5/t, 1. - 0.5/t, steps=t)
        sx = torch.linspace(0. + 0.5/x, 1. - 0.5/x, steps=x)
        sy = torch.linspace(0. + 0.5/y, 1. - 0.5/y, steps=y)
        coordinates = torch.meshgrid([st, sx, sy], indexing='ij')
        coordinates = torch.stack(coordinates, dim=-1)
        coordinates = torch.flatten(coordinates, end_dim=-2)
        coordinates = coordinates.detach().clone()
        return coordinates.requires_grad_(True)   


    def get_metric_train(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        psnr = self.psnr(data_true, data_pred)
        return {
            'psnr': psnr
        }
    

    def get_metric_valid(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        psnr = self.psnr(data_true, data_pred)
        return {
            'psnr': psnr
        }


    def get_metric_test(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> float:
        data_pred = data_pred[:,-1]                                         # last step
        data_pred = torch.clamp(data_pred, 0, 1)                            # clip model output
        data_pred = self.merge(data_pred)
        data_true = self.merge(data_true)
        data_pred = self.cut(data_pred)
        data_true = self.cut(data_true)
        psnr = self.psnr(data_true, data_pred)
        ssim = self.ssim(data_true, data_pred)
        return {
            'psnr': psnr,
            'ssim': ssim,
        }


    def get_wandb(self, data_pred:torch.Tensor, data_true:torch.Tensor) -> tuple[wandb.Video, wandb.Video]:
        data_pred = data_pred[0]                                            # first in batch
        data_true = data_true[0]                                            # first in batch
        data_pred = torch.clamp(data_pred, 0, 1)
        data_true = torch.unsqueeze(data_true, 0)                           # (1, nt, nx, ny, c, wt, wx, wy)
        data = torch.cat((data_pred, data_true))                            # (s+2, nt, nx, ny, c, wt, wx, wy)
        data = self.merge(data)                                             # (s+2, c, t, x, y)
        data = self.cut(data)                                               # (s+2, c, t, x, y)
        data = data * 255
        data = data.to(torch.uint8)
        data = data.detach().cpu()
        return wandb.Video(torch.permute(data[-2],(1,0,2,3)), fps=1, format="mp4"), wandb.Video(torch.permute(data[-1],(1,0,2,3)), fps=1, format="mp4")
    

    def partition(self, data:torch.Tensor) -> torch.Tensor:
        c, t, x, y = data.shape
        wt, wx, wy = self.dim_slice
        nt, nx, ny = t//wt, x//wx, y//wy
        data = torch.reshape(data, (c, nt, wt, nx, wx, ny, wy))             # (c, nt, wt, nx, wx, ny, wy)
        data = torch.permute(data, (1, 3, 5, 0, 2, 4, 6))                   # (nt, nx, ny, c, wt, wx, wy)
        return data.detach()


    def partition_batch(self, data:torch.Tensor) -> torch.Tensor:
        b, c, t, x, y = data.shape
        wt, wx, wy = self.dim_slice
        nt, nx, ny = t//wt, x//wx, y//wy
        data = torch.reshape(data, (b, c, nt, wt, nx, wx, ny, wy))          # (b, c, nt, wt, nx, wx, ny, wy)
        data = torch.permute(data, (0, 2, 4, 6, 1, 3, 5, 7))                # (b, nt, nx, ny, c, wt, wx, wy)
        return data.detach()


    def merge(self, data:torch.Tensor) -> torch.Tensor:
        b, nt, nx, ny, c, wt, wx, wy = data.shape
        data = torch.permute(data, (0, 4, 1, 5, 2, 6, 3, 7))                # (b, c, nt, wt, nx, wx, ny, wy)
        data = torch.reshape(data, (b, c, nt*wt, nx*wx, ny*wy))             # (b, c, t, x, y)
        return data.detach()


    def cut(self, data:torch.Tensor) -> torch.Tensor:
        t, x, y = self.dim_data
        assert data.ndim == 5
        return data[:,:,:t,:x,:y].detach()


    def pad(self, data:torch.Tensor) -> torch.Tensor:
        assert data.ndim == 4
        c, t, x, y = data.shape
        st, sx, sy = self.dim_slice
        wt, wx, wy = self.dim_window
        pt, px, py = st*wt, sx*wx, sy*wy
        pt = (pt - (t%pt))%pt
        px = (px - (x%px))%px
        py = (py - (y%py))%py
        result = torch.zeros((c, t+pt, x+px, y+py)).to(data)
        result[:,:t,:x,:y] = data
        return result.detach().clone()


    def calculate_pad(self) -> tuple[int, int ,int]:
        st, sx, sy = self.dim_slice
        wt, wx, wy = self.dim_window
        pt, px, py = st*wt, sx*wx, sy*wy
        t, x, y = self.dim_data
        return [(((t+pt-1)//pt)*pt)//st,(((x+px-1)//px)*px)//sx, (((y+py-1)//py)*py)//sy]


    def random_crop(self, video:torch.Tensor, cut:list[int, int, int]=None) -> torch.Tensor:
        assert video.ndim == 4
        c, t, h, w = video.shape
        if cut is None:
            ct, ch, cw = self.dim_slice
        else:
            ct, ch, cw = cut

        st = torch.randint(0, max(0, t - ct), (1,))
        sh = torch.randint(0, max(0, h - ch), (1,))
        sw = torch.randint(0, max(0, w - cw), (1,))
        video = video[:, st:st+ct, sh:sh+ch, sw:sw+cw]
        return video.detach().clone()
    
    
    def downsample(self, data_hr:torch.Tensor, scale:int) -> torch.Tensor:
        assert data_hr.ndim == 8
        assert scale in [2,3,4]
        b, nt, nx, ny, c, wt, wx, wy = data_hr.shape
        data_hr = self.merge(data_hr)
        b, c, t, x, y = data_hr.shape

        # frame 7-skip
        data_lr = torch.zeros_like(data_hr)
        data_lr[:,:,0] = data_hr[:,:,0]
        data_lr[:,:,8] = data_hr[:,:,8]
        data_lr[:,:,16] = data_hr[:,:,16]
        data_lr[:,:,24] = data_hr[:,:,24]

        data_lr = torch.permute(data_lr, (0,2,1,3,4))       # (b, t, c, x, y)
        data_lr = torch.reshape(data_lr, (b*t, c, x, y))    # (b*t, c, x, y)

        data_lr = torch.nn.functional.interpolate(data_lr, size=[x//4, y//4], mode='bilinear')
        data_lr = torch.nn.functional.interpolate(data_lr, size=[x, y], mode='bilinear')

        data_lr = torch.reshape(data_lr, (b, t, c, x, y))   # (b, t, c, x, y)
        data_lr = torch.permute(data_lr, (0,2,1,3,4))       # (b, c, t, x, y)

        data_lr = self.partition_batch(data_lr)
        return data_lr.detach().clone()
    

    def psnr(self,
        video_true:torch.Tensor,
        video_test:torch.Tensor,
    ) -> torch.Tensor:
        
        assert video_true.dim() == 5 # batch of videos with shape (b, c, t, h, w)
        assert video_true.shape[1] == 3 #rgb video
        assert video_test.shape == video_true.shape
        assert video_true.dtype == torch.float32
        assert video_test.dtype == video_true.dtype
        assert video_true.max() <= 1.0
        assert video_test.max() <= 1.0

        video_true = torch.round(video_true*255)
        video_test = torch.round(video_test*255)

        mse = torch.mean(torch.square(video_true - video_test))
        if mse == 0:
            return 100.0 # this corresponds to a epsilon of 1e-10
        psnr = 20*torch.log10(torch.tensor(255.)) - 10*torch.log10(mse)
        return psnr.item()
    

    def ssim(self,
        video_true:torch.Tensor,
        video_pred:torch.Tensor,
    ) -> torch.Tensor:
        assert video_true.dim() == 5 # batch of videos with shape (b, c, t, h, w)
        assert video_true.shape[1] == 3 #rgb video
        assert video_pred.shape == video_true.shape
        assert video_true.dtype == torch.float32
        assert video_pred.dtype == video_true.dtype
        assert video_true.max() <= 1.0
        assert video_pred.max() <= 1.0    

        b = video_true.shape[0]
        c = video_true.shape[1]
        t = video_true.shape[2]
        h = video_true.shape[3]
        w = video_true.shape[4]

        video_true = torch.permute(video_true, (0,2,1,3,4))
        video_pred = torch.permute(video_pred, (0,2,1,3,4))
        video_true = torch.reshape(video_true, (b*t, c, h, w))
        video_pred = torch.reshape(video_pred, (b*t, c, h, w))

        video_true = torch.round(video_true*255)
        video_pred = torch.round(video_pred*255)

        c1 = (0.01 * 255)**2
        c2 = (0.03 * 255)**2

        def gaussian_kernel(window_size, sigma):
            x = torch.arange(window_size)
            gauss = torch.exp(-((x - window_size // 2) ** 2) / (2 * sigma ** 2))
            return gauss / gauss.sum()

        kernel_size = 11
        sigma = 1.5
        kernel = gaussian_kernel(kernel_size, sigma).unsqueeze(1)
        window = kernel.mm(kernel.t()).unsqueeze(0).unsqueeze(0)
        window = window.expand(video_true.size(1), 1, kernel_size, kernel_size).to(video_true.dtype).to(video_true.device)

        mu1 = torch.nn.functional.conv2d(video_true, window, stride=1, padding=0, groups=video_true.shape[1])
        mu2 = torch.nn.functional.conv2d(video_pred, window, stride=1, padding=0, groups=video_pred.shape[1])
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = torch.nn.functional.conv2d(video_true * video_true, window, stride=1, padding=0, groups=video_true.shape[1]) - mu1_sq
        sigma2_sq = torch.nn.functional.conv2d(video_pred * video_pred, window, stride=1, padding=0, groups=video_pred.shape[1]) - mu2_sq
        sigma12 = torch.nn.functional.conv2d(video_true * video_pred, window, stride=1, padding=0, groups=video_true.shape[1]) - mu1_mu2

        cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
        ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
        ssim_map = ssim_map.mean()
        return ssim_map.item()
