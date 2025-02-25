import torch

def image_psnr(
    image_true:torch.Tensor,
    image_test:torch.Tensor,
    border:int
) -> torch.Tensor:
    
    assert image_true.dim() == 4 # batch of images with shape (b, c, h, w)
    assert image_true.shape[1] == 3 #rgb image
    assert image_test.shape == image_true.shape
    assert image_true.dtype == torch.float32
    assert image_test.dtype == image_true.dtype
    assert image_true.max() <= 1.0
    assert image_test.max() <= 1.0

    image_true = torch.round(image_true*255)
    image_test = torch.round(image_test*255)

    with torch.no_grad():
        if border > 0:
            image_true = image_true[:,:,border:-border,border:-border]
            image_test = image_test[:,:,border:-border,border:-border]

        mse = torch.mean(torch.square(image_true - image_test))
        if mse == 0:
            return 100.0 # this corresponds to a epsilon of 1e-10
        psnr = 20*torch.log10(torch.tensor(255.)) - 10*torch.log10(mse)
        return psnr.item()


def audio_psnr(
    audio_true:torch.Tensor,
    audio_test:torch.Tensor,
    border:int
) -> torch.Tensor:
    
    assert audio_true.dim() == 3 # batch of images with shape (b, c, t)
    assert audio_test.shape == audio_true.shape
    assert audio_true.dtype == torch.float32
    assert audio_test.dtype == audio_true.dtype
    assert audio_true.max() <= 1.0
    assert audio_test.max() <= 1.0

    with torch.no_grad():
        if border > 0:
            audio_true = audio_true[:,:,border:-border]
            audio_test = audio_test[:,:,border:-border]
        mse = torch.mean(torch.square(audio_true - audio_test))
        if mse == 0:
            return 100.0 # this corresponds to a epsilon of 1e-10
        psnr = -10*torch.log10(mse)
        return psnr.item()


def manifold_psnr(
    manifold_true:torch.Tensor,
    manifold_test:torch.Tensor,
    border:int
) -> torch.Tensor:
    
    assert manifold_true.dim() == 4 # batch of images with shape (b, c, lat, lon)
    assert manifold_test.shape == manifold_true.shape
    assert manifold_true.dtype == torch.float32
    assert manifold_test.dtype == manifold_true.dtype
    assert manifold_true.max() <= 1.
    assert manifold_test.max() <= 1.

    with torch.no_grad():
        if border > 0:
            manifold_true = manifold_true[:,:,border:-border,border:-border]
            manifold_test = manifold_test[:,:,border:-border,border:-border]
        mse = torch.mean(torch.square(manifold_true - manifold_test))
        if mse == 0:
            return 100.0 # this corresponds to a epsilon of 1e-10
        psnr = -10*torch.log10(mse)
        return psnr.item()
    

def video_psnr(
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