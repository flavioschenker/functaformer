import torch

def image_ssim(    
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
        ssi = ssim(image_test, image_true)
        return ssi

def video_ssim(
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
    ssi = ssim(video_test, video_true)
    return ssi


def ssim(
    img1:torch.Tensor,
    img2:torch.Tensor,
) -> float:

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
    window = window.expand(img1.size(1), 1, kernel_size, kernel_size).to(img1.dtype).to(img1.device)

    mu1 = torch.nn.functional.conv2d(img1, window, stride=1, padding=0, groups=img1.shape[1])
    mu2 = torch.nn.functional.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, stride=1, padding=0, groups=img1.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    ssim_map = ssim_map.mean()
    return ssim_map.item()