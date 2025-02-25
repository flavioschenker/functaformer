
# 1) Download the Adobe240 dataset from: https://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring/DeepVideoDeblurring_Dataset_Original_High_FPS_Videos.zip
# 2) Move the 133 videos in your "data/datasets/adobe240/raw/" folder
# 3) Set your data directory
data_dir = '/your/path/to/the/data/folder/'
# 4) Run this script with "python data/extract_video.py"

import cv2
import torch
import os
import wandb

def video_to_tensor(video_path):
    video = cv2.VideoCapture(video_path)
    
    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    tensor = torch.empty((frames, height, width, 3), dtype=torch.uint8)     # (f, h, w, c)

    print(f'{frames} frames, {height}x{width} resolution, {video_path}')

    c = 0
    while c<frames:
        print(f'{c:>4}',end='\r')
        state, frame = video.read()
        if not state:
            break
        tensor[c] = torch.from_numpy(frame)                                 # (h, w, c)
        c += 1

    video.release()

    tensor = torch.permute(tensor, (3,0,1,2))                               # (c, f, h, w)
    tensor = tensor[[2, 1, 0], ...]                                         # BGR to RGB
    return tensor

input_dir = os.path.join(data_dir,'datasets','adobe240','raw')
output_dir = os.path.join(data_dir,'datasets','adobe240','tensors')


files = os.listdir(input_dir)
print(len(files), 'videos found!')

wandb.init(
    project='video',
)

c = 0

for i, file in enumerate(files):
    video = video_to_tensor(os.path.join(input_dir,file))

    if i < 100:
        output_dir = os.path.join(data_dir,'datasets','adobe240','tensors','train')
    elif i >= 100 and i < 116:
        output_dir = os.path.join(data_dir,'datasets','adobe240','tensors','valid')
    else:
        output_dir = os.path.join(data_dir,'datasets','adobe240','tensors','test')


    if video.shape[2] == 1280:
        print(video.shape)
        print('video in portrait mode, change to landscape')
        video = video.permute(0, 1, 3, 2)

    chunks = video.shape[1] // 240
    print(f'{chunks} chunks')
    for j in range(chunks):
        chunk = video[:,j*240:j*240+240].clone()
        torch.save(chunk, os.path.join(output_dir,f'{str(c).zfill(3)}_{str(i).zfill(3)}.pt'))
        print(chunk.max(), chunk.min(), chunk.dtype, chunk.shape)
        if c < 1:
            chunk = torch.permute(chunk,(1,0,2,3)).numpy()
            wandb.log({"videos": wandb.Video(chunk, fps=24, format="mp4")})
        c += 1