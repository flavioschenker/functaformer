import os
import numpy
import torch
import wandb

# Data for the Velodyne HDL-64E sensor
elevation_bins = 64
elevation_range = 26.8  # Elevation range [-24.8, 2.0] (degrees)
elevation_min = -24.8
elevation_max = 2
azimuth_bins = 1024
azimuth_range = 360
azimuth_min = -180      # Azimuth range [-180, 180] (degrees)
azimuth_max = 180
laser_max_range = 80

dataset_size = 25000

def categorize_intensity(data:torch.Tensor) -> torch.Tensor:
    category_min = 1
    category_max = 14

    i = data[:,3]
    i = (i - i.min()) / (i.max() - i.min())
    i = (i*(category_max-category_min) + category_min).to(int)

    data[:,3] = i
    return data


def grid_to_velodyne(data:torch.Tensor) -> torch.Tensor:
    elevation_bins = data.shape[0]
    elevation_id = torch.arange(elevation_bins)
    elevation_resolution = (elevation_max - elevation_min) / (elevation_bins - 1)

    azimuth_bins = data.shape[1]
    azimuth_id = torch.arange(azimuth_bins)
    azimuth_resolution = (azimuth_max - azimuth_min) / (azimuth_bins - 1)
    
    elevation_grid, azimuth_grid = torch.meshgrid(elevation_id, azimuth_id, indexing='ij')
    elevation_grid = elevation_grid.flatten()
    azimuth_grid = azimuth_grid.flatten()

    phi = elevation_min + elevation_grid * elevation_resolution
    phi = torch.deg2rad(phi)
    theta = azimuth_min + azimuth_grid * azimuth_resolution
    theta = torch.deg2rad(theta)
    r = data[:,:,0].flatten()
    i = data[:,:,1].flatten()
    
    phi = phi[r<=1]
    theta = theta[r<=1]
    i = i[r<=1]
    r = r[r<=1]

    # denormalizing
    r = r * laser_max_range

    x = r * torch.cos(phi) * torch.cos(theta)
    y = r * torch.cos(phi) * torch.sin(theta)
    z = r * torch.sin(phi)

    result = torch.stack((x, y, z, i), axis=-1)
    return result


def velodyne_to_grid(data:torch.Tensor) -> torch.Tensor:
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    i = data[:,3]
    r = torch.sqrt(x**2 + y**2 + z**2)

    phi = torch.arctan2(z, torch.sqrt(x**2 + y**2)) * 180 / torch.pi
    theta = torch.arctan2(y, x) * 180 / torch.pi

    r = r / laser_max_range
    if torch.any(r > 1.01): raise ValueError
    r = torch.clip(r, 0, 1)

    result = torch.zeros((elevation_bins, azimuth_bins, 2))
    result[:,:,0] = 10
    result[:,:,1] = 14

    phi_norm = (phi - elevation_min) / (elevation_max - elevation_min)
    theta_norm = (theta - azimuth_min) / (azimuth_max - azimuth_min)

    elevation_id = torch.clip(phi_norm*(elevation_bins-1), 0, elevation_bins-1).to(int)
    azimuth_id = torch.clip(theta_norm*(azimuth_bins-1), 0, azimuth_bins-1).to(int)

    result[elevation_id, azimuth_id, 0] = r
    result[elevation_id, azimuth_id, 1] = i
    
    return result


def extract_file(path:str) -> None:
    data = numpy.fromfile(path, dtype=numpy.float32)
    data = torch.from_numpy(data)
    data = torch.reshape(data, (-1, 4))
    raw = categorize_intensity(data)
    data = velodyne_to_grid(raw)
    return raw, data


def scan_for_files(root_dir:str) -> None:
    files = []
    with os.scandir(root_dir) as entries:
        entries = sorted(entries, key=lambda e: e.name)
        for entry in entries:
            if entry.is_dir():               
                files.extend(scan_for_files(entry.path))
            elif entry.is_file() and entry.name.endswith('.bin'):
                files.append(entry.path)
    return files


data_dir = '/your/path/to/the/data/folder/'
output_dir = '/your/path/to/the/data/folder/'

wandb.init(
    project='lidar',
)

files = scan_for_files(data_dir)
print(len(files))
indices = numpy.linspace(0, len(files) - 1, dataset_size, dtype=int)
files = [files[i] for i in indices]
print(len(files))

dataset = []
for i, file in enumerate(files):
    print(f'{i:>4}/{len(files)}', end='\r')
    raw, data = extract_file(file)
    dataset.append(data)

    if i < 10:
        data_hr = grid_to_velodyne(data)
        data_lr = grid_to_velodyne(data[::4])
        wandb.log({
            'raw': wandb.Object3D(raw.numpy(force=True)),
            'hr': wandb.Object3D(data_hr.numpy(force=True)),
            'lr': wandb.Object3D(data_lr.numpy(force=True)),
        })

dataset = torch.stack(dataset)
dataset = dataset[:,:,:,0]
dataset = torch.unsqueeze(dataset, 1)
print(dataset.shape)
torch.save(dataset[:20000].clone(), os.path.join(output_dir, 'kitty360_train.pt'))
torch.save(dataset[20000:22500].clone(), os.path.join(output_dir, 'kitty360_valid.pt'))
torch.save(dataset[22500:].clone(), os.path.join(output_dir, 'kitty360_test.pt'))
