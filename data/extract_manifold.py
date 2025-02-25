import os
import xarray
import torch
from torch.utils.data import random_split

device = torch.device('cuda:0')
data_dir = '/your/path/to/the/data/folder/'


data_raw = xarray.open_dataset(os.path.join(data_dir,'raw','era5.grib'), engine='cfgrib')
print(data_raw)

data = torch.tensor(data_raw['t2m'].values)
print(data.shape, data.dtype)

# data statistics
any_nan = torch.isnan(data).any()
any_inf = torch.isinf(data).any()
min_temp = torch.min(data).item()
max_temp = torch.max(data).item()
mean_temp = torch.mean(data).item()
std_temp = torch.std(data).item()

print("Any NaN values:", any_nan.item())
print("Any Inf values:", any_inf.item())
print(f"Minimum Temperature: {min_temp}K")
print(f"Maximum Temperature: {max_temp}K")
print(f"Mean Temperature: {mean_temp}K")
print(f"Standard Deviation: {std_temp}K")

# interpolate
data_interpolated = torch.nn.functional.interpolate(data.view(14400, 1, 721, 1440), size=(180,360), mode='bilinear')

# min-max normalization
data_interpolated = (data_interpolated - data_interpolated.min()) / (data_interpolated.max() - data_interpolated.min())

print(data_interpolated.min(), data_interpolated.max(), data_interpolated.mean())

torch.manual_seed(42)
data_interpolated = data_interpolated[torch.randperm(len(data_interpolated))]
train, valid, test = torch.split(data_interpolated, [10000, 2200, 2200])
train = train.detach().clone()
valid = valid.detach().clone()
test = test.detach().clone()
torch.save(train, os.path.join(data_dir,'tensor','era5_train.pt'))
torch.save(valid, os.path.join(data_dir,'tensor','era5_valid.pt'))
torch.save(test, os.path.join(data_dir,'tensor','era5_test.pt'))
