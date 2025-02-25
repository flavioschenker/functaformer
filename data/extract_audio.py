import os
import torch
import torchaudio
import soundfile

def compute_statistics(audio:torch.Tensor, sample_rate:int):
    duration = len(audio) / sample_rate
    amplitude_min = audio.min()
    amplitude_max = audio.max()
    amplitude_mean = audio.mean()
    amplitude_std = audio.std()
    signal_power = torch.mean(audio**2)
    noise_power = torch.mean((audio - torch.mean(audio))**2)
    snr = 10*torch.log10(signal_power/noise_power)
    spec = torch.stft(audio, n_fft=1024, window=torch.hann_window(1024, device='cpu'), return_complex=True)
    spec_mag = torch.abs(spec)
    spec_mean = spec_mag.mean()
    spec_std = spec_mag.std()
    silence_ratio = torch.sum(torch.abs(audio) < 0.01) / len(audio)

    return duration, amplitude_min.item(), amplitude_max.item(), amplitude_mean.item(), amplitude_std.item(), snr.item(), spec_mean.item(), spec_std.item(), silence_ratio.item()


def load_flac_files_to_tensors(folder_path):
    count = 0
    dataset_count = 0
    mean_duration         = 0
    min_duration          = 999
    max_duration          = 0
    amplitudes_min        = 999
    amplitudes_max        = 0
    mean_amplitude_mean   = 0
    mean_amplitude_std    = 0
    mean_snr              = 0
    mean_spec_mean        = 0
    mean_spec_std         = 0
    mean_silence_ratio    = 0

    tensors = []
    original_sample_rate = 16000
    target_sample_rate = 12000
    target_length = original_sample_rate * 5  # 5 seconds
    resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=target_sample_rate, dtype=torch.float64)

    for root, _, files in os.walk(folder_path):
        for filename in files:
            print(filename)
            # if dataset_count >= 120:
            #     break
            if filename.endswith(".flac"):
                file_path = os.path.join(root, filename)

                audio, sample_rate = soundfile.read(file_path)
                audio = torch.tensor(audio)

                duration, amplitude_min, amplitude_max, amplitude_mean, amplitude_std, snr, spec_mean, spec_std, silence_ratio = compute_statistics(audio, sample_rate)

                if duration < min_duration:
                    min_duration = duration

                if duration > max_duration:
                    max_duration = duration

                if amplitude_max > amplitudes_max:
                    amplitudes_max = amplitude_max
                if amplitude_min < amplitudes_min:
                    amplitudes_min = amplitude_min

                count                 += 1
                mean_duration         += duration
                mean_amplitude_mean   += amplitude_mean
                mean_amplitude_std    += amplitude_std
                mean_snr              += snr
                mean_spec_mean        += spec_mean
                mean_spec_std         += spec_std
                mean_silence_ratio    += silence_ratio

                if dataset_count < 22000 and len(audio) >= target_length:
                    print(count, dataset_count)               
                    audio = audio[:target_length]
                    audio = resampler(audio)
                    audio = audio.to(torch.float32)
                    audio = torch.clamp(audio, -1., 1.)         # clip interpolation errors
                    tensors.append(audio)
                    dataset_count += 1

    print('duration', mean_duration)

    mean_duration         /= count
    mean_amplitude_mean   /= count
    mean_amplitude_std    /= count
    mean_snr              /= count
    mean_spec_mean        /= count
    mean_spec_std         /= count
    mean_silence_ratio    /= count

    print('original')
    print('count', count)
    print('mean_duration',mean_duration)
    print('min_duration',min_duration)
    print('max_duration',max_duration)
    print('amplitudes_min',amplitudes_min)
    print('amplitudes_max',amplitudes_max)
    print('mean_amplitude_mean',mean_amplitude_mean)
    print('mean_amplitude_std',mean_amplitude_std)
    print('mean_snr',mean_snr)
    print('mean_spec_mean',mean_spec_mean)
    print('mean_spec_std',mean_spec_std)
    print('mean_silence_ratio',mean_silence_ratio)
            
    return torch.stack(tensors)

data_dir = '/your/path/to/the/data/folder/'
dataset = load_flac_files_to_tensors(os.path.join(data_dir,'raw'))
print(dataset.shape)

print('ours')
print('amplitudes_min',dataset.min().item())
print('amplitudes_max',dataset.max().item())
print('mean_amplitude_mean',torch.mean(dataset).item())
print('mean_amplitude_std',torch.mean(torch.std(dataset, dim=-1)).item())
signal_power = torch.mean(dataset**2, dim=-1)
noise_power = torch.mean((dataset - torch.mean(dataset, dim=-1).unsqueeze(-1))**2, dim=-1)
snr = torch.mean(10*torch.log10(signal_power/noise_power))
spec = torch.stft(dataset, n_fft=1024, window=torch.hann_window(1024, device='cpu'), return_complex=True)
spec_mag = torch.abs(spec)
spec_mean = torch.mean(spec_mag)
spec_std = torch.mean(torch.std(spec_mag, dim=-1))
silence_ratio = torch.mean(torch.sum(torch.abs(dataset) < 0.01, dim=-1) / 120000)
print('mean_snr',snr.item())
print('mean_spec_mean',spec_mean.item())
print('mean_spec_std',spec_std.item())
print('mean_silence_ratio',silence_ratio.item())

dataset = (dataset + 1) / 2                     # normalize to [0, 1]
torch.manual_seed(42)
dataset = dataset[torch.randperm(len(dataset))]
train, valid, test = torch.split(dataset, [20000, 1000, 1000])
train = train.detach().clone().unsqueeze(1)
valid = valid.detach().clone().unsqueeze(1)
test = test.detach().clone().unsqueeze(1)
print(train.shape, valid.shape, test.shape)
torch.save(train, os.path.join(data_dir,'tensor','libs_train.pt'))
torch.save(valid, os.path.join(data_dir,'tensor','libs_valid.pt'))
torch.save(test, os.path.join(data_dir,'tensor','libs_test.pt'))