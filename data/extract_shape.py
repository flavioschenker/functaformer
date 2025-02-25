import os
import numpy
import torch
import zipfile

def read_binvox_header(fp):
    line = fp.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')
    dims = list(map(int, fp.readline().strip().split(b' ')[1:]))
    translate = list(map(float, fp.readline().strip().split(b' ')[1:]))
    scale = list(map(float, fp.readline().strip().split(b' ')[1:]))[0]
    line = fp.readline()
    return dims


def binvox_to_tensor(fp):
    dims = read_binvox_header(fp)
    raw_data = numpy.frombuffer(fp.read(), dtype=numpy.uint8)
    values, counts = raw_data[::2], raw_data[1::2]
    data = numpy.repeat(values, counts).astype(float)
    data = data.reshape(dims)
    data = torch.from_numpy(data)
    data = data.unsqueeze(0).unsqueeze(0)
    data = torch.nn.functional.interpolate(data, size=(64,64,64), mode='trilinear')
    data = (data > 0.05)
    data = torch.transpose(data, 3, 4) # xzy -> xyz
    return data.squeeze()


def extract_shapenet(zip_folder, categories):
    models = []
    count = 0

    for zip_file in os.listdir(zip_folder):
        if zip_file.endswith('.zip') and zip_file in categories:
            zip_path = os.path.join(zip_folder, zip_file)
            with zipfile.ZipFile(zip_path, 'r') as zip:
                for file in zip.namelist():
                    if file.endswith('.solid.binvox'):
                        with zip.open(file) as fp:
                            model = binvox_to_tensor(fp)
                            print(f'{count:>5}', model.shape, model.dtype)
                            models.append(model)
                            count += 1

    dataset = torch.stack(models)
    return dataset


if __name__ == "__main__":
    categories = ['03001627.zip']#['04379243.zip','02958343.zip','03001627.zip','02691156.zip'] # table, car, chairs, airplane
    data_dir = '/your/path/to/the/data/folder/'
    zip_folder = os.path.join(data_dir,'raw')
    dataset = extract_shapenet(zip_folder, categories)
    dataset = dataset.unsqueeze(1)                                              # (b, c, x, y, z)
    print(dataset.shape)

    torch.manual_seed(42)
    dataset = dataset[torch.randperm(len(dataset))]
    train, valid, test = torch.split(dataset, [6575, 100, 100])
    train = train.detach().clone()
    valid = valid.detach().clone()
    test = test.detach().clone()
    print(train.shape, valid.shape, test.shape)
    torch.save(train, os.path.join(data_dir,'tensor','shapenet_train.pt'))
    torch.save(valid, os.path.join(data_dir,'tensor','shapenet_valid.pt'))
    torch.save(test, os.path.join(data_dir,'tensor','shapenet_test.pt'))