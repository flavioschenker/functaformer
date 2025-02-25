import os
import psutil
import argparse
import datetime
import random
import torch
from torch.utils.data import DataLoader
from models import SuperSiren


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-j',
        '--job_id',
        type=str,
        default='local'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/your/path/to/the/data/folder/'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        type=int,
        default=10000
    )
    parser.add_argument(
        '-w',
        '--wandbosity',
        action='store_true'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '-s',
        '--steps',
        type=int,
        default=100000
    )
    parser.add_argument(
        '-l',
        '--lr',
        type=float,
        default=3e-6
    )
    parser.add_argument(
        '--scheduler_milestones',
        type=int,
        nargs='+',
        default=[20000, 40000, 60000, 80000]
    )
    parser.add_argument(
        '-b',
        '--dim_batch',
        type=int,
        default=8
    )
    parser.add_argument(
        '-q',
        '--dim_queue',
        type=int,
        default=100
    )
    parser.add_argument(
        '-t',
        '--task',
        type=str,
        default='image',
        choices=['image','manifold','shape','audio','video','lidar']
    )
    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=64
    )
    parser.add_argument(
        '--dim_layers',
        type=int,
        default=3
    )
    parser.add_argument(
        '--dim_functa',
        type=int,
        default=64
    )
    parser.add_argument(
        '--dim_window',
        type=int,
        nargs='+',
        default=[1,1]
    )
    parser.add_argument(
        '--dim_slice',
        type=int,
        nargs='+',
        default=[2,2]
    )
    parser.add_argument(
        '--inner_steps',
        type=int,
        default=3
    )
    parser.add_argument(
        '-o',
        '--omega',
        type=int,
        default=30
    )
    parser.add_argument(
        '-a',
        '--activation',
        type=str,
        default='siren',
        choices=['siren','spder','wire']
    )

    args = parser.parse_args()
    # set job_id
    if args.job_id == 'local':
        args.job_id = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # make everything deterministic
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # set environmental variables
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['TORCH_USE_CUDA_DSA'] = 'true'

    # init wandb run and set environmental variables
    if args.wandbosity:
        import wandb
        wandbrun = wandb.init(project=f'ma_functa', config=args, name=args.job_id, settings=wandb.Settings(_service_wait=300))
    else:
        wandbrun = None

    if args.task == 'image':
        print('fitting image data')
        from datasets import DF2K
        dim_input = 2
        dim_output = 3

        dataset_train = DF2K(
            data_dir=args.data_dir,
            train=True,
            windowed=True,
            dim_window=args.dim_window,
        )
        dataset_test = DF2K(
            data_dir=args.data_dir,
            train=False,
            windowed=False,
            dim_window=args.dim_window,
        )

    elif args.task == 'manifold':
        print('fitting manifold data')
        from datasets import ERA5
        dim_input = 2
        dim_output = 1

        dataset_train = ERA5(
            data_dir=args.data_dir,
            windowed=True,
            train=True,
            dim_window=args.dim_window,
        )
        dataset_test = ERA5(
            data_dir=args.data_dir,
            train=False,
            windowed=False,
            dim_window=args.dim_window,
        )

    elif args.task == 'shape':
        from datasets import SHAPENET
        print('fitting 3D shapes data')
        dim_input = 3
        dim_output = 1

        dataset_train = SHAPENET(
            data_dir=args.data_dir,
            split='train',
            sliced=True,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_valid = SHAPENET(
            data_dir=args.data_dir,
            split='valid',
            sliced=True,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_test = SHAPENET(
            data_dir=args.data_dir,
            split='test',
            sliced=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )

    elif args.task == 'audio':
        print('fitting audio data')
        from datasets import LIBS
        dim_input = 1
        dim_output = 1

        dataset_train = LIBS(
            data_dir=args.data_dir,
            train=True,
            windowed=True,
            dim_window=args.dim_window,
        )
        dataset_test = LIBS(
            data_dir=args.data_dir,
            train=False,
            windowed=False,
            dim_window=args.dim_window,
        )

    elif args.task == 'video':
        print('fitting video data')
        from datasets import ADOBE240
        dim_input = 3
        dim_output = 3

        dataset_train = ADOBE240(
            data_dir=args.data_dir,
            split='train',
            sliced=True,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_valid = ADOBE240(
            data_dir=args.data_dir,
            split='valid',
            sliced=True,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_test = ADOBE240(
            data_dir=args.data_dir,
            split='test',
            sliced=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )

    elif args.task == 'lidar':
        print('fitting lidar data')
        from datasets import KITTY360
        dim_input = 2
        dim_output = 1

        dataset_train = KITTY360(
            data_dir=args.data_dir,
            split='train',
            sliced=True,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_valid = KITTY360(
            data_dir=args.data_dir,
            split='valid',
            sliced=True,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_test = KITTY360(
            data_dir=args.data_dir,
            split='test',
            sliced=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )

    model = SuperSiren(
        device=device,
        dim_input=dim_input,
        dim_output=dim_output,
        dim_queue=args.dim_queue,
        dim_slice=args.dim_slice,
        dim_hidden=args.dim_hidden,
        dim_layers=args.dim_layers,
        dim_functa=args.dim_functa,
        inner_steps=args.inner_steps,
        omega=args.omega,
        activation=args.activation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=0.5)
    
    train_loader = DataLoader(dataset_train, batch_size=args.dim_batch, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(dataset_valid, batch_size=args.dim_batch, shuffle=True, pin_memory=True, drop_last=True)
    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    model.train()
    step = 0
    while True:
        for i, (train_cord, train_data) in enumerate(train_loader):
            if step > args.steps:
                break
            train_cord, train_data = train_cord.to(device), train_data.to(device)

            optimizer.zero_grad()
            train_pred, train_loss, train_functa = model(train_cord, train_data)

            train_loss.backward()
            optimizer.step()
            scheduler.step()

            train_dict = dataset_train.get_metric_train(train_pred, train_data)
            del train_cord, train_data, train_pred, train_loss, train_functa

            log = f'STEP {step+1:>5}, '
            if step%args.verbosity==0:
                torch.save(model.state_dict(), os.path.join(args.data_dir,'models',f'functa_{args.task}_{args.job_id}.pt'))
                valid_metric = []
                for j, (valid_cord, valid_data) in enumerate(valid_loader):
                    print(f'validating {j+1:>5}/{len(valid_loader)}', end='\r')
                    valid_cord, valid_data = valid_cord.to(device), valid_data.to(device)
                    valid_pred, valid_functa = model.data_to_functa(valid_cord, valid_data)
                    valid_metric.append(dataset_valid.get_metric_valid(valid_pred, valid_data))
                    del valid_cord, valid_data, valid_pred, valid_functa

                valid_dict = {}
                for dict in valid_metric:
                    for key, value in dict.items():
                        valid_dict[key] = valid_dict.setdefault(key, 0) + value
                for key, value in valid_dict.items():
                    valid_dict[key] /= len(valid_metric)

                log += f'TRAIN ' + ', '.join(f'{key}: {value:.3f}' for key, value in train_dict.items())
                log += f' VALID ' + ', '.join(f'{key}: {value:.3f}' for key, value in valid_dict.items())

                if args.wandbosity and step > 0:
                    test_cord, test_data = dataset_test.get_sample()
                    test_cord, test_data = test_cord.to(device), test_data.to(device)
                    test_cord = test_cord.unsqueeze(0)
                    test_data = test_data.unsqueeze(0)
                    test_pred, test_functa = model.data_to_functa(test_cord, test_data)
                    wandb_pred, wandb_true = dataset_test.get_wandb(test_pred, test_data)
                    del test_cord, test_data, test_pred, test_functa
                    wandbrun.log({
                        'train.pred': wandb_pred,
                        'train.true': wandb_true,
                        'train': train_dict,
                        'valid': valid_dict,
                    })
            
            else:
                log = f'STEP {step+1:>5}, '
                log += f'TRAIN ' + ', '.join(f'{key}: {value:.3f}' for key, value in train_dict.items())
                if args.wandbosity:
                    wandbrun.log({
                        'train': train_dict,
                    })

            print(log)
            step += 1
        if step > args.steps:
            break

    print('done training!')

    model.eval()
    print('begin testing...')
    test_metrics = []
    for j, (test_cord, test_data) in enumerate(test_loader):
        log = f'STEP {j+1:>5}/{len(test_loader)}, '
        test_cord, test_data = test_cord.to(device), test_data.to(device)
        test_pred, test_functa = model.data_to_functa(test_cord, test_data)
        test_metric = dataset_test.get_metric_test(test_pred, test_data)
        test_metrics.append(test_metric)
        log += f', '.join(f'{key}: {value:.3f}' for key, value in test_metric.items())
        print(log)
        wandb_pred, wandb_true = dataset_test.get_wandb(test_pred, test_data)
        del test_cord, test_data, test_pred, test_functa
        if args.wandbosity:
            wandbrun.log({
                'test.pred': wandb_pred,
                'test.true': wandb_true,
                'test': test_metric,
            })

    test_dict = {}
    for dict in test_metrics:
        for key, value in dict.items():
            test_dict[key] = test_dict.setdefault(key, 0) + value
    for key, value in test_dict.items():
        test_dict[key] /= len(test_metrics)

    print('done testing!')
    print('final results:')
    log = f', '.join(f'{key}: {value:.3f}' for key, value in test_dict.items())
    print(log)
    if args.wandbosity:
        wandbrun.log({
            'final': test_dict,
        })
        wandb.finish()