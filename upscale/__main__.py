if __name__ == '__main__':
    import os
    import argparse
    import datetime
    import random
    import torch
    from models import SuperSiren, FunctaTransformer
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t',
        '--task',
        type=str,
        required=True,
        choices=['image','manifold','shape','audio','video','lidar']
    )
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
        '-w',
        '--wandbosity',
        action='store_true'
    )
    parser.add_argument(
        '-v',
        '--verbosity',
        type=int,
        default=5000
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42
    )
    parser.add_argument(
        '-m',
        '--model_id',
        type=str,
        required=True,
    )
    parser.add_argument(
        '-c',
        '--checkpoint',
        type=str,
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
        default=2e-4
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
        default=1
    )
    parser.add_argument(
        '-q',
        '--dim_queue',
        type=int,
        default=1000
    )
    parser.add_argument(
        '--dim_functa',
        type=int,
        default=64
    )
    parser.add_argument(
        '--dim_embedding',
        type=int,
        default=180
    )
    parser.add_argument(
        '--dim_layers',
        type=int,
        default=10
    )
    parser.add_argument(
        '--dim_blocks',
        type=int,
        nargs='+',
        default=[6,6]
    )
    parser.add_argument(
        '--dim_hidden',
        type=int,
        default=128
    )
    parser.add_argument(
        '--dim_ff',
        type=int,
        default=360
    )
    parser.add_argument(
        '--dim_window',
        type=int,
        nargs='+',
        default=[2,2]
    )
    parser.add_argument(
        '--dim_slice',
        type=int,
        nargs='+',
        default=[2,2]
    )
    parser.add_argument(
        '--num_head',
        type=int,
        default=6
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
        '--scale',
        type=int,
        default=4,
        choices=[2,3,4]
    )
    parser.add_argument(
        '-a',
        '--activation',
        type=str,
        default='siren',
        choices=['siren','spder','wire']
    )
    parser.add_argument(
        '--drop_conn',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--drop_attn',
        type=float,
        default=0.
    )
    parser.add_argument(
        '--drop_ffn',
        type=float,
        default=0.
    )

    args = parser.parse_args()
    # set job_id
    if args.job_id == 'local':
        args.job_id = f'{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # make everything deterministic
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # set pytorch cuda and environmental variables
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    os.environ['TORCH_USE_CUDA_DSA'] = 'true'

    # init wandb run and set environmental variables
    if args.wandbosity:
        import wandb
        os.environ['WANDB_DIR'] = os.path.join(args.data_dir)
        os.environ['WANDB_SILENT'] = 'true'
        os.environ['WANDB__SERVICE_WAIT'] = '300'
        wandbrun = wandb.init(project=f'ma_upscale', config=args, name=f'{args.job_id}_x{args.scale}', settings=wandb.Settings(_service_wait=300))
    else:
        wandbrun = None

    if args.task == 'image':
        print('fitting image data')
        from datasets import DF2K, SET5
        dim_input = 2
        dim_output = 3

        dataset_train = DF2K(
            data_dir=args.data_dir,
            train=True,
            windowed=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_test = SET5(
            data_dir=args.data_dir,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dim_data = dataset_train.dim_data_pad

    elif args.task == 'manifold':
        print('fitting manifold data')
        from datasets import ERA5
        dim_input = 2
        dim_output = 1

        dataset_train = ERA5(
            data_dir=args.data_dir,
            train=True,
            windowed=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_test = ERA5(
            data_dir=args.data_dir,
            train=False,
            windowed=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dim_data = dataset_train.dim_data_pad

    elif args.task == 'shape':
        from datasets import SHAPENET
        print('fitting 3D shapes data')
        dim_input = 3
        dim_output = 1

        dataset_train = SHAPENET(
            data_dir=args.data_dir,
            split='train',
            sliced=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_valid = SHAPENET(
            data_dir=args.data_dir,
            split='valid',
            sliced=False,
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
        dim_data = dataset_train.dim_data_pad

    elif args.task == 'audio':
        print('fitting audio data')
        from datasets import LIBS
        dim_input = 1
        dim_output = 1

        dataset_train = LIBS(
            data_dir=args.data_dir,
            train=True,
            windowed=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_test = LIBS(
            data_dir=args.data_dir,
            train=False,
            windowed=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dim_data = dataset_train.dim_data_pad

    elif args.task == 'video':
        print('fitting video data')
        from datasets import ADOBE240
        dim_input = 3
        dim_output = 3

        dataset_train = ADOBE240(
            data_dir=args.data_dir,
            split='train',
            sliced=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_valid = ADOBE240(
            data_dir=args.data_dir,
            split='valid',
            sliced=False,
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
        dim_data = dataset_train.dim_data_pad

    elif args.task == 'lidar':
        print('fitting lidar data')
        from datasets import KITTY360
        dim_input = 2
        dim_output = 1

        dataset_train = KITTY360(
            data_dir=args.data_dir,
            split='train',
            sliced=False,
            dim_slice=args.dim_slice,
            dim_window=args.dim_window,
        )
        dataset_valid = KITTY360(
            data_dir=args.data_dir,
            split='valid',
            sliced=False,
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
        dim_data = dataset_train.dim_data_pad

    model_transformer = FunctaTransformer(
        dim_data=dim_data,
        dim_window=args.dim_window,
        dim_functa=args.dim_functa,
        dim_embedding=args.dim_embedding,
        dim_blocks=args.dim_blocks,
        dim_hidden=args.dim_ff,
        num_head=args.num_head,
        drop_conn=args.drop_conn,
        drop_attn=args.drop_attn,
        drop_ffn=args.drop_ffn,
    ).to(device)

    if args.checkpoint:
        state_dict = torch.load(os.path.join(args.data_dir,'models',f'upscale_{args.task}_{args.checkpoint}.pt'), weights_only=True)
        model_transformer.load_state_dict(state_dict)
        print(f'checkpoint model {args.checkpoint} loaded')

    model_functa = SuperSiren(
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
    state_dict = torch.load(os.path.join(args.data_dir,'models',f'functa_{args.task}_{args.model_id}.pt'), weights_only=True)
    model_functa.load_state_dict(state_dict)

    loader_train = DataLoader(dataset_train, batch_size=args.dim_batch, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=args.dim_batch, shuffle=False)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model_transformer.parameters(), args.lr, betas=(0.9,0.99))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.scheduler_milestones, gamma=0.5)

    model_transformer.train()
    best_test_metric = 0
    step = 0

    while True:
        for train_cord, train_data_hr in loader_train:
            if step > args.steps:
                break

            train_cord, train_data_hr = train_cord.to(device), train_data_hr.to(device)
            train_data_lr = dataset_train.downsample(train_data_hr, args.scale)

            train_pred_hr, train_functa_hr = model_functa.data_to_functa(train_cord, train_data_hr)
            train_pred_lr, train_functa_lr = model_functa.data_to_functa(train_cord, train_data_lr)

            # layer norm across features
            train_functa_hr_mean = torch.mean(train_functa_hr, dim=1, keepdim=True)
            train_functa_hr_std = torch.std(train_functa_hr, dim=1, keepdim=True)
            train_functa_lr_mean = torch.mean(train_functa_lr, dim=1, keepdim=True)
            train_functa_lr_std = torch.std(train_functa_lr, dim=1, keepdim=True)
            train_functa_hr = (train_functa_hr - train_functa_hr_mean) / train_functa_hr_std
            train_functa_lr = (train_functa_lr - train_functa_lr_mean) / train_functa_lr_std

            train_functa_lr = train_functa_lr.detach().clone()
            train_functa_hr = train_functa_hr.detach().clone()
            torch.cuda.empty_cache()

            train_functa_pr = model_transformer(train_functa_lr)

            loss = loss_fn(train_functa_pr, train_functa_hr)
            loss.backward()
            del loss
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            log = f'STEP {step+1:>5}'
            if step%(max(args.verbosity//10,1))==0:
                torch.save(model_transformer.state_dict(), os.path.join(args.data_dir,'models',f'upscale_{args.task}_{args.job_id}.pt'))
                # denormalize
                train_functa_hr = train_functa_hr * train_functa_hr_std + train_functa_hr_mean
                train_functa_pr = train_functa_pr * train_functa_hr_std + train_functa_hr_mean
                train_functa_lr = train_functa_lr * train_functa_lr_std + train_functa_lr_mean

                train_pred_pr = model_functa.functa_to_data(train_cord, train_functa_pr)
                train_dict = dataset_train.get_metric_train(train_pred_pr, train_data_hr)

                log += f', TRAIN ' + ', '.join(f'{key}: {value:.3f}' for key, value in train_dict.items())

                if args.wandbosity and step > 0:
                    wandb_pred_hr, wandb_data_hr = dataset_train.get_wandb(train_pred_hr, train_data_hr)
                    wandb_pred_lr, wandb_data_lr = dataset_train.get_wandb(train_pred_lr, train_data_lr)
                    wandb_pred_pr, _ = dataset_train.get_wandb(train_pred_pr, train_data_hr)
                    wandb.log({
                        'train': train_dict,
                        'test.data_hr': wandb_data_hr,
                        'test.data_lr': wandb_data_lr,
                        'test.functa_hr': wandb_pred_hr,
                        'test.functa_lr': wandb_pred_lr,
                        'test.functa_pr': wandb_pred_pr,
                    })
            print(log)

            if step%args.verbosity==0:
                model_transformer.eval()
                print('begin testing...')

                test_metrics = []
                for j, (test_cord, test_data_hr) in enumerate(loader_test):
                    log = f'STEP {j+1:>5}/{len(loader_test)}, '

                    test_cord, test_data_hr = test_cord.to(device), test_data_hr.to(device)
                    test_data_lr = dataset_test.downsample(test_data_hr, args.scale)

                    test_pred_hr, test_functa_hr = model_functa.data_to_functa(test_cord, test_data_hr)
                    test_pred_lr, test_functa_lr = model_functa.data_to_functa(test_cord, test_data_lr)

                    # layer norm across features
                    test_functa_hr_mean = torch.mean(test_functa_hr, dim=1, keepdim=True)
                    test_functa_hr_std = torch.std(test_functa_hr, dim=1, keepdim=True)
                    test_functa_lr_mean = torch.mean(test_functa_lr, dim=1, keepdim=True)
                    test_functa_lr_std = torch.std(test_functa_lr, dim=1, keepdim=True)
                    test_functa_hr = (test_functa_hr - test_functa_hr_mean) / test_functa_hr_std
                    test_functa_lr = (test_functa_lr - test_functa_lr_mean) / test_functa_lr_std

                    test_functa_lr = test_functa_lr.detach().clone()
                    test_functa_hr = test_functa_hr.detach().clone()
                    torch.cuda.empty_cache()

                    test_functa_pr = model_transformer(test_functa_lr)

                    # denormalize
                    test_functa_hr = test_functa_hr * test_functa_hr_std + test_functa_hr_mean
                    test_functa_pr = test_functa_pr * test_functa_hr_std + test_functa_hr_mean
                    test_functa_lr = test_functa_lr * test_functa_lr_std + test_functa_lr_mean

                    test_pred_pr = model_functa.functa_to_data(test_cord, test_functa_pr)
                    test_metric = dataset_test.get_metric_test(test_pred_pr, test_data_hr)
                    test_metrics.append(test_metric)
                    log += f', '.join(f'{key}: {value:.3f}' for key, value in test_metric.items())

                    if args.wandbosity and step > 0:
                        wandb_pred_hr, wandb_data_hr = dataset_test.get_wandb(test_pred_hr, test_data_hr)
                        wandb_pred_lr, wandb_data_lr = dataset_test.get_wandb(test_pred_lr, test_data_lr)
                        wandb_pred_pr, _ = dataset_test.get_wandb(test_pred_pr, test_data_hr)
                        wandb.log({
                            'test': test_metric,
                            'test.data_hr': wandb_data_hr,
                            'test.data_lr': wandb_data_lr,
                            'test.functa_hr': wandb_pred_hr,
                            'test.functa_lr': wandb_pred_lr,
                            'test.functa_pr': wandb_pred_pr,
                        })
                    print(log)
                        
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
                model_transformer.train()


            step += 1
        if step > args.steps:
            break
    print('done training!')