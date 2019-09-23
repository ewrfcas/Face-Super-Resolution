import os
import math
import argparse
import random
from utils import check_args, display_online_results
from data_loader import create_dataloader
from glob import glob
from models.SRGAN_model import SRGANModel
import torch


def main():
    #### options
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0,1,2,3,4,5,6,7')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dev_ratio', type=float, default=0.01)
    parser.add_argument('--lr_G', type=float, default=1e-4)
    parser.add_argument('--weight_decay_G', type=float, default=0)
    parser.add_argument('--beta1_G', type=float, default=0.9)
    parser.add_argument('--beta2_G', type=float, default=0.99)
    parser.add_argument('--lr_D', type=float, default=1e-4)
    parser.add_argument('--weight_decay_D', type=float, default=0)
    parser.add_argument('--beta1_D', type=float, default=0.9)
    parser.add_argument('--beta2_D', type=float, default=0.99)
    parser.add_argument('--lr_scheme', type=str, default='MultiStepLR')
    parser.add_argument('--niter', type=int, default=100000)
    parser.add_argument('--warmup_iter', type=int, default=-1)
    parser.add_argument('--lr_steps', type=list, default=[50000])
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--pixel_criterion', type=str, default='l1')
    parser.add_argument('--pixel_weight', type=float, default=1e-2)
    parser.add_argument('--feature_criterion', type=str, default='l1')
    parser.add_argument('--feature_weight', type=float, default=1)
    parser.add_argument('--gan_type', type=str, default='ragan')
    parser.add_argument('--gan_weight', type=float, default=5e-3)
    parser.add_argument('--D_update_ratio', type=int, default=1)
    parser.add_argument('--D_init_iters', type=int, default=0)

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--save_freq', type=int, default=10000)
    parser.add_argument('--crop_size', type=float, default=0.85)
    parser.add_argument('--lr_size', type=int, default=128)
    parser.add_argument('--hr_size', type=int, default=512)

    # network G
    parser.add_argument('--which_model_G', type=str, default='RRDBNet')
    parser.add_argument('--G_in_nc', type=int, default=3)
    parser.add_argument('--out_nc', type=int, default=3)
    parser.add_argument('--G_nf', type=int, default=64)
    parser.add_argument('--nb', type=int, default=16)

    # network D
    parser.add_argument('--which_model_D', type=str, default='discriminator_vgg_128')
    parser.add_argument('--D_in_nc', type=int, default=3)
    parser.add_argument('--D_nf', type=int, default=32)

    # data dir
    parser.add_argument('--hr_path', type=list, default=['data/celebahq-512/', 'data/ffhq-512/'])
    parser.add_argument('--lr_path', type=str, default='data/lr-128/')
    parser.add_argument('--checkpoint_dir', type=str, default='check_points/ESRGAN-V1/')
    parser.add_argument('--val_dir', type=str, default='dev_show')
    parser.add_argument('--training_state', type=str, default='check_points/ESRGAN-V1/state/')

    # resume the training
    parser.add_argument('--resume_state', type=str, default=None)
    parser.add_argument('--pretrain_model_G', type=str, default=None)
    parser.add_argument('--pretrain_model_D', type=str, default=None)

    parser.add_argument('--setting_file', type=str, default='setting.txt')
    parser.add_argument('--log_file', type=str, default='log.txt')
    args = check_args(parser.parse_args())
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    #### loading resume state if exists
    if args.resume_state is not None:
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(args.resume_state, map_location=lambda storage, loc: storage.cuda(device_id))
    else:
        resume_state = None

    # load dataset
    total_img_list = []
    for hr_path in args.hr_path:
        total_img_list.extend(glob(hr_path + '/*'))

    random.shuffle(total_img_list)
    dev_list = total_img_list[:int(len(total_img_list) * args.dev_ratio)]
    train_list = total_img_list[int(len(total_img_list) * args.dev_ratio):]

    train_loader = create_dataloader(args, train_list, is_train=True, n_threads=len(args.gpu_ids.split(',')))
    dev_loader = create_dataloader(args, dev_list, is_train=False, n_threads=len(args.gpu_ids.split(',')))

    #### create model
    model = SRGANModel(args, is_train=True)
    if resume_state is not None:
        model.load()

    #### resume training
    if resume_state is not None:
        print('Resuming training from epoch: {}, iter: {}.'.format(resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    total_epochs = int(math.ceil(args.niter / len(train_loader)))

    #### training
    print('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > args.niter:
                break
            #### update learning rate
            model.update_learning_rate(current_step, warmup_iter=args.warmup_iter)

            #### training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            #### log
            if current_step % args.print_freq == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                print(message)

            # validation
            if current_step % args.val_freq == 0:
                show_dir = os.path.join(args.checkpoint_dir, 'show_dir')
                os.makedirs(show_dir, exist_ok=True)
                dev_data = None
                for val_data in dev_loader:
                    dev_data = val_data
                    break

                model.feed_data(dev_data)
                model.test()

                visuals = model.get_current_visuals()
                display_online_results(visuals, current_step, show_dir, show_size=args.hr_size)

            #### save models and training states
            if current_step % args.save_freq == 0:
                print('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    print('Saving the final model.')
    model.save('latest')
    print('End of training.')


if __name__ == '__main__':
    main()
