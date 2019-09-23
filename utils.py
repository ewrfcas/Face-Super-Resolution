import os
import numpy as np
from PIL import Image
import cv2
import torch


def check_args(args, rank=0):
    args.setting_file = args.checkpoint_dir + args.setting_file
    args.log_file = args.checkpoint_dir + args.log_file
    if rank == 0:
        os.makedirs(args.training_state, exist_ok=True)
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        with open(args.setting_file, 'w') as opt_file:
            opt_file.write('------------ Options -------------\n')
            print('------------ Options -------------')
            for k in args.__dict__:
                v = args.__dict__[k]
                opt_file.write('%s: %s\n' % (str(k), str(v)))
                print('%s: %s' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
            print('------------ End -------------')

    return args


def read_cv2_img(path):
    '''
    Read color images
    :param path: Path to image
    :return: Only returns color images
    '''
    img = cv2.imread(path, -1)

    if img is not None:
        if len(img.shape) != 3:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


# utils
def tensor2im(input_image, imtype=np.uint8, show_size=128):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor.cpu().float().numpy()
    im = []
    for i in range(image_numpy.shape[0]):
        im.append(
            np.array(numpy2im(image_numpy[i], imtype).resize((show_size, show_size), Image.ANTIALIAS)))
    return np.array(im)


def numpy2im(image_numpy, imtype=np.uint8):
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) / 2. + 0.5) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    image_numpy = image_numpy.astype(imtype)
    im = Image.fromarray(image_numpy)
    return im


def display_online_results(visuals, steps, vis_saved_dir, show_size=128):
    images = []
    labels = []
    for label, image in visuals.items():
        image_numpy = tensor2im(image, show_size=show_size)  # [10, 128, 128, 3]
        image_numpy = np.reshape(image_numpy, (-1, show_size, 3))
        images.append(image_numpy)
        labels.append(label)
    save_images = np.array(images)  # [8, 128*10, 128, 3]
    save_images = np.transpose(save_images, [1, 0, 2, 3])
    save_images = np.reshape(save_images, (save_images.shape[0], -1, 3))
    title_img = get_title(labels, show_size)
    save_images = np.concatenate([title_img, save_images], axis=0)
    save_image(save_images, os.path.join(vis_saved_dir, 'display_' + str(steps) + '.jpg'))


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def get_title(labels, show_size=128):
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_img = []
    for label in labels:
        x = np.ones((40, show_size, 3)) * 255.0
        textsize = cv2.getTextSize(label, font, 0.5, 2)[0]
        x = cv2.putText(x, label, ((x.shape[1] - textsize[0]) // 2, x.shape[0] // 2), font, 0.5, (0, 0, 0), 1)
        title_img.append(x)

    title_img = np.array(title_img)
    title_img = np.transpose(title_img, [1, 0, 2, 3])
    title_img = np.reshape(title_img, [title_img.shape[0], -1, 3])
    title_img = title_img.astype(np.uint8)

    return title_img
