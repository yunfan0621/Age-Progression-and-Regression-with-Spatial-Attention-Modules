from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from torch import is_tensor
from torch.autograd import Variable

from pdb import set_trace as ST

# create a C x 1 x 1 tensor with 'age_group'th channel set to 1
def get_label_tensor(age_group_index, age_group):
        label = np.zeros(len(age_group_index), dtype=float)
        label[age_group] = 1
        label = torch.from_numpy(label).float()
        label = torch.unsqueeze(label, 1)
        label = torch.unsqueeze(label, 2)
        return label
        
# Converts a Tensor into a float
def tensor2float(input_error):
    if is_tensor(input_error):
        error = input_error[0]
    elif isinstance(input_error, Variable):
        error = input_error.data[0]
    else:
        error = input_error
    return error


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if is_tensor(input_image):
        image_tensor = input_image
    elif isinstance(input_image, Variable):
        image_tensor = input_image.data
    else:
        return input_image

    image_list = []
    N, C, W, H = image_tensor.size()
    for i in range(N):
        image_numpy = image_tensor[i].cpu().float().numpy()
        if image_numpy.shape[0] == 1:
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        image_list.append(image_numpy.astype(imtype))

    return image_list


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def print_current_errors(epoch, i, errors_forward, errors_backward, opt):
    message = '(epoch: %d, iters: %d)\n' % (epoch, i)
    for k, v in errors_forward.items():
        message += '%s: %.4f ' % (k, v)
    message += '\n'
    for k, v in errors_backward.items():
        message += '%s: %.4f ' % (k, v)
    message += '\n'

    print(message)
    log_name = os.path.join(opt.checkpoints_dir, opt.name, opt.suffix, 'loss_log.txt')
    with open(log_name, "a") as log_file:
        log_file.write('%s\n' % message)

