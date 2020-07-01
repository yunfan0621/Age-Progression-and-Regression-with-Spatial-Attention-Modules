import os
import numpy as np
import torch
from collections import OrderedDict
import util.util as util
import torchvision.transforms as transforms

from pdb import set_trace as ST

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir)
        if opt.resize_or_crop != 'scale_width':
            torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    # return visual results in arranged tensors
    def get_current_visuals_grid_image(self):
        
        # for forward process (A)
        visual_ret_list_A = []
        for name in self.visual_names_A:
            if isinstance(name, str):
                visual_ret = getattr(self, name)
                visual_ret_im = util.tensor2im(visual_ret.data) # convert batch of samples to a list
                visual_ret_list_A.append(visual_ret_im)

        visual_ret_list_A = map(list, zip(*visual_ret_list_A))

        # for backward process (B)
        visual_ret_list_B = []
        for name in self.visual_names_B:
            if isinstance(name, str):
                visual_ret = getattr(self, name)
                visual_ret_im = util.tensor2im(visual_ret.data) # convert batch of samples to a list
                visual_ret_list_B.append(visual_ret_im)

        visual_ret_list_B = map(list, zip(*visual_ret_list_B))

        # combine results
        visual_ret = []
        for (visual_ret_A, visual_ret_B) in zip(visual_ret_list_A, visual_ret_list_B):
            visual_concat_ret_A = np.concatenate(visual_ret_A, axis=1)
            visual_concat_ret_B = np.concatenate(visual_ret_B, axis=1)
            visual_concat_ret = np.concatenate([visual_concat_ret_A, visual_concat_ret_B], axis=0)

            # convert concat results back to tensor
            visual_ret.append(transforms.ToTensor()(visual_concat_ret))

        return visual_ret

    # return traning losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret_A = OrderedDict()
        errors_ret_B = OrderedDict()
        for (name_A, name_B) in zip(self.loss_names_A, self.loss_names_B):
            if isinstance(name_A, str) and isinstance(name_B, str):
                errors_ret_A[name_A] = getattr(self, 'loss_' + name_A)
                errors_ret_B[name_B] = getattr(self, 'loss_' + name_B)
        return errors_ret_A, errors_ret_B

    # save models to the disk
    def save_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    # load models from the disk
    def load_networks(self, which_epoch):
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (which_epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = getattr(self, 'net' + name)
                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    net.module.load_state_dict(torch.load(save_path))
                else:
                    net.load_state_dict(torch.load(save_path))

    # print network information
    def print_networks(self, verbose):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requies_grad=Fasle to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def set_eval(self):
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, 'net' + net_name)
                net.eval()

    def set_train(self):
        for net_name in self.model_names:
            if isinstance(net_name, str):
                net = getattr(self, 'net' + net_name)
                net.train()