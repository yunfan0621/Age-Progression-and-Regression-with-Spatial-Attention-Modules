import os.path
# import pandas
from PIL import Image
import random
import numpy as np
import torch

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset

from pdb import set_trace as ST

class UnalignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.img_root = os.path.join(opt.dataroot, opt.phase)

        # read and parse the image list file
        self.img_list_file = os.path.join(opt.dataroot, opt.phase + '_img_list.txt')
        self.img_list, self.age_group_list = self.parse_img_list_file(self.img_list_file)

        # manage the layout of data
        self.age_group_img_dict = {} # age_group -> img
        self.img_age_group_dict = {} # img -> age_group

        # 'age_group_index' is an ordered list of age groups (e.g. [0,1,2,3,4,5,6,7,8])
        # when training, it is obtained based on the information the training set
        # when testing,  it is obtained based on how the model is trained
        if self.opt.isTrain:
            self.age_group_index = list(set(self.age_group_list)) 
        else:
            self.age_group_index = range(self.opt.label_nc)

        for index in self.age_group_index:
            self.age_group_img_dict[index] = []

        for (img, age_group) in zip(self.img_list, self.age_group_list):
            self.img_age_group_dict[img] = age_group
            self.age_group_img_dict[age_group].append(img)

        self.dataset_size = len(self.img_list)
        self.transform = get_transform(opt)


    def __getitem__(self, index):

        A_name = self.img_list[index % self.dataset_size]
        A_path = os.path.join(self.img_root, A_name)
        A_age_group = self.img_age_group_dict[A_name]

        if self.opt.isTrain:
            # sample B from another random age group
            B_age_group = random.choice(self.age_group_index)
            while B_age_group == A_age_group:
                B_age_group = random.choice(self.age_group_index)
            B_name = random.choice(self.age_group_img_dict[B_age_group])
            B_path = os.path.join(self.img_root, B_name)

            # re-order A and B to make age(A) < age(B)
            if self.opt.ordered_input:
                if B_age_group < A_age_group:
                    A_path, B_path = B_path, A_path
                    A_age_group, B_age_group = B_age_group, A_age_group
        else:
            # for testing, set B to A as dummy data
            B_path = A_path
            B_name = A_name
            B_age_group = A_age_group

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A = self.transform(A_img)
        B = self.transform(B_img)

        # create label tensor for A and B
        A_label = self.get_label_tensor(A_age_group)
        B_label = self.get_label_tensor(B_age_group)
    
        # preprocess
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)
        return {'A': A, 'B': B,
                'A_name': A_name, 'B_name': B_name,
                'A_paths': A_path, 'B_paths': B_path,
                'A_age_group': A_age_group, 'B_age_group': B_age_group,
                'A_label': A_label, 'B_label': B_label,
                'age_group_index': self.age_group_index}

    def __len__(self):
        #return max(self.A_size, self.B_size)
        return self.dataset_size
        
    def name(self):
        return 'UnalignedDataset'

    def parse_img_list_file(self, file_path):
        assert os.path.isfile(file_path), 'File %s does not exist!' % file_path

        # df = pandas.read_csv(file_path, delimiter=' ', header=None)
        img_list = []
        age_group = []
        with open(file_path, 'r') as f:
            for line in f.readlines():
                line_parts = line.strip('\n').split(' ')
                img_list.append(line_parts[0])
                age_group.append(int(line_parts[1]))

        # shuffle data list
        z = list(zip(img_list, age_group))
        random.shuffle(z)
        img_list, age_group = zip(*z)

        return img_list, age_group

    def get_label_tensor(self, age_group):
        label = np.zeros(len(self.age_group_index), dtype=float)
        label[age_group] = 1
        label = torch.from_numpy(label).float()
        label = torch.unsqueeze(label, 1)
        label = torch.unsqueeze(label, 2)
        return label
