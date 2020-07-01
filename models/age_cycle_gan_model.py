import numpy as np

import torch
from torch.autograd import Variable
import itertools
from util.image_pool import ImagePool
from util.util import get_label_tensor, tensor2im
from .base_model import BaseModel
from . import networks

from pdb import set_trace as ST

class AgeCycleGANModel(BaseModel):
    def name(self):
        return 'AgeCycleGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names_A = ['G_A_GAN', 'G_A_reg', 'G_A_idf', 'G_A_pixel', 'cycle_A', 'idt_G_A', 'mask_activation_A', 'mask_smooth_A', 'D_A_real_GAN', 'D_A_real_reg', 'D_A_fake_GAN']
        self.loss_names_B = ['G_B_GAN', 'G_B_reg', 'G_B_idf', 'G_B_pixel', 'cycle_B', 'idt_G_B', 'mask_activation_B', 'mask_smooth_B', 'D_B_real_GAN', 'D_B_real_reg', 'D_B_fake_GAN']
        if self.isTrain and self.opt.do_GP:
            self.loss_names_A.append('D_A_GP')
            self.loss_names_B.append('D_B_GP')

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        self.visual_names_A = ['real_A', 'fake_B_mask', 'fake_B', 'rec_A']
        self.visual_names_B = ['real_B', 'fake_A_mask', 'fake_A', 'rec_B']

        if self.isTrain and self.opt.lambda_identity > 0.0:
            self.visual_names_A.append('idt_A')
            self.visual_names_B.append('idt_B')

        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.label_nc, 
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.label_nc, 
                                        opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.fineSize, opt.output_nc, opt.label_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.fineSize, opt.input_nc, opt.label_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            # load feature extractor for identity loss if necessary
            if opt.lambda_identification > 0.0:
                self.IDfeatExtractor = networks.lightCNN_9layers(input_nc=3, gpu_ids=self.gpu_ids)
                self.IDfeatExtractor.load_pretrained_model(opt.LightCNN_pretrained_model_dir)
                self.IDfeatExtractor.eval()
                if len(self.gpu_ids) > 0:
                    assert(torch.cuda.is_available())
                    self.IDfeatExtractor.cuda(self.gpu_ids[0])
                    self.IDfeatExtractor = torch.nn.DataParallel(self.IDfeatExtractor, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            
            # define loss functions
            self.criterionGAN   = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionREG   = torch.nn.MSELoss()
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt   = torch.nn.L1Loss()
            self.criterionPixel = torch.nn.L1Loss()
            self.criterionIdf   = torch.nn.MSELoss()
            self.criterionMaskActivation = networks.mask_activation_loss
            self.criterionMaskSmooth     = networks.mask_smooth_loss
            
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if not self.isTrain or opt.continue_train:
            self.load_networks(opt.which_epoch)

        self.print_networks(opt.verbose)


    def get_face_aging_results(self):
        # intialization
        test_im = Variable(self.A, volatile=True)  # test for 'A' only, as 'B' will be sampled in other iterations
        test_age_label = self.A_age_group
        test_age_label = test_age_label[0]  # convert tensor to scalar

        generation_rets = {} # for individual generation results
        for target_age_group_index in self.age_group_index:
            target_age_group_index = target_age_group_index[0] # convert tensor to scalar

            if target_age_group_index == test_age_label:
                continue

            if target_age_group_index < test_age_label:
                generator = self.netG_B
            else:
                generator = self.netG_A

            # create label tensor
            target_age_group_tensor = get_label_tensor(self.age_group_index, target_age_group_index)
            target_age_group_tensor = torch.unsqueeze(target_age_group_tensor, 0) # add the batch dimension to label
            target_age_group_tensor = Variable(target_age_group_tensor, volatile=True)

            # age progression/regression
            chunk, mask = generator(test_im, target_age_group_tensor)
            fake_im = mask * test_im + (1 - mask) * chunk
            generation_rets[target_age_group_index] = fake_im

        # convert tensor to image
        generation_rets_im = {}
        for age_label in generation_rets.keys():
            generation_rets_im[age_label] = tensor2im(generation_rets[age_label])[0]

        return generation_rets_im


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.age_group_index = input['age_group_index']

        A = input['A' if AtoB else 'B']
        B = input['B' if AtoB else 'A']
        A_label = input['A_label' if AtoB else 'B_label']
        B_label = input['B_label' if AtoB else 'A_label']
        A_age_group = input['A_age_group' if AtoB else 'B_age_group']
        B_age_group = input['B_age_group' if AtoB else 'A_age_group']
        
        # move to gpu if possible
        if len(self.gpu_ids) > 0:
            A = A.cuda(self.gpu_ids[0], async=True)
            B = B.cuda(self.gpu_ids[0], async=True)
            A_label = A_label.cuda(self.gpu_ids[0], async=True)
            B_label = B_label.cuda(self.gpu_ids[0], async=True)

        self.A = A
        self.B = B
        self.A_label = A_label
        self.B_label = B_label
        self.A_age_group = A_age_group
        self.B_age_group = B_age_group


    def full_inference(self):

        lambda_idt = self.opt.lambda_identity

        # real images
        self.real_A = Variable(self.A, volatile=True)
        self.real_B = Variable(self.B, volatile=True)
        self.origin_label  = Variable(self.A_label, volatile=True)
        self.desired_label = Variable(self.B_label, volatile=True)

        # synthetic results
        self.fake_B_img, self.fake_B_mask = self.netG_A(self.real_A, self.desired_label)
        self.fake_B = self.fake_B_mask * self.real_A + (1 - self.fake_B_mask) * self.fake_B_img

        self.fake_A_img, self.fake_A_mask = self.netG_B(self.real_B, self.origin_label)
        self.fake_A = self.fake_A_mask * self.real_B + (1 - self.fake_A_mask) * self.fake_A_img

        # identity mapping results
        if lambda_idt > 0:
            idt_A_img, idt_A_mask = self.netG_A(self.real_A, self.origin_label)
            self.idt_A = idt_A_mask * self.real_A + (1 - idt_A_mask) * idt_A_img

            idt_B_img, idt_B_mask = self.netG_B(self.real_B, self.desired_label)
            self.idt_B = idt_B_mask * self.real_B + (1 - idt_B_mask) * idt_B_img

        # cycle reconstrucion results
        self.rec_real_img_A, self.rec_real_mask_A = self.netG_B(self.fake_B, self.origin_label)
        self.rec_A = self.rec_real_mask_A * self.fake_B + (1 - self.rec_real_mask_A) * self.rec_real_img_A

        self.rec_real_img_B, self.rec_real_mask_B = self.netG_A(self.fake_A, self.desired_label)
        self.rec_B = self.rec_real_mask_B * self.fake_A + (1 - self.rec_real_mask_B) * self.rec_real_img_B


    def forward(self):
        self.real_A = Variable(self.A)
        self.real_B = Variable(self.B)
        self.origin_label  = Variable(self.A_label) # c_s
        self.desired_label = Variable(self.B_label) # c_t

    def gradient_penalty_D(self):

        lambda_gp = self.opt.lambda_gp
        
        ##############################
        ### Forward Process (A->B) ###
        ##############################
        # interpolate sample
        curr_batchSize = self.real_A.data.size(0) # size of batch does not necessarily equals to opt.batchSize at the end of epoch
        alpha = torch.rand(curr_batchSize, 1, 1, 1).cuda().expand_as(self.real_A)
        interpolated_A = Variable(alpha * self.real_A.data + (1 - alpha) * self.fake_B.data, requires_grad=True)
        interpolated_A_prob, _ = self.netD_A(interpolated_A)

        # compute gradients
        grad_A = torch.autograd.grad(outputs=interpolated_A_prob,
                                   inputs=interpolated_A,
                                   grad_outputs=torch.ones(interpolated_A_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad_A = grad_A.view(grad_A.size(0), -1)
        grad_A_l2norm = torch.sqrt(torch.sum(grad_A ** 2, dim=1))
        self.loss_D_A_GP = torch.mean((grad_A_l2norm - 1) ** 2) * lambda_gp

        ###############################
        ### Backward Process (B->A) ###
        ###############################
        # interpolate sample
        alpha = torch.rand(curr_batchSize, 1, 1, 1).cuda().expand_as(self.real_B)
        interpolated_B = Variable(alpha * self.real_B.data + (1 - alpha) * self.fake_A.data, requires_grad=True)
        interpolated_B_prob, _ = self.netD_A(interpolated_B)

        # compute gradients
        grad_B = torch.autograd.grad(outputs=interpolated_B_prob,
                                   inputs=interpolated_B,
                                   grad_outputs=torch.ones(interpolated_B_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad_B = grad_B.view(grad_B.size(0), -1)
        grad_B_l2norm = torch.sqrt(torch.sum(grad_B ** 2, dim=1))
        self.loss_D_B_GP = torch.mean((grad_B_l2norm - 1) ** 2) * lambda_gp

        self.loss_D_GP = self.loss_D_A_GP + self.loss_D_B_GP

    def forward_D(self):
        # lambda_gan = 1, i.e. all lambda are set relative to the gan loss
        lambda_reg = self.opt.lambda_regression # lambda for label regression loss, same as in forward_G()

        ##############################
        ### Forward Process (A->B) ###
        ##############################
        
        # for real input
        pred_real_B_heatmap, pred_real_B_reglabel  = self.netD_A(self.real_B)
        self.loss_D_A_real_GAN = self.criterionGAN(pred_real_B_heatmap, True)
        self.loss_D_A_real_reg = self.criterionREG(pred_real_B_reglabel, Variable(self.desired_label.data.float())) / self.opt.batchSize * lambda_reg
        
        # for fake input
        fake_B = self.fake_B_pool.query(self.fake_B) # use the one generated by G or stored in pool
        pred_fake_B, _ = self.netD_A(fake_B.detach())
        self.loss_D_A_fake_GAN = self.criterionGAN(pred_fake_B, False)

        self.loss_D_A = self.loss_D_A_real_GAN + self.loss_D_A_real_reg + self.loss_D_A_fake_GAN

        ###############################
        ### Backward Process (B->A) ###
        ###############################
        
        # for real input
        pred_real_A_heatmap, pred_real_A_reglabel = self.netD_B(self.real_A)
        self.loss_D_B_real_GAN = self.criterionGAN(pred_real_A_heatmap, True)
        self.loss_D_B_real_reg = self.criterionREG(pred_real_A_reglabel, Variable(self.origin_label.data.float())) / self.opt.batchSize * lambda_reg

        # for fake input
        fake_A = self.fake_A_pool.query(self.fake_A) # use the one generated by G or stored in pool
        pred_fake_A, _ = self.netD_B(fake_A.detach())
        self.loss_D_B_fake_GAN = self.criterionGAN(pred_fake_A, False)

        self.loss_D_B = self.loss_D_B_real_GAN + self.loss_D_B_real_reg + self.loss_D_B_fake_GAN

        ####################
        ### Combine loss ###
        ####################
        self.loss_D = self.loss_D_A + self.loss_D_B


    def forward_G(self):
        # lambda_gan = 1, i.e. all lambda are set relative to the gan loss
        lambda_idt = self.opt.lambda_identity        # lambda for identity mapping loss
        lambda_reg = self.opt.lambda_regression      # lambda for label regression loss
        lambda_cyc = self.opt.lambda_cycle           # lambda for cycle consistency loss
        lambda_pixel = self.opt.lambda_pixel         # lambda for pixel-wise consistency loss
        lambda_idf = self.opt.lambda_identification  # lambda for identity preserving loss
        lambda_m_a = self.opt.lambda_mask_activation # lambda for mask activation loss
        lambda_m_s = self.opt.lambda_mask_smooth     # lambda for mask smooth loss

        ###############################
        ### Forward Cycle (A->B->A) ###
        ###############################

        # identity mapping and loss (idt_A = G_A(A, c_s))
        if lambda_idt > 0:
            idt_A_img, idt_A_mask = self.netG_A(self.real_A, self.origin_label)
            self.idt_A = idt_A_mask * self.real_A + (1 - idt_A_mask) * idt_A_img
            self.loss_idt_G_A = self.criterionIdt(self.idt_A, self.real_A) * lambda_idt
        else:
            self.loss_idt_G_A = 0

        # fake_B = G_A(A, c_t)
        self.fake_B_img, self.fake_B_mask = self.netG_A(self.real_A, self.desired_label)
        self.fake_B = self.fake_B_mask * self.real_A + (1 - self.fake_B_mask) * self.fake_B_img

        # pixel-wise loss and identity preserving loss
        if lambda_idf > 0:
            real_A_IDfeat = self.IDfeatExtractor(self.real_A)
            fake_B_IDfeat = self.IDfeatExtractor(self.fake_B)
            self.loss_G_A_idf = self.criterionIdf(fake_B_IDfeat, real_A_IDfeat) * lambda_idf
        else:
            self.loss_G_A_idf = 0

        if lambda_pixel > 0:
            self.loss_G_A_pixel = self.criterionPixel(self.fake_B, self.real_A) * lambda_pixel
        else:
            self.loss_G_A_pixel = 0

        # GAN loss, regression loss, and cycle loss
        d_A_heatmap, d_A_reglabel = self.netD_A(self.fake_B)
        self.loss_G_A_GAN = self.criterionGAN(d_A_heatmap, True)
        self.loss_G_A_reg = self.criterionREG(d_A_reglabel, self.desired_label) / self.opt.batchSize * lambda_reg

        # rec_A = G_B(G_A(A, c_t), c_s)
        self.rec_real_img_A, self.rec_real_mask_A = self.netG_B(self.fake_B, self.origin_label)
        self.rec_A = self.rec_real_mask_A * self.fake_B + (1 - self.rec_real_mask_A) * self.rec_real_img_A
        
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_cyc 

        # Activition loss and smooth loss for masks
        self.loss_mask_activation_A = (self.criterionMaskActivation(self.fake_B_mask) + self.criterionMaskActivation(self.rec_real_mask_A)) * lambda_m_a
        self.loss_mask_smooth_A = (self.criterionMaskSmooth(self.fake_B_mask) + self.criterionMaskSmooth(self.rec_real_mask_A)) * lambda_m_s
            

        ################################
        ### Backward Cycle (B->A->B) ###
        ################################

        # identity mapping and loss (idt_B = G_B(B, c_t))
        if lambda_idt > 0:
            idt_B_img, idt_B_mask = self.netG_B(self.real_B, self.desired_label)
            self.idt_B = idt_B_mask * self.real_B + (1 - idt_B_mask) * idt_B_img
            
            self.loss_idt_G_B = self.criterionIdt(self.idt_B, self.real_B) * lambda_idt
        else:
            self.loss_idt_G_B = 0

        # fake_A = G_B(B, c_s)
        self.fake_A_img, self.fake_A_mask = self.netG_B(self.real_B, self.origin_label)
        self.fake_A = self.fake_A_mask * self.real_B + (1 - self.fake_A_mask) * self.fake_A_img

        # pixel-wise loss and identity preserving loss
        if lambda_idf > 0:
            real_B_IDfeat = self.IDfeatExtractor(self.real_B)
            fake_A_IDfeat = self.IDfeatExtractor(self.fake_A)
            self.loss_G_B_idf = self.criterionIdf(fake_A_IDfeat, real_B_IDfeat) * lambda_idf
        else:
            self.loss_G_B_idf = 0

        if lambda_pixel > 0:
            self.loss_G_B_pixel = self.criterionPixel(self.fake_A, self.real_B) * lambda_pixel
        else:
            self.loss_G_B_pixel = 0

        # GAN loss, regression loss, and cycle loss
        d_B_heatmap, d_B_reglabel = self.netD_B(self.fake_A)
        self.loss_G_B_GAN = self.criterionGAN(d_B_heatmap, True)
        self.loss_G_B_reg = self.criterionREG(d_B_reglabel, self.origin_label) / self.opt.batchSize * lambda_reg

        # rec_B = G_A(G_B(B, c_s), c_t)
        self.rec_real_img_B, self.rec_real_mask_B = self.netG_A(self.fake_A, self.desired_label)
        self.rec_B = self.rec_real_mask_B * self.fake_A + (1 - self.rec_real_mask_B) * self.rec_real_img_B
        
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_cyc 

        # Activition loss and smooth loss for masks
        self.loss_mask_activation_B = (self.criterionMaskActivation(self.fake_A_mask) + self.criterionMaskActivation(self.rec_real_mask_B)) * lambda_m_a
        self.loss_mask_smooth_B = (self.criterionMaskSmooth(self.fake_A_mask) + self.criterionMaskSmooth(self.rec_real_mask_B)) * lambda_m_s

        ####################
        ### Combine loss ###
        ####################
        self.loss_G_A = self.loss_G_A_GAN + self.loss_G_A_reg + self.loss_G_A_idf + self.loss_G_A_pixel + \
                        self.loss_cycle_A + self.loss_idt_G_A + self.loss_mask_activation_A + self.loss_mask_smooth_A
        
        self.loss_G_B = self.loss_G_B_GAN + self.loss_G_B_reg + self.loss_G_B_idf + self.loss_G_B_pixel + \
                        self.loss_cycle_B + self.loss_idt_G_B + self.loss_mask_activation_B + self.loss_mask_smooth_B
        
        self.loss_G = self.loss_G_A + self.loss_G_B                      


    def optimize_parameters(self, train_G=True):
        # warp input to variables for preparation
        self.forward()

        # do train G
        if train_G:
            # G_A and G_B combined
            self.forward_G()
            self.optimizer_G.zero_grad()
            self.loss_G.backward()
            self.optimizer_G.step()

        # train D
        self.forward_D()
        self.optimizer_D.zero_grad()
        self.loss_D.backward()
        self.optimizer_D.step()

        # do GP if necessary
        if self.opt.do_GP:
            self.gradient_penalty_D()
            self.optimizer_D.zero_grad()
            self.loss_D_GP.backward()
            self.optimizer_D.step()