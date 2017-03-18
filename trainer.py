from __future__ import print_function

import os
from glob import glob
from tqdm import trange
from itertools import chain

import torch
from torch import nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

from models import *

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Trainer(object):
    def __init__(self, config, a_data_loader, b_data_loader):
        self.config = config

        self.a_data_loader = a_data_loader
        self.b_data_loader = b_data_loader

        self.num_gpu = config.num_gpu
        self.dataset = config.dataset

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.optimizer = config.optimizer
        self.batch_size = config.batch_size
        self.weight_decay = config.weight_decay
        self.cnn_type = config.cnn_type

        self.model_dir = config.model_dir
        self.load_path = config.load_path

        self.start_step = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.save_step = config.save_step

        self.build_model()

        if self.num_gpu > 0:
            self.G_AB.cuda()
            self.G_BA.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

        if self.load_path:
            self.load_model()

    def build_model(self):
        if self.dataset == 'toy':
            self.G_AB = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)
            self.G_BA = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)

            self.D_A = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
            self.D_B = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
        else:
            a_height, a_width, a_channel = self.a_data_loader.shape
            b_height, b_width, b_channel = self.b_data_loader.shape

            if self.cnn_type == 0:
                #conv_dims, deconv_dims = [64, 128, 256, 512], [512, 256, 128, 64]
                conv_dims, deconv_dims = [64, 128, 256, 512], [256, 128, 64]
            elif self.cnn_type == 1:
                #conv_dims, deconv_dims = [32, 64, 128, 256], [256, 128, 64, 32]
                conv_dims, deconv_dims = [32, 64, 128, 256], [128, 64, 32]
            else:
                raise Exception("[!] cnn_type {} is not defined".format(self.cnn_type))

            self.G_AB = GeneratorCNN(
                    a_channel, b_channel, conv_dims, deconv_dims, self.num_gpu)
            self.G_BA = GeneratorCNN(
                    b_channel, a_channel, conv_dims, deconv_dims, self.num_gpu)

            self.D_A = DiscriminatorCNN(
                    a_channel, 1, conv_dims, self.num_gpu)
            self.D_B = DiscriminatorCNN(
                    b_channel, 1, conv_dims, self.num_gpu)

            self.G_AB.apply(weights_init)
            self.G_BA.apply(weights_init)

            self.D_A.apply(weights_init)
            self.D_B.apply(weights_init)

    def load_model(self):
        print("[*] Load models from {}...".format(self.load_path))

        paths = glob(os.path.join(self.load_path, 'G_AB_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.load_path))
            return

        self.start_step = int(os.path.basename(paths[-1].split('.')[0].split('_')[-1]))

        self.G_AB.load_state_dict(torch.load('{}/G_AB_{}.pth'.format(self.load_path, self.start_step)))
        self.G_BA.load_state_dict(torch.load('{}/G_BA_{}.pth'.format(self.load_path, self.start_step)))

        self.D_A.load_state_dict(torch.load('{}/D_A_{}.pth'.format(self.load_path, self.start_step)))
        self.D_B.load_state_dict(torch.load('{}/D_B_{}.pth'.format(self.load_path, self.start_step)))

        print("[*] Load {} th step model complete!".format(self.start_step))

    def train(self):
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(self.batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        if self.num_gpu > 0:
            d.cuda()
            bce.cuda()

            real_tensor = real_tensor.cuda()
            fake_tensor = fake_tensor.cuda()

        if self.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        optimizer_d = optimizer(
            chain(self.D_A.parameters(), self.D_B.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2), weight_decay=self.weight_decay)
        optimizer_g = optimizer(
            chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=self.lr, betas=(self.beta1, self.beta2))

        A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
        valid_x_A, valid_x_B = Variable(A_loader.next()), Variable(B_loader.next())

        if self.num_gpu > 0:
            valid_x_A, valid_x_B = valid_x_A.cuda(), valid_x_B.cuda()

        vutils.save_image(valid_x_A.data, '{}/valid_x_A.png'.format(self.model_dir))
        vutils.save_image(valid_x_B.data, '{}/valid_x_B.png'.format(self.model_dir))

        for step in trange(self.start_step, self.max_step):
            try:
                x_A, x_B = A_loader.next(), B_loader.next()
            except StopIteration:
                A_loader, B_loader = iter(self.a_data_loader), iter(self.b_data_loader)
                x_A, x_B = A_loader.next(), B_loader.next()

            if self.num_gpu > 0:
                x_A, x_B = Variable(x_A.cuda()), Variable(x_B.cuda())
            else:
                x_A, x_B = Variable(x_A), Variable(x_B)

            batch_size = x_A.size(0)
            real_tensor.data.resize_(batch_size).fill_(real_label)
            fake_tensor.data.resize_(batch_size).fill_(fake_label)

            # update D network
            self.D_A.zero_grad()
            self.D_B.zero_grad()

            x_AB = self.G_AB(x_A).detach()
            x_BA = self.G_BA(x_B).detach()

            x_ABA = self.G_BA(x_AB).detach()
            x_BAB = self.G_AB(x_BA).detach()

            l_d_A_real, l_d_A_fake = bce(self.D_A(x_A), real_tensor), bce(self.D_A(x_BA), fake_tensor)
            l_d_B_real, l_d_B_fake = bce(self.D_B(x_B), real_tensor), bce(self.D_B(x_AB), fake_tensor)

            l_d_A = l_d_A_real + l_d_A_fake
            l_d_B = l_d_B_real + l_d_B_fake

            l_d = l_d_A + l_d_B

            l_d.backward()
            optimizer_d.step()

            # update G network
            self.G_AB.zero_grad()
            self.G_BA.zero_grad()

            x_AB = self.G_AB(x_A)
            x_BA = self.G_BA(x_B)

            x_ABA = self.G_BA(x_AB)
            x_BAB = self.G_AB(x_BA)

            l_const_A = d(x_ABA, x_A)
            l_const_B = d(x_BAB, x_B)

            l_gan_A = bce(self.D_A(x_BA), real_tensor)
            l_gan_B = bce(self.D_B(x_AB), real_tensor)

            l_g = l_gan_A + l_gan_B + l_const_A + l_const_B

            l_g.backward()
            optimizer_g.step()

            if step % self.log_step == 0:
                print("[{}/{}] Loss_D: {:.4f} Loss_G: {:.4f}". \
                      format(step, self.max_step, l_d.data[0], l_g.data[0]))

                print("[{}/{}] l_d_A_real: {:.4f} l_d_A_fake: {:.4f}, l_d_B_real: {:.4f}, l_d_B_fake: {:.4f}". \
                      format(step, self.max_step, l_d_A_real.data[0], l_d_A_fake.data[0],  
                             l_d_B_real.data[0], l_d_B_fake.data[0]))

                print("[{}/{}] l_const_A: {:.4f} l_const_B: {:.4f}, l_gan_A: {:.4f}, l_gan_B: {:.4f}". \
                      format(step, self.max_step, l_const_A.data[0], l_const_B.data[0],  
                             l_gan_A.data[0], l_gan_B.data[0]))

                valid_x_AB = self.G_AB(valid_x_A)
                valid_x_BA = self.G_BA(valid_x_B)

                valid_x_ABA = self.G_BA(valid_x_AB)
                valid_x_BAB = self.G_AB(valid_x_BA)

                vutils.save_image(valid_x_AB.data, '{}/x_AB_{}.png'.format(self.model_dir, step))
                vutils.save_image(valid_x_BA.data, '{}/x_BA_{}.png'.format(self.model_dir, step))
                vutils.save_image(valid_x_ABA.data, '{}/x_ABA_{}.png'.format(self.model_dir, step))
                vutils.save_image(valid_x_BAB.data, '{}/x_BAB_{}.png'.format(self.model_dir, step))

            if step % self.save_step == self.save_step - 1:
                print("[*] Save models to {}...".format(self.model_dir))

                torch.save(self.G_AB.state_dict(), '{}/G_AB_{}.pth'.format(self.model_dir, step))
                torch.save(self.G_BA.state_dict(), '{}/G_BA_{}.pth'.format(self.model_dir, step))

                torch.save(self.D_A.state_dict(), '{}/D_A_{}.pth'.format(self.model_dir, step))
                torch.save(self.D_B.state_dict(), '{}/D_B_{}.pth'.format(self.model_dir, step))

    def test(self):
        pass
