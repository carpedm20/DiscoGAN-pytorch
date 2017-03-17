import torch
from torch import nn
from torch.autograd import Variable

class Trainer(object):
    def __init__(self, a_data_loader, b_data_loader):
        self.config = config
        self.a_data_loader = a_data_loader
        self.b_data_loader = b_data_loader

        self.use_gpu = config.use_gpu
        self.dataset = config.dataset

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.build_model()

        if self.use_gpu:
            self.G_AB.cuda()
            self.G_BA.cuda()
            self.D_A.cuda()
            self.D_B.cuda()

    def build_model(self):
        if self.dataset == 'toy':
            self.G_AB = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)
            self.G_BA = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)

            self.D_A = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
            self.D_B = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
        else:
            self.G_AB = GeneratorCNN(2, 2, [hidden_dim] * g_num_layer)
            self.G_BA = GeneratorCNN(2, 2, [hidden_dim] * g_num_layer)

            self.D_A = DiscriminatorCNN(2, 1, [hidden_dim] * d_num_layer)
            self.D_B = DiscriminatorCNN(2, 1, [hidden_dim] * d_num_layer)

    def train(self):
        d = nn.MSELoss()
        bce = nn.BCELoss()

        real_label = 1
        fake_label = 0

        real_tensor = Variable(torch.FloatTensor(batch_size))
        _ = real_tensor.data.fill_(real_label)

        fake_tensor = Variable(torch.FloatTensor(batch_size))
        _ = fake_tensor.data.fill_(fake_label)

        if self.use_gpu:
            d.cuda()
            bce.cuda()

            real_tensor.cuda()
            fake_tensor.cuda()

        if config.optimizer == 'adam':
            optimizer = torch.optim.Adam
        else:
            raise Exception("[!] Caution! Paper didn't use {} opimizer other than Adam".format(config.optimizer))

        optimizer_d = optimizer(
            chain(D_A.parameters(), D_B.parameters()), lr=lr, betas=(beta1, beta2))
        optimizer_g = optimizer(
            chain(G_AB.parameters(), G_BA.parameters()), lr=lr, betas=(beta1, beta2))

    def test(self):
        pass
