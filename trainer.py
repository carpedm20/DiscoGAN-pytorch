
# optimizer
lr = 0.0002
beta1 = 0.5
beta2 = 0.999

if config.dataset == 'toy':
    G_AB = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)
    G_BA = GeneratorFC(2, 2, [config.fc_hidden_dim] * config.g_num_layer)

    D_A = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
    D_B = DiscriminatorFC(2, 1, [config.fc_hidden_dim] * config.d_num_layer)
else:
    G_AB = GeneratorCNN(2, 2, [hidden_dim] * g_num_layer)
    G_BA = GeneratorCNN(2, 2, [hidden_dim] * g_num_layer)

    D_A = DiscriminatorCNN(2, 1, [hidden_dim] * d_num_layer)
    D_B = DiscriminatorCNN(2, 1, [hidden_dim] * d_num_layer)

d = nn.MSELoss()
bce = nn.BCELoss()

real_label = 1
fake_label = 0

real_tensor = Variable(torch.FloatTensor(batch_size))
_ = real_tensor.data.fill_(real_label)

fake_tensor = Variable(torch.FloatTensor(batch_size))
_ = fake_tensor.data.fill_(fake_label)

if config.cuda:
    G_AB.cuda()
    G_BA.cuda()
    D_A.cuda()
    D_B.cuda()

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
