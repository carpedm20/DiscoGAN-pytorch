#-*- coding: utf-8 -*-
import argparse

def str2bool(v):
    return v.lower() in ('true', '1')

arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# Network
net_arg = add_argument_group('Network')
net_arg.add_argument('--g_num_layer', type=int, default=3)
net_arg.add_argument('--d_num_layer', type=int, default=5)
net_arg.add_argument('--fc_hidden_dim', type=int, default=128, help='only for toy dataset')

# Data
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='edges2handbags')
data_arg.add_argument('--batch_size', type=int, default=200)

# Training / test parameters
train_arg = add_argument_group('Training')
train_arg.add_argument('--is_train', type=str2bool, default=True)
train_arg.add_argument('--optimizer', type=str, default='adam')
train_arg.add_argument('--max_step', type=int, default=500000)
train_arg.add_argument('--lr', type=float, default=0.0002)
train_arg.add_argument('--beta1', type=float, default=0.5)
train_arg.add_argument('--beta2', type=float, default=0.999)

# Misc
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--log_step', type=int, default=50)
misc_arg.add_argument('--num_log_samples', type=int, default=3)
misc_arg.add_argument('--log_level', type=str, default='INFO', choices=['INFO', 'DEBUG', 'WARN'])
misc_arg.add_argument('--log_dir', type=str, default='logs')
misc_arg.add_argument('--data_dir', type=str, default='data')
misc_arg.add_argument('--output_dir', type=str, default='outputs')
misc_arg.add_argument('--load_path', type=str, default='')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--random_seed', type=int, default=123)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
