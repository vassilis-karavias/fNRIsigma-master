"""
This code is based on https://github.com/ekwebb/fNRI which in turn is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""
from __future__ import division
from __future__ import print_function

import matplotlib
import torch
import argparse
import csv
import datetime
import os
import pickle
import time
import numpy as np

import torch.optim as optim
from torch.optim import lr_scheduler

from modules_sigma import *
from utils import *
import train_functions as tfn

parser = argparse.ArgumentParser()
## arguments related to training ##
parser.add_argument('--epochs', type=int, default=50,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=128,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0005,
                    help='Initial learning rate.')
parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                    help='Num steps to predict before re-using teacher forcing.')
parser.add_argument('--lr-decay', type=int, default=200,
                    help='After how epochs to decay LR by a factor of gamma.')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='LR decay factor.')
parser.add_argument('--patience', type=int, default=500,
                    help='Early stopping patience')
parser.add_argument('--encoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--decoder-dropout', type=float, default=0.0,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dont-split-data', action='store_true', default=False,
                    help='Whether to not split training and validation data into two parts')
parser.add_argument('--split-enc-only', action='store_true', default=False,
                    help='Whether to give the encoder the first half of trajectories \
                          and the decoder the whole of the trajectories')
parser.add_argument('--loss_type', type=str, default='fixed_var',
                    help='The loss function to be used. Can be one of "fixed_var", "isotropic", "anisotropic", "semi_isotropic", "lorentzian","norminvwishart","kalmanfilter", "ani_convex" (with thanks to Edoardo Calvello for the algorithm) or "KL"')


## arguments related to loss function ##
parser.add_argument('--var', type=float, default=5e-5,
                    help='Output variance.')
parser.add_argument('--beta', type=float, default=1.0,
                    help='KL-divergence beta factor')
parser.add_argument('--mse-loss', action='store_true', default=False,
                    help='Use the MSE as the loss')

## arguments related to weight and bias initialisation ##
parser.add_argument('--seed', type=int, default=1,
                    help='Random seed.')
parser.add_argument('--encoder-init-type', type=str, default='xavier_normal',
                    help='The type of weight initialization to use in the encoder')
parser.add_argument('--decoder-init-type', type=str, default='default',
                    help='The type of weight initialization to use in the decoder')
parser.add_argument('--encoder-bias-scale', type=float, default=0.1,
                    help='The type of weight initialization to use in the encoder')

## arguments related to changing the model ##
parser.add_argument('--NRI', action='store_true', default=False,
                    help='Use the NRI model, rather than the fNRI model')
parser.add_argument('--edge-types-list', nargs='+', default=[2, 2],
                    help='The number of edge types to infer.')  # takes arguments from cmd line as: --edge-types-list 2 2
parser.add_argument('--split-point', type=int, default=0,
                    help='The point at which factor graphs are split up in the encoder')
parser.add_argument('--encoder', type=str, default='mlp',
                    help='Type of path encoder model (mlp or cnn).')
parser.add_argument('--decoder', type=str, default='mlp',
                    help='Type of decoder model (mlp, mlp3, mlpr, rnn, or sim).')
parser.add_argument('--encoder-hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--temp_softplus', type=float, default=5,
                    help='Temperature for softplus.')
parser.add_argument('--temp_sigmoid', type=float, default= 25,
                    help='Temperature for sigmoid for the changing loss function')
parser.add_argument('--skip-first', action='store_true', default=False,
                    help='Skip the first edge type in each block in the decoder, i.e. it represents no-edge.')
parser.add_argument('--hard', action='store_true', default=False,
                    help='Uses discrete samples in training forward pass.')
parser.add_argument('--soft-valid', action='store_true', default=False,
                    help='Dont use hard in validation')
parser.add_argument('--prior', action='store_true', default=False,
                    help='Whether to use sparsity prior.')

## arguments related to the simulation data ##
parser.add_argument('--sim-folder', type=str, default='springcharge_5',
                    help='Name of the folder in the data folder to load simulation data from')
parser.add_argument('--phys-folder', type=str, default='springcharge_physerrors_1',
                    help='Name of the folder in the data folder to load physical errors from')
parser.add_argument('--comp-folder', type=str, default='springcharge_comperrors',
                    help='Name of the folder in the data folder to load computational errors from')
parser.add_argument('--data-folder', type=str, default='data',
                    help='Name of the data folder to load data from')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder_dims', type=int, default=4,
                    help='The number of input dimensions for the encoder (position + velocity).')
parser.add_argument('--decoder_dims', type=int, default=8,
                    help='The number of input dimensions for the decoder (position + velocity + sigma).')
parser.add_argument('--timesteps', type=int, default=49,
                    help='The number of time steps per sample.')

## Saving, loading etc. ##
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--save-folder', type=str, default='logs',
                    help='Where to save the trained model, leave empty to not save anything.')
parser.add_argument('--load-folder', type=str, default='',
                    help='Where to load the trained model if finetunning. ' +
                         'Leave empty to train from scratch')
parser.add_argument('--test', action='store_true', default=False,
                    help='Skip training and validation')
parser.add_argument('--plot', action='store_true', default=False,
                    help='Skip training and plot trajectories against actual')
parser.add_argument('--no-edge-acc', action='store_true', default=False,
                    help='Skip training and plot accuracy distributions')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.edge_types_list = list(map(int, args.edge_types_list))
args.edge_types_list.sort(reverse=True)

# anisotropic models
if args.loss_type.lower() == 'anisotropic' or args.loss_type.lower() == 'norminvwishart' or args.loss_type.lower() == 'kalmanfilter' or args.loss_type.lower() == 'ani_convex'.lower() or args.loss_type.upper() == 'KL':
    args.anisotropic = True
else:
    args.anisotropic = False

# ensures correct decoder dimensions for 2D- comment out if not using 2D (will also have to change indices selection in
# loss function and plots will not work correctly as they were designed for 2D - e.g. for 3D position indices become [0,1,2]
# and velocity indices become [3,4,5] instead)
if args.loss_type.lower() == 'fixed_var'.lower():
    args.decoder_dims = 4
else:
    if args.loss_type.lower() == 'isotropic' or args.loss_type.lower() == 'lorentzian':
        args.decoder_dims = 5
    else:
        if args.loss_type.lower() == 'semi_isotropic'.lower():
            args.decoder_dims = 6
        else:
            if args.anisotropic:
                args.decoder_dims = 8

# for random features need 1 more dimension than normal.
if args.decoder == 'mlpr':
    if args.loss_type.lower() == 'fixed_var'.lower():
        args.decoder_dims = 5
    else:
        if args.loss_type.lower() == 'isotropic' or args.loss_type.lower() == 'lorentzian':
            args.decoder_dims = 6
        else:
            if args.loss_type.lower() == 'semi_isotropic'.lower():
                args.decoder_dims = 7
            else:
                if args.anisotropic:
                    args.decoder_dims = 9

if args.NRI:
    print('Using NRI model')
    if args.split_point != 0:
        args.split_point = 0
print(args)

if all((isinstance(k, int) and k >= 1) for k in args.edge_types_list):
    if args.NRI:
        edge_types = np.prod(args.edge_types_list)
    else:
        edge_types = sum(args.edge_types_list)
else:
    raise ValueError('Could not compute the edge-types-list')

if args.prior:
    prior = [[0.9, 0.1], [0.9, 0.1]]  # TODO: hard coded for now
    if not all(prior[i].size == args.edge_types_list[i] for i in range(len(args.edge_types_list))):
        raise ValueError('Prior is incompatable with the edge types list')
    print("Using prior: " + str(prior))
    log_prior = []
    for i in range(len(args.edge_types_list)):
        prior_i = np.array(prior[i])
        log_prior_i = torch.FloatTensor(np.log(prior))
        log_prior_i = torch.unsqueeze(log_prior_i, 0)
        log_prior_i = torch.unsqueeze(log_prior_i, 0)
        log_prior_i = Variable(log_prior_i)
        log_prior.append(log_prior_i)
    if args.cuda:
        log_prior = log_prior.cuda()
else:
    log_prior = []

# Save model and meta-data. Always saves in a new sub-folder.
# Saves best model. Also saves current epoch model -> BUS errors make it impossible to run the code without doing this
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat().replace(':', '-')[:-7]
    save_folder = os.path.join(args.save_folder, 'exp' + timestamp)
    save_folder_currentepoch = os.path.join(args.save_folder, 'exp' + timestamp + 'currentepoch')
    os.makedirs(save_folder)
    os.makedirs(save_folder_currentepoch)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')
    current_encoder_file = os.path.join(save_folder_currentepoch, 'encoder.pt')
    current_decoder_file = os.path.join(save_folder_currentepoch, 'decoder.pt')
    loss_data_file = os.path.join(save_folder_currentepoch, 'loss_data.txt')
    log_file = os.path.join(save_folder, 'log.txt')
    log_csv_file = os.path.join(save_folder, 'log_csv.csv')
    log = open(log_file, 'w')
    log_csv = open(log_csv_file, 'w')
    loss_data = open(loss_data_file, 'w')
    csv_writer = csv.writer(log_csv, delimiter=',')
    pickle.dump({'args': args}, open(meta_file, "wb"))
    par_file = open(os.path.join(save_folder, 'args.txt'), 'w')
    print(args, file=par_file)
    par_file.flush
    par_file.close()

    perm_csv_file = os.path.join(save_folder, 'perm_csv.csv')
    perm_csv = open(perm_csv_file, 'w')
    perm_writer = csv.writer(perm_csv, delimiter=',')
else:
    encoder_file = None
    decoder_file = None
    current_encoder_file = None
    current_decoder_file = None
    log = None
    loss_data = None
    csv_writer = None
    perm_writer = None

    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if args.NRI:
    edge_types_list = [edge_types]
else:
    edge_types_list = args.edge_types_list

if args.encoder == 'mlp':
    encoder = MLPEncoder_multi(args.timesteps * args.encoder_dims, args.encoder_hidden,
                                    edge_types_list, args.encoder_dropout,
                                    split_point=args.split_point,
                                    init_type=args.encoder_init_type,
                                    bias_init=args.encoder_bias_scale)

# 2-Layer MLP
if args.decoder == 'mlp':
    decoder = MLPDecoder_multi(n_in_node=args.decoder_dims,
                                    edge_types=edge_types,
                                    edge_types_list=edge_types_list,
                                    msg_hid=args.decoder_hidden,
                                    msg_out=args.decoder_hidden,
                                    n_hid=args.decoder_hidden,
                                    do_prob=args.decoder_dropout,
                                    skip_first=args.skip_first,
                                    init_type=args.decoder_init_type)

# 3-layer MLP
if args.decoder == 'mlp3':
    decoder = MLPDecoder_multi_threelayers(n_in_node=args.decoder_dims,
                                                edge_types=edge_types,
                                                edge_types_list=edge_types_list,
                                                msg_hid=args.decoder_hidden,
                                                msg_out=args.decoder_hidden,
                                                n_hid=args.decoder_hidden,
                                                do_prob=args.decoder_dropout,
                                                skip_first=args.skip_first,
                                                init_type=args.decoder_init_type)
# 2-layer MLP with random features
if args.decoder == 'mlpr':
    decoder = MLPDecoder_multi_randomfeatures(n_in_node=args.decoder_dims,
                                                   edge_types=edge_types,
                                                   edge_types_list=edge_types_list,
                                                   msg_hid=args.decoder_hidden,
                                                   msg_out=args.decoder_hidden,
                                                   n_hid=args.decoder_hidden,
                                                   do_prob=args.decoder_dropout,
                                                   skip_first=args.skip_first,
                                                   init_type=args.decoder_init_type)

# loading a pre-existing model
if args.load_folder:
    print('Loading model from: ' + args.load_folder)
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    if not args.cuda:
        encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
        decoder.load_state_dict(torch.load(decoder_file, map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))
    # comment this line out if you want to retrain after training once
    args.save_folder = False

# optimiser - use an Adam optimiser
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                            lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                     gamma=args.gamma)

if args.cuda:
    encoder.cuda()
    decoder.cuda()

# Train model
if not args.test:
    if not args.plot:
        # training without plotting
        t_total = time.time()
        best_val_loss = np.inf
        best_epoch = 0
        trainer = tfn.Trainer(args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler)
        for epoch in range(args.epochs):
            # trainer class and then runs an epoch of training
            acc_val, val_loss, acc_perlayer = trainer.train(epoch, best_val_loss, args)
            # condition for saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
            if epoch - best_epoch > args.patience and epoch > 99:
                break
        print("Optimization Finished!")
        print("Best Epoch: {:04d}".format(best_epoch))
        if args.save_folder:
            print("Best Epoch: {:04d}".format(best_epoch), file=trainer.log)
            trainer.log.flush()
    else:
        # plot graphs for validation set ignoring the training
        trainer = tfn.Trainer(args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler)
        trainer.train_plot(0, args)

# test
tester = tfn.Tester(args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler)
tester.test(args)
tester.close_log(save_folder, log_csv, perm_csv)


