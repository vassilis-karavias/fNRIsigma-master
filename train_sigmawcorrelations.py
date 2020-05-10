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

parser = argparse.ArgumentParser()
## arguments related to training ##
parser.add_argument('--epochs', type=int, default=500,
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
parser.add_argument('--fixed-var', type=bool, default=False,
                    help='If true will use a fixed small variance. If false will solve with variable variance.')
parser.add_argument('--isotropic', type=bool, default=False,
                    help='If true use anisotropic sigma. If more that one of fixed-var, anisotropic and semi-isotropic is true will use least complex.')
parser.add_argument('--anisotropic', type=bool, default=True,
                    help='If true use anisotropic sigma. If more that one of fixed-var, anisotropic and semi-isotropic is true will use least complex.')
parser.add_argument('--semi_isotropic', type=bool, default= False,
                    help='If true use semi-isotropic sigma (isotropic in (x,y) and isotropic in (vx,vy)). If more that one of fixed-var, anisotropic and semi-isotropic is true will use least complex.')


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
                    help='Type of decoder model (mlp, mlp3, rnn, or sim).')
parser.add_argument('--encoder-hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--decoder-hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--temp', type=float, default=0.5,
                    help='Temperature for Gumbel softmax.')
parser.add_argument('--temp_softplus', type=float, default= 5,
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
parser.add_argument('--data-folder', type=str, default='data',
                    help='Name of the data folder to load data from')
parser.add_argument('--num-atoms', type=int, default=5,
                    help='Number of atoms in simulation.')
parser.add_argument('--encoder_dims', type=int, default=4,
                    help='The number of input dimensions for the encoder (position + velocity).')
parser.add_argument('--decoder_dims', type=int, default=12,
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

if all((isinstance(k, int) and k >= 1) for k in args.edge_types_list):
    if args.NRI:
        edge_types = np.prod(args.edge_types_list)
    else:
        edge_types = sum(args.edge_types_list)
else:
    raise ValueError('Could not compute the edge-types-list')

# uses least complex of isotropic, semi-isotropic and anisotropic if more than 1 is true:
if not (args.fixed_var):
    if args.isotropic and (args.semi_isotropic):
        args.semi_isotropic = False
        if (args.anisotropic):
            args.anisotropic = False
        print('More than one of isotropic, semi-isotropic and anisotropic is true. Using isotropic.')
    else:
        if args.isotropic and args.anisotropic:
            args.anisotropic = False
            print('More than one of isotropic, semi-isotropic and anisotropic is true. Using isotropic.')
        else:
            if args.semi_isotropic and args.anisotropic:
                args.anisotropic = False
                print('More than one of isotropic, semi-isotropic and anisotropic is true. Using semi-isotropic.')

# if args.fixed_var:
#     args.decoder_dims = 4
# else:
#     if args.isotropic:
#         args.decoder_dims = 5
#     else:
#         if args.semi_isotropic:
#             args.decoder_dims = 6
#         else:
#             if args.anisotropic:
#                 args.decoder_dims = 8

if args.NRI:
    print('Using NRI model')
    if args.split_point != 0:
        args.split_point = 0
print(args)

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

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

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
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")

if args.NRI:
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data_NRI(
        args.batch_size, args.sim_folder, shuffle=True,
        data_folder=args.data_folder)
else:
    train_loader, valid_loader, test_loader, loc_max, loc_min, vel_max, vel_min = load_data_fNRI(
        args.batch_size, args.sim_folder, shuffle=True,
        data_folder=args.data_folder)

# Generate off-diagonal interaction graph
off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
rel_rec = torch.FloatTensor(rel_rec)
rel_send = torch.FloatTensor(rel_send)

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

# elif args.encoder == 'cnn':
#     encoder = CNNEncoder_multi(args.dims, args.encoder_hidden,
#                                edge_types_list,
#                                args.encoder_dropout,
#                                split_point=args.split_point,
#                                init_type=args.encoder_init_type)
#
# elif args.encoder == 'random':
#     encoder = RandomEncoder(args.edge_types_list, args.cuda)
#
# elif args.encoder == 'ones':
#     encoder = OnesEncoder(args.edge_types_list, args.cuda)

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

# elif args.decoder == 'stationary':
#     decoder = StationaryDecoder()
#
# elif args.decoder == 'velocity':
#     decoder = VelocityStepDecoder()

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
    # we want to save the new model to test if pre-training works
    args.save_folder = False

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)

scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay,
                                gamma=args.gamma)

if args.cuda:
    encoder.cuda()
    decoder.cuda()
    rel_rec = rel_rec.cuda()
    rel_send = rel_send.cuda()

rel_rec = Variable(rel_rec)
rel_send = Variable(rel_send)



def train(epoch, best_val_loss):
    t = time.time()
    nll_train = []
    nll_var_train = []
    mse_train = []

    kl_train = []
    kl_list_train = []
    kl_var_list_train = []

    acc_train = []
    acc_var_train = []
    perm_train = []
    acc_var_blocks_train = []
    acc_blocks_train = []

    KLb_train = []
    KLb_blocks_train = []

    # array of loss components
    loss_1_array = []
    loss_2_array = []
    # gets an array of the sigma tensor per run through of the batch
    sigmadecoderoutput = []

    encoder.train()
    decoder.train()
    scheduler.step()
    if not args.plot:
        for batch_idx, (data, relations) in enumerate(train_loader):  # relations are the ground truth interactions graphs
            # tottime = time.time()
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data), Variable(relations)

            if args.dont_split_data:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, :args.timesteps, :].contiguous()
            elif args.split_enc_only:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data
            else:
                # assert (data.size(2) - args.timesteps) >= args.timesteps
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            # stores the values of the uncertainty. This will be an array of size [batchsize, no. of particles, time,no. of axes (isotropic = 1, anisotropic = 4]
            # initialise sigma to an array large negative numbers, under softplus function this will make them small positive numbers
            sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms, inversesoftplus(pow(args.var,1/2), args.temp_softplus), ani_dims=8)
            if args.cuda:
                sigma = sigma.cuda()
            if args.semi_isotropic:
                sigma = tile(sigma, 3, 2)
            sigma = Variable(sigma)
            optimizer.zero_grad()

            logits = encoder(data_encoder, rel_rec, rel_send)

            if args.NRI:
                # dim of logits, edges and prob are [batchsize, N^2-N, edgetypes] where N = no. of particles
                edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
                prob = my_softmax(logits, -1)

                loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
                loss_kl_split = [loss_kl]
                loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, edge_types)]

                KLb_train.append(0)
                KLb_blocks_train.append([0])

                if args.no_edge_acc:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = 0, np.array([0]), np.zeros(len(args.edge_types_list)), 0, np.zeros(len(args.edge_types_list))
                else:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_NRI(logits, relations, args.edge_types_list)

            else:
                # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
                logits_split = torch.split(logits, args.edge_types_list, dim=-1)
                edges_split = tuple([gumbel_softmax(logits_i, tau=args.temp, hard=args.hard)
                                     for logits_i in logits_split])
                edges = torch.cat(edges_split, dim=-1)
                prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split]

                if args.prior:
                    loss_kl_split = [kl_categorical(prob_split[type_idx], log_prior[type_idx], args.num_atoms)
                                     for type_idx in range(len(args.edge_types_list))]
                    loss_kl = sum(loss_kl_split)
                else:
                    loss_kl_split = [kl_categorical_uniform(prob_split[type_idx], args.num_atoms,
                                                            args.edge_types_list[type_idx])
                                     for type_idx in range(len(args.edge_types_list))]
                    loss_kl = sum(loss_kl_split)

                    loss_kl_var_split = [kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms,
                                                                    args.edge_types_list[type_idx])
                                         for type_idx in range(len(args.edge_types_list))]

                if args.no_edge_acc:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = 0, np.array([0]), np.zeros(len(args.edge_types_list)), 0, np.zeros(len(args.edge_types_list))
                else:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_fNRI(logits_split, relations,
                                                                                                  args.edge_types_list, args.skip_first)

                KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                KLb_train.append(sum(KLb_blocks).data.item())
                KLb_blocks_train.append([KL.data.item() for KL in KLb_blocks])
            # fixed variance
            if args.fixed_var:
                target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                # forward() in decoder called here - will need to alter decoder to include sigma
                output, sigma, accel, vel = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, args.prediction_steps)
                loss_nll = nll_gaussian(output, target, args.var)
                loss_nll_var = nll_gaussian_var(output, target, args.var)
            # variable variance
            else:
                target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]


                if args.anisotropic:
                    # forward() in decoder called here - will need to alter decoder to include sigma
                    # ti = time.time()
                    output, sigma, accel, vel = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, args.prediction_steps)
                    # print('decodertime: {:.1f}s'.format(time.time() - ti))
                    sigmadecoderoutput.append(sigma)
                    ### here needs an anisotropic Loss function with sigmas along the 4 directions (along vi, ai and perp to vi, ai )
                    # tim = time.time()
                    loss_nll, loss_1, loss_2 = nll_gaussian_multivariatesigma_withcorrelations(output, target, sigma, accel, vel)
                    loss_nll_var = nll_gaussian_var_multivariatesigma_withcorrelations(output, target,sigma, accel, vel)
                    loss_1_array.append(loss_1)
                    loss_2_array.append(loss_2)
                    # print('losstime: {:.1f}s'.format(time.time() - tim))
                else:
                    if args.isotropic:
                        # forward() in decoder called here - will need to alter decoder to include sigma
                        # ti = time.time()
                        output, sigma, accel, vel = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, args.prediction_steps)
                        # print('decodertime: {:.1f}s'.format(time.time() - ti))
                        sigmadecoderoutput.append(sigma)
                        # in case of isotropic we need to recast sigma to the same shape as output as it is required in the gaussian function
                        sigma = tile(sigma, 3, list(output.size())[3])
                        # tim = time.time()
                        loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma(output, target, sigma, epoch , args.temp_sigmoid, args.epochs)
                        loss_nll_var = nll_gaussian_var__variablesigma(output, target, sigma, epoch, args.temp_sigmoid, args.epochs)
                        loss_1_array.append(loss_1)
                        loss_2_array.append(loss_2)
                        # print('losstime: {:.1f}s'.format(time.time() - tim))
                    if args.semi_isotropic:
                        output, sigma, accel, vel = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False,
                                                       args.temp_softplus, args.prediction_steps)
                        # print('decodertime: {:.1f}s'.format(time.time() - ti))
                        sigmadecoderoutput.append(sigma)
                        # tim = time.time()
                        loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma_semiisotropic(output, target, sigma, epoch,
                                                                              args.temp_sigmoid, args.epochs)
                        loss_nll_var = nll_gaussian_var__variablesigma_semiisotropic(output, target, sigma, epoch, args.temp_sigmoid,
                                                                       args.epochs)
                        loss_1_array.append(loss_1)
                        loss_2_array.append(loss_2)
                # print('totaltime: {:.1f}s'.format(time.time() - tottime))
            if args.mse_loss:
                loss = F.mse_loss(output, target)
            else:
                loss = loss_nll
                if not math.isclose(args.beta, 0, rel_tol=1e-6):
                    loss += args.beta * loss_kl

            perm_train.append(perm)
            acc_train.append(acc_perm)
            acc_blocks_train.append(acc_blocks)
            acc_var_train.append(acc_var)
            acc_var_blocks_train.append(acc_var_blocks)

            loss.backward()
            optimizer.step()

            mse_train.append(F.mse_loss(output, target).data.item())
            nll_train.append(loss_nll.data.item())
            kl_train.append(loss_kl.data.item())
            kl_list_train.append([kl.data.item() for kl in loss_kl_split])

            nll_var_train.append(loss_nll_var.data.item())
            kl_var_list_train.append([kl_var.data.item() for kl_var in loss_kl_var_split])

        if (args.plot):
            if not(args.fixed_var):
                import matplotlib.pyplot as plt
                # gets the iterations array [1,2, ..... , final]
                iteration = np.linspace(1,len(loss_1_array), len(loss_1_array));
                # plots Loss_1 and Loss_2 vs iteration normalised by total loss
                for i in range(len(loss_1_array)):
                    loss = loss_1_array[i] + loss_2_array[i]
                    loss_1_array[i] = loss_1_array[i] / loss
                    loss_2_array[i] = loss_2_array[i] / loss

                fig = plt.figure()
                plt.plot(iteration, loss_1_array, label = 'loss 1')
                plt.plot(iteration, loss_2_array, label = 'loss 2')
                plt.xlabel('iteration')
                plt.ylabel('Loss Component/Total Loss')
                plt.legend('loss 1', 'loss 2')
                plt.show()

    nll_val = []
    nll_var_val = []
    mse_val = []

    kl_val = []
    kl_list_val = []
    kl_var_list_val = []

    acc_val = []
    acc_var_val = []
    acc_blocks_val = []
    acc_var_blocks_val = []
    perm_val = []

    KLb_val = []
    KLb_blocks_val = []  # KL between blocks list

    nll_M_val = []
    nll_M_var_val = []

    # for z-score analysis
    zscorelist_x = []
    zscorelist_y = []

    encoder.eval()
    decoder.eval()
    for batch_idx, (data, relations) in enumerate(valid_loader):
        with torch.no_grad():
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            if args.dont_split_data:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, :args.timesteps, :].contiguous()
            elif args.split_enc_only:
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data
            else:
                assert (data.size(2) - args.timesteps) >= args.timesteps
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            # stores the values of the uncertainty. This will be an array of size [batchsize, no. of particles, time,no. of axes (isotropic = 1, anisotropic = 4)]
            # initialise sigma to an array of large negative numbers which become small positive numbers when passed through softplus
            sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms, inversesoftplus(pow(args.var,1/2), args.temp_softplus), ani_dims=8)
            if args.cuda:
                sigma = sigma.cuda()
            if args.semi_isotropic:
                sigma = tile(sigma, 3, 2)
            # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
            logits = encoder(data_encoder, rel_rec, rel_send)

            if args.NRI:
                # dim of logits, edges and prob are [batchsize, N^2-N, edgetypes] where N = no. of particles
                edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)  # uses concrete distribution (for hard=False) to sample edge types
                prob = my_softmax(logits, -1)  # my_softmax returns the softmax over the edgetype dim

                loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
                loss_kl_split = [loss_kl]
                loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, edge_types)]

                KLb_val.append(0)
                KLb_blocks_val.append([0])

                if args.no_edge_acc:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = 0, np.array([0]), np.zeros(len(args.edge_types_list)), 0, np.zeros(len(args.edge_types_list))
                else:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_NRI(logits, relations, args.edge_types_list)

            else:
                # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
                logits_split = torch.split(logits, args.edge_types_list, dim=-1)
                edges_split = tuple([gumbel_softmax(logits_i, tau=args.temp, hard=args.hard)
                                     for logits_i in logits_split])
                edges = torch.cat(edges_split, dim=-1)
                prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split]

                if args.prior:
                    loss_kl_split = [kl_categorical(prob_split[type_idx], log_prior[type_idx], args.num_atoms)
                                     for type_idx in range(len(args.edge_types_list))]
                    loss_kl = sum(loss_kl_split)
                else:
                    loss_kl_split = [kl_categorical_uniform(prob_split[type_idx], args.num_atoms,
                                                            args.edge_types_list[type_idx])
                                     for type_idx in range(len(args.edge_types_list))]
                    loss_kl = sum(loss_kl_split)

                    loss_kl_var_split = [kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms,
                                                                    args.edge_types_list[type_idx])
                                         for type_idx in range(len(args.edge_types_list))]

                if args.no_edge_acc:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = 0, np.array([0]), np.zeros(len(args.edge_types_list)), 0, np.zeros(len(args.edge_types_list))
                else:
                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_fNRI(logits_split, relations,
                                                                                                  args.edge_types_list, args.skip_first)

                KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                KLb_val.append(sum(KLb_blocks).data.item())
                KLb_blocks_val.append([KL.data.item() for KL in KLb_blocks])
            if args.fixed_var:
                target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                # one prediction step
                output, sigma, accel, vel = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, 1)

                if args.plot:
                    import matplotlib.pyplot as plt
                    output_plot, sigma_plot, accel_plot, vel_plot = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, 49)

                    if args.NRI:
                        acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                         args.edge_types_list)
                    else:
                        acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split, relations,
                                                                                          args.edge_types_list)

                    from trajectory_plot import draw_lines

                    for i in range(args.batch_size):
                        fig = plt.figure(figsize=(7, 7))
                        ax = fig.add_axes([0, 0, 1, 1])
                        xmin_t, ymin_t, xmax_t, ymax_t = draw_lines(target, i, linestyle=':', alpha=0.6)
                        xmin_o, ymin_o, xmax_o, ymax_o = draw_lines(output_plot.detach().cpu().numpy(), i, linestyle='-')
                        ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                        ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                        ax.set_xticks([])
                        ax.set_yticks([])
                        block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
                        # block_names = [ 'springs', 'charges' ]
                        acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                                    for j in range(acc_blocks_batch.shape[1])]
                        acc_text = ', '.join(acc_text)
                        plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)

                        # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
                        plt.show()

                loss_nll = nll_gaussian(output, target, args.var)
                loss_nll_var = nll_gaussian_var(output, target, args.var)
                # all prediction steps needed
                output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, args.prediction_steps)
                loss_nll_M = nll_gaussian(output_M, target, args.var)
                loss_nll_M_var = nll_gaussian_var(output_M, target, args.var)

                perm_val.append(perm)
                acc_val.append(acc_perm)
                acc_blocks_val.append(acc_blocks)
                acc_var_val.append(acc_var)
                acc_var_blocks_val.append(acc_var_blocks)

                mse_val.append(F.mse_loss(output_M, target).data.item())
                nll_val.append(loss_nll.data.item())
                nll_var_val.append(loss_nll_var.data.item())

                kl_val.append(loss_kl.data.item())
                kl_list_val.append([kl_loss.data.item() for kl_loss in loss_kl_split])
                kl_var_list_val.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                nll_M_val.append(loss_nll_M.data.item())
                nll_M_var_val.append(loss_nll_M_var.data.item())
            else:
                if not args.anisotropic:
                    target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                    # in case of isotropic we need to recast sigma to the same shape as output as it is required in the gaussian function
                    output, sigmaone, accelone, velone = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 1)

                    if args.plot:
                        import matplotlib.pyplot as plt
                        output_plot, sigma_plot, accel_plot, vel_plot = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 49)
                        if args.isotropic:
                            sigma_plot = tile(sigma_plot, 3, list(output_plot.size())[3])
                        else:
                            indices_pos = torch.LongTensor([0])
                            indices_vel = torch.LongTensor([1])
                            if args.cuda:
                                indices_pos = indices_pos.cuda()
                            sigma_plot_pos = torch.index_select(sigma_plot, 3, indices_pos)
                            sigma_plot_vel = torch.index_select(sigma_plot, 3, indices_vel)
                            sigma_plot_pos = tile(sigma_plot_pos, 3, 2)
                            sigma_plot_vel = tile(sigma_plot_vel, 3, 2)
                            sigma_plot = torch.cat((sigma_plot_pos, sigma_plot_vel), 3)
                        if args.NRI:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                             args.edge_types_list)
                        else:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split, relations,
                                                                                              args.edge_types_list)

                        from trajectory_plot import draw_lines_sigma
                        from matplotlib.patches import Ellipse
                        for i in range(args.batch_size):
                            fig = plt.figure(figsize=(7, 7))
                            ax = fig.add_axes([0, 0, 1, 1])
                            xmin_t, ymin_t, xmax_t, ymax_t = -1, -1, 1, 1
                            xmin_o, ymin_o, xmax_o, ymax_o = -0.5, -0.5, 0.5, 0.5
                            xmin_t, ymin_t, xmax_t, ymax_t = draw_lines_sigma(target, i, sigma_plot.detach().cpu().numpy(),ax, linestyle=':', alpha=0.6)
                            xmin_o, ymin_o, xmax_o, ymax_o = draw_lines_sigma(output_plot.detach().cpu().numpy(), i, sigma_plot.detach().cpu().numpy(),ax, linestyle='-', plot_ellipses=True)
                            # isotropic therefore the ellipses become circles
    #                         indices = torch.LongTensor([0, 1])
    #                         if args.cuda:
    #                             indices = indices.cuda()
    #                         positions = torch.index_select(output_plot, 3, indices)
    #                         sigma_plot_pos = torch.index_select(sigma_plot, 3, indices)
    #
    #                         # iterate through each of the atoms
    #                         for j in range(positions.size()[1]):
    #                             ellipses = []
    #                             # get the first timestep component of (x,y)
    #                             ellipses.append(Ellipse((positions.tolist()[i][j][0][0], positions.tolist()[i][j][0][1]), width = sigma_plot_pos.tolist()[i][j][0][0], height = sigma_plot_pos.tolist()[i][j][0][0], angle = 0.0))
    #                             # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
    #                             # keeps track of current plot value
    #                             l=0
    #                             for k in range(positions.size()[2]-1):
    #                                 deltar = (torch.from_numpy(positions.cpu().numpy()[i][j][k+1])- torch.from_numpy(positions.numpy()[i][j][l])).norm(p=2, dim=0, keepdim=True)
    #                                 deltasigma = (torch.from_numpy(sigma_plot_pos.cpu().numpy()[i][j][l])).norm(p=2, dim=0, keepdim=True)
    #                                 if (deltar.item()>2*deltasigma.item()):
    #                                     # check that it is far away from others
    #                                     isfarapart = True
    #                                     for m in range(positions.size()[1]):
    #                                         for n in range(positions.size()[2]):
    #                                             if (m!= j):
    #                                                 deltar =  (torch.from_numpy(positions.cpu().numpy()[i][m][n])- torch.from_numpy(positions.numpy()[i][j][k+1])).norm(p=2, dim=0, keepdim=True)
    #                                                 deltasigma = (torch.from_numpy(sigma_plot_pos.cpu().numpy()[i][j][k+1])).norm(p=2, dim=0, keepdim=True)
    #                                                 if (deltar< deltasigma.item()):
    #                                                     isfarapart = False
    #                                     if isfarapart:
    #                                         ellipses.append(Ellipse((positions.tolist()[i][j][k+1][0], positions[i][j][k+1][1]), width = sigma_plot_pos.tolist()[i][j][k+1][0], height = sigma_plot_pos.tolist()[i][j][k+1][0], angle = 0.0))
    #                                         # updates to new r0 : Deltar = r - r0:
    #                                         l = k
    #                         # fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    #                             colour = np.random.rand(3)
    #                             for e in ellipses:
    #                                 ax.add_artist(e)
    #                                 e.set_clip_box(ax.bbox)
    # #                               e.set_alpha(0.6)
    #                                 e.set_facecolor(colour)
                            ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                            ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
                            # block_names = [ 'springs', 'charges' ]
                            acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                                        for j in range(acc_blocks_batch.shape[1])]
                            acc_text = ', '.join(acc_text)
                            plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
                            # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
                            plt.show()
                        # for z score
                        # make sure we aren't dividing by 0
                        if (torch.min(sigma_plot)< pow(10, -7)):
                            accuracy = np.full((sigma_plot.size(0), sigma_plot.size(1), sigma_plot.size(2), sigma_plot.size(3)), pow(10, -7),dtype=np.float32)
                            accuracy = torch.from_numpy(accuracy)
                            if args.cuda:
                                accuracy = accuracy.cuda()
                            output_plot = torch.max(output_plot, accuracy)
                        zscore = (output_plot - target) / (sigma_plot)
                        indices = torch.LongTensor([2, 3])
                        if args.cuda:
                            indices = indices.cuda()
                        velocities = torch.index_select(output_plot, 3, indices)
                        velnorm = velocities.norm(p=2, dim=3, keepdim=True)
                        normalisedvel = velocities.div(velnorm.expand_as(velocities))
                        accelnorm = accel_plot.norm(p=2, dim=3, keepdim=True)
                        normalisedaccel = accel_plot.div(accelnorm.expand_as(accel_plot))
                        # get perpendicular components to the accelerations and velocities accelperp, velperp
                        # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
                        rotationmatrix = np.zeros((velocities.size(0), velocities.size(1), velocities.size(2), 2, 2),
                                                  dtype=np.float32)
                        for i in range(len(rotationmatrix)):
                            for j in range(len(rotationmatrix[i])):
                                for l in range(len(rotationmatrix[i][j])):
                                    rotationmatrix[i][j][l][0][1] = np.float32(-1)
                                    rotationmatrix[i][j][l][1][0] = np.float32(1)
                        rotationmatrix = torch.from_numpy(rotationmatrix)
                        if args.cuda:
                            rotationmatrix = rotationmatrix.cuda()
                        velperp = torch.matmul(rotationmatrix, normalisedvel.unsqueeze(4))
                        velperp = velperp.squeeze()
                        accelperp = torch.matmul(rotationmatrix, normalisedaccel.unsqueeze(4))
                        accelperp = accelperp.squeeze()
                        indices = torch.LongTensor([0,1])
                        if args.cuda:
                            indices = indices.cuda()
                        zscore = torch.index_select(zscore, 3, indices)
                        zscore_x  = torch.matmul(zscore.unsqueeze(3), normalisedvel.unsqueeze(4))
                        zscore_y = torch.matmul(zscore.unsqueeze(3), velperp.unsqueeze(4))
                        zscorelist_x.append(zscore_x)
                        zscorelist_y.append(zscore_y)
                    # in case of isotropic we need to recast sigma to the same shape as output as it is required in the gaussian function
                    loss_nll, loss_nll_var = 0, 0
                    loss_nll_M, loss_nll_M_var = 0, 0
                    if args.isotropic:
                        sigmaone = tile(sigmaone, 3, list(output.size())[3])
                        loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma(output, target, sigmaone, epoch, args.temp_sigmoid, args.epochs)
                        loss_nll_var = nll_gaussian_var__variablesigma(output, target, sigmaone, epoch, args.temp_sigmoid, args.epochs)
                    else:
                        loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma_semiisotropic(output, target, sigmaone, epoch,
                                                                              args.temp_sigmoid, args.epochs)
                        loss_nll_var = nll_gaussian_var__variablesigma_semiisotropic(output, target, sigmaone, epoch,
                                                                       args.temp_sigmoid, args.epochs)
                    output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, args.prediction_steps)
                    if args.isotropic:
                        loss_nll_M, loss_1_M, loss_2_M = nll_gaussian_variablesigma(output_M, target, sigma_M, epoch, args.temp_sigmoid, args.epochs)
                        loss_nll_M_var = nll_gaussian_var__variablesigma(output_M, target, sigma_M, epoch, args.temp_sigmoid, args.epochs)
                    else:
                        loss_nll_M, loss_1_M, loss_2_M = nll_gaussian_variablesigma_semiisotropic(output_M, target, sigma_M, epoch,
                                                                                    args.temp_sigmoid, args.epochs)
                        loss_nll_M_var = nll_gaussian_var__variablesigma_semiisotropic(output_M, target, sigma_M, epoch,
                                                                         args.temp_sigmoid, args.epochs)
                    perm_val.append(perm)
                    acc_val.append(acc_perm)
                    acc_blocks_val.append(acc_blocks)
                    acc_var_val.append(acc_var)
                    acc_var_blocks_val.append(acc_var_blocks)

                    mse_val.append(F.mse_loss(output_M, target).data.item())
                    nll_val.append(loss_nll.data.item())
                    nll_var_val.append(loss_nll_var.data.item())

                    kl_val.append(loss_kl.data.item())
                    kl_list_val.append([kl_loss.data.item() for kl_loss in loss_kl_split])
                    kl_var_list_val.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                    nll_M_val.append(loss_nll_M.data.item())
                    nll_M_var_val.append(loss_nll_M_var.data.item())
                else:
                    # anisotropic
                    target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                    output, sigmaone, accelone, velone = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 1)

                    if args.plot:
                        import matplotlib.pyplot as plt
                        output_plot, sigma_plot, accel_plot, vel_plot = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 49)

                        if args.NRI:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                             args.edge_types_list)
                        else:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split, relations,
                                                                                              args.edge_types_list)

                        from trajectory_plot import draw_lines_anisotropic
                        from matplotlib.patches import Ellipse
                        for i in range(args.batch_size):
                            fig = plt.figure(figsize=(7, 7))
                            ax = fig.add_axes([0, 0, 1, 1])
                            xmin_t, ymin_t, xmax_t, ymax_t = draw_lines_anisotropic(target, i, sigma_plot.detach().cpu().numpy(),vel_plot, ax,  linestyle=':', alpha=0.6)
                            xmin_o, ymin_o, xmax_o, ymax_o = draw_lines_anisotropic(output_plot.detach().cpu().numpy(), i,sigma_plot.detach().cpu().numpy(),vel_plot, ax, linestyle='-', plot_ellipses=True)
                            # indices_1 = torch.LongTensor([0, 1])
                            # indices_2 = torch.LongTensor([2,3])
                            # indices_3 = torch.LongTensor([0])
                            # if args.cuda:
                            #     indices_1, indices_2, indices_3 = indices_1.cuda(), indices_2.cuda(), indices_3.cuda()
                            # positions = torch.index_select(output_plot, 3, indices_1)
                            # sigma_plot_pos = torch.index_select(sigma_plot, 3, indices_1)
                            #
                            # # plots the uncertainty ellipses for gaussian case.
                            # # iterate through each of the atoms
                            # # need to get the angles of the terms to be plotted:
                            # velnorm = vel_plot.norm(p=2, dim=3, keepdim=True)
                            # normalisedvel = vel_plot.div(velnorm.expand_as(vel_plot))
                            # normalisedvel[torch.isnan(normalisedvel)] = np.power(1 / 2, 1 / 2)
                            # # v||.x is just the first term of the tensor
                            # normalisedvelx = torch.index_select(normalisedvel, 3, indices_3)
                            # # angle of rotation is Theta = acos(v||.x) for normalised v|| and x (need angle in degrees not radians)
                            # angle = torch.acos(normalisedvelx).squeeze() * 180/3.14159
                            # for j in range(positions.size()[1]):
                            #     # get the first timestep component of (x,y) and angles
                            #     ellipses = []
                            #     ellipses.append(
                            #         Ellipse((positions.tolist()[i][j][0][0], positions.tolist()[i][j][0][1]),
                            #                 width=sigma_plot.tolist()[i][j][0][0],
                            #                 height=sigma_plot.tolist()[i][j][0][1], angle=angle.tolist()[i][j][0]))
                            #     # iterate through each of the atoms
                            #         # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
                            #         # keeps track of current plot value
                            #     l = 0
                            #     for k in range(positions.size()[2] - 1):
                            #         deltar = (torch.from_numpy(
                            #             positions.cpu().numpy()[i][j][k + 1]) - torch.from_numpy(
                            #             positions.numpy()[i][j][l])).norm(p=2, dim=0, keepdim=True)
                            #         deltasigma = (torch.from_numpy(sigma_plot_pos.cpu().numpy()[i][j][l])).norm(p=2,
                            #                                                                                    dim=0,
                            #                                                                             keepdim=True)
                            #         if (deltar.item() > 2 * deltasigma.item()):
                            #             # check that it is far away from others
                            #             isfarapart = True
                            #             for m in range(positions.size()[1]):
                            #                for n in range(positions.size()[2]):
                            #                     if (m != j):
                            #                         deltar = (torch.from_numpy(
                            #                             positions.cpu().numpy()[i][m][n]) - torch.from_numpy(
                            #                             positions.numpy()[i][j][k + 1])).norm(p=2, dim=0,
                            #                                                                 keepdim=True)
                            #                         deltasigma = (torch.from_numpy(
                            #                             sigma_plot_pos.cpu().numpy()[i][j][k + 1])).norm(p=2, dim=0,
                            #                                                                              keepdim=True)
                            #                         if (deltar < deltasigma.item()):
                            #                             isfarapart = False
                            #             if isfarapart:
                            #                 ellipses.append(Ellipse(
                            #                     (positions.tolist()[i][j][k + 1][0], positions[i][j][k + 1][1]),
                            #                         width=sigma_plot_pos.tolist()[i][j][k + 1][0],
                            #                         height=sigma_plot_pos.tolist()[i][j][k + 1][0], angle=angle.tolist()[i][j][k+1]))
                            #                 # updates to new r0 : Deltar = r - r0:
                            #                 l = k
                            #
                            #     # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
                            #     # for k in range(positions.size()[2] - 1):
                            #     #     deltar = (torch.from_numpy(positions.cpu().numpy()[i][j][k + 1]) - torch.from_numpy(
                            #     #         positions.cpu().numpy()[i][j][k])).norm(p=2, dim=0, keepdim=True)
                            #     #     deltasigma = (torch.from_numpy(sigma_plot.cpu().numpy()[i][j][k + 1])).norm(p=2, dim=0,
                            #     #                                                                           keepdim=True)
                            #     #     if (deltar.item() > 2 * deltasigma.item()):
                            #     #         ellipses.append(
                            #     #             Ellipse((positions.tolist()[i][j][k + 1][0], positions[i][j][k + 1][1]),
                            #     #                     width=sigma_plot.tolist()[i][j][k + 1][0],
                            #     #                     height=sigma_plot.tolist()[i][j][k + 1][1], angle=angle.tolist()[i][j][k+1]))
                            # # fig1, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
                            #     colour = np.random.rand(3)
                            #     for e in ellipses:
                            #         ax.add_artist(e)
                            #         e.set_clip_box(ax.bbox)
                            #         #  e.set_alpha(0.6)
                            #         e.set_facecolor(colour)
                            ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                            ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
                            # block_names = [ 'springs', 'charges' ]
                            acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                                        for j in range(acc_blocks_batch.shape[1])]
                            acc_text = ', '.join(acc_text)
                            plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
                            # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
                            plt.show()
                            # for z score
                            # make sure we aren't dividing by 0
                        if (torch.min(sigma_plot) < pow(10, -7)):
                            accuracy = np.full((sigma_plot.size(0), sigma_plot.size(1), sigma_plot.size(2), sigma_plot.size(3)), pow(10, -7), dtype=np.float32)
                            accuracy = torch.from_numpy(accuracy)
                            if args.cuda:
                                accuracy = accuracy.cuda()
                            output_plot = torch.max(output_plot, accuracy)
                        zscore = (output_plot - target) / (sigma_plot)
                        velnorm = vel_plot.norm(p=2, dim=3, keepdim=True)
                        normalisedvel = vel_plot.div(velnorm.expand_as(vel_plot))
                        accelnorm = accel_plot.norm(p=2, dim=3, keepdim=True)
                        normalisedaccel = accel_plot.div(accelnorm.expand_as(accel_plot))
                        # get perpendicular components to the accelerations and velocities accelperp, velperp
                        # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
                        rotationmatrix = np.zeros((normalisedvel.size(0), normalisedvel.size(1), normalisedvel.size(2), 2, 2),
                                                  dtype=np.float32)
                        for i in range(len(rotationmatrix)):
                            for j in range(len(rotationmatrix[i])):
                                for l in range(len(rotationmatrix[i][j])):
                                    rotationmatrix[i][j][l][0][1] = np.float32(-1)
                                    rotationmatrix[i][j][l][1][0] = np.float32(1)
                        rotationmatrix = torch.from_numpy(rotationmatrix)
                        if args.cuda:
                            rotationmatrix = rotationmatrix.cuda()
                        velperp = torch.matmul(rotationmatrix, normalisedvel.unsqueeze(4))
                        velperp = velperp.squeeze()
                        accelperp = torch.matmul(rotationmatrix, normalisedaccel.unsqueeze(4))
                        accelperp = accelperp.squeeze()
                        indices = torch.LongTensor([0,1])
                        if args.cuda:
                            indices = indices.cuda()
                        zscore = torch.index_select(zscore, 3, indices)
                        zscore_x = torch.matmul(zscore.unsqueeze(3), normalisedvel.unsqueeze(4))
                        zscore_y = torch.matmul(zscore.unsqueeze(3), velperp.unsqueeze(4))
                        zscorelist_x.append(zscore_x)
                        zscorelist_y.append(zscore_y)

                    loss_nll, loss_1, loss_2 = nll_gaussian_multivariatesigma_withcorrelations(output, target,  sigmaone, accelone, velone)
                    loss_nll_var = nll_gaussian_var_multivariatesigma_withcorrelations(output, target, sigmaone, accelone, velone)

                    output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, args.prediction_steps)
                    loss_nll_M, loss_1_M, loss_2_M = nll_gaussian_multivariatesigma_withcorrelations(output_M, target, sigma_M, accel_M ,vel_M)
                    loss_nll_M_var = nll_gaussian_var_multivariatesigma_withcorrelations(output_M, target, sigma_M, accel_M, vel_M)
                    sigma = sigmaone

                    perm_val.append(perm)
                    acc_val.append(acc_perm)
                    acc_blocks_val.append(acc_blocks)
                    acc_var_val.append(acc_var)
                    acc_var_blocks_val.append(acc_var_blocks)

                    mse_val.append(F.mse_loss(output_M, target).data.item())
                    nll_val.append(loss_nll.data.item())
                    nll_var_val.append(loss_nll_var.data.item())

                    kl_val.append(loss_kl.data.item())
                    kl_list_val.append([kl_loss.data.item() for kl_loss in loss_kl_split])
                    kl_var_list_val.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                    nll_M_val.append(loss_nll_M.data.item())
                    nll_M_var_val.append(loss_nll_M_var.data.item())
    # deal with z-score here
    if (args.plot):
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        zscorelistintx = np.empty((0))
        zscorelistinty = np.empty((0))
        for i in range(len(zscorelist_x)):
            zscorelistintx = np.append(zscorelistintx, zscorelist_x[i].numpy())
            zscorelistinty = np.append(zscorelistinty, zscorelist_y[i].numpy())
        bins = np.arange(-4, 4.1, 0.1)
        # get histogram distribution
        histdatax, bin_edges, patches = plt.hist(zscorelistintx, bins, density = True)

        # take the histdata point to be at the centre of the bin_edges:
        # Gaussian fit- we expect a good model to give mean = 0 and sigma = 1
        xcoords = np.empty(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            xcoords[i] = (bin_edges[i] + bin_edges[i+1]) /2
        numberofpoints = len(xcoords)
        # mean is 1/N SUM(xy)
        mean_gaussian_x = np.sum(xcoords * histdatax) / numberofpoints
        # var = 1/N SUM(y*(x-mean) ** 2)
        sigma_x = np.sqrt(np.sum(histdatax * (xcoords - mean_gaussian_x) ** 2) / numberofpoints)
        optimised_params_x, pcov =  curve_fit(gaussian, xcoords, histdatax, p0 = [1, mean_gaussian_x, sigma_x])
        plt.plot(xcoords, gaussian(xcoords, *optimised_params_x), label =  'fit')
        optimised_params_lor_x, pcov = curve_fit(lorentzian, xcoords, histdatax, p0=[1, mean_gaussian_x, sigma_x])
        plt.plot(xcoords, lorentzian(xcoords, *optimised_params_lor_x), 'k')
        plt.xlabel("z-score")
        plt.ylabel("frequency")
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.xlim(-4, 4)
        plt.show()
        # get histogram distribution
        histdatay, bin_edges, patches = plt.hist(zscorelistinty, bins, density=True)

        # take the histdata point to be at the centre of the bin_edges:
        # Gaussian fit- we expect a good model to give mean = 0 and sigma = 1
        ycoords = np.empty(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            ycoords[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        numberofpoints = len(ycoords)
        # mean is 1/N SUM(xy)
        mean_gaussian_y = np.sum(ycoords * histdatay) / numberofpoints
        # var = 1/N SUM(y*(x-mean) ** 2)
        sigma_y = np.sqrt(np.sum(histdatay * (ycoords - mean_gaussian_y) ** 2) / numberofpoints)
        optimised_params_y, pcov = curve_fit(gaussian, ycoords, histdatay, p0=[1, mean_gaussian_y, sigma_y])
        plt.plot(ycoords, gaussian(ycoords, *optimised_params_y), label='fit')
        optimised_params_lor_y, pcov = curve_fit(lorentzian, ycoords, histdatay, p0=[1, mean_gaussian_y, sigma_y])
        plt.plot(ycoords, lorentzian(ycoords, *optimised_params_lor_y), 'k')
        plt.xlabel("z-score")
        plt.ylabel("frequency")
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.xlim(-4, 4)
        plt.show()

        print("Gaussian Fit parallel to vel with mean: " + str(optimised_params_x[1]) + " and std: " + str(optimised_params_x[2]))
        print("Lorentzian Fit parallel to vel with mean: " + str(optimised_params_lor_x[1]) + " and std: " + str(optimised_params_lor_x[2]))
        print(
            "Gaussian Fit perpendicular to vel with mean: " + str(optimised_params_y[1]) + " and std: " + str(optimised_params_y[2]))
        print("Lorentzian Fit perpendicular to vel with mean: " + str(optimised_params_lor_y[1]) + " and std: " + str(
            optimised_params_lor_y[2]))

    print('Epoch: {:03d}'.format(epoch),
          'perm_val: ' + str(np.around(np.mean(np.array(perm_val), axis=0), 4)),
          'time: {:.1f}s'.format(time.time() - t))
    print('nll_trn: {:.2f}'.format(np.mean(nll_train)),
          'kl_trn: {:.5f}'.format(np.mean(kl_train)),
          'mse_trn: {:.10f}'.format(np.mean(mse_train)),
          'acc_trn: {:.5f}'.format(np.mean(acc_train)),
          'KLb_trn: {:.5f}'.format(np.mean(KLb_train))
          )
    print('acc_b_trn: ' + str(np.around(np.mean(np.array(acc_blocks_train), axis=0), 4)),
          'kl_trn: ' + str(np.around(np.mean(np.array(kl_list_train), axis=0), 4))
          )
    print('nll_val: {:.2f}'.format(np.mean(nll_M_val)),
          'kl_val: {:.5f}'.format(np.mean(kl_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.5f}'.format(np.mean(acc_val)),
          'KLb_val: {:.5f}'.format(np.mean(KLb_val))
          )
    print('acc_b_val: ' + str(np.around(np.mean(np.array(acc_blocks_val), axis=0), 4)),
          'kl_val: ' + str(np.around(np.mean(np.array(kl_list_val), axis=0), 4))
          )
    print('Epoch: {:04d}'.format(epoch),
          'perm_val: ' + str(np.around(np.mean(np.array(perm_val), axis=0), 4)),
          'time: {:.4f}s'.format(time.time() - t),
          file=log)
    print('nll_trn: {:.5f}'.format(np.mean(nll_train)),
          'kl_trn: {:.5f}'.format(np.mean(kl_train)),
          'mse_trn: {:.10f}'.format(np.mean(mse_train)),
          'acc_trn: {:.5f}'.format(np.mean(acc_train)),
          'KLb_trn: {:.5f}'.format(np.mean(KLb_train)),
          'acc_b_trn: ' + str(np.around(np.mean(np.array(acc_blocks_train), axis=0), 4)),
          'kl_trn: ' + str(np.around(np.mean(np.array(kl_list_train), axis=0), 4)),
          file=log)
    print('nll_val: {:.5f}'.format(np.mean(nll_M_val)),
          'kl_val: {:.5f}'.format(np.mean(kl_val)),
          'mse_val: {:.10f}'.format(np.mean(mse_val)),
          'acc_val: {:.5f}'.format(np.mean(acc_val)),
          'KLb_val: {:.5f}'.format(np.mean(KLb_val)),
          'acc_b_val: ' + str(np.around(np.mean(np.array(acc_blocks_val), axis=0), 4)),
          'kl_val: ' + str(np.around(np.mean(np.array(kl_list_val), axis=0), 4)),
          file=log)
    if epoch == 0:
        labels = ['epoch', 'nll trn', 'kl trn', 'mse train', 'KLb trn', 'acc trn']
        labels += ['b' + str(i) + ' acc trn' for i in range(len(args.edge_types_list))] + ['nll var trn']
        labels += ['b' + str(i) + ' kl trn' for i in range(len(kl_list_train[0]))]
        labels += ['b' + str(i) + ' kl var trn' for i in range(len(kl_list_train[0]))]
        labels += ['acc var trn'] + ['b' + str(i) + ' acc var trn' for i in range(len(args.edge_types_list))]
        labels += ['nll val', 'nll_M_val', 'kl val', 'mse val', 'KLb val', 'acc val']
        labels += ['b' + str(i) + ' acc val' for i in range(len(args.edge_types_list))]
        labels += ['nll var val', 'nll_M var val']
        labels += ['b' + str(i) + ' kl val' for i in range(len(kl_list_val[0]))]
        labels += ['b' + str(i) + ' kl var val' for i in range(len(kl_list_val[0]))]
        labels += ['acc var val'] + ['b' + str(i) + ' acc var val' for i in range(len(args.edge_types_list))]
        csv_writer.writerow(labels)

        labels = ['trn ' + str(i) for i in range(len(perm_train[0]))]
        labels += ['val ' + str(i) for i in range(len(perm_val[0]))]
        perm_writer.writerow(labels)

    csv_writer.writerow([epoch, np.mean(nll_train), np.mean(kl_train),
                         np.mean(mse_train), np.mean(KLb_train), np.mean(acc_train)] +
                        list(np.mean(np.array(acc_blocks_train), axis=0)) +
                        [np.mean(nll_var_train)] +
                        list(np.mean(np.array(kl_list_train), axis=0)) +
                        list(np.mean(np.array(kl_var_list_train), axis=0)) +
                        # list(np.mean(np.array(KLb_blocks_train),axis=0)) +
                        [np.mean(acc_var_train)] + list(np.mean(np.array(acc_var_blocks_train), axis=0)) +
                        [np.mean(nll_val), np.mean(nll_M_val), np.mean(kl_val), np.mean(mse_val),
                         np.mean(KLb_val), np.mean(acc_val)] +
                        list(np.mean(np.array(acc_blocks_val), axis=0)) +
                        [np.mean(nll_var_val), np.mean(nll_M_var_val)] +
                        list(np.mean(np.array(kl_list_val), axis=0)) +
                        list(np.mean(np.array(kl_var_list_val), axis=0)) +
                        # list(np.mean(np.array(KLb_blocks_val),axis=0))
                        [np.mean(acc_var_val)] + list(np.mean(np.array(acc_var_blocks_val), axis=0))
                        )
    perm_writer.writerow(list(np.mean(np.array(perm_train), axis=0)) +
                         list(np.mean(np.array(perm_val), axis=0))
                         )

    log.flush()
    if args.save_folder and np.mean(nll_M_val) < best_val_loss:
        torch.save(encoder.state_dict(), encoder_file)
        torch.save(decoder.state_dict(), decoder_file)
        print('Best model so far, saving...')
    # save model in different folder even if not the best model. This is a temporary fix for BUS errors- not ideal but
    # the best we can do..
    if args.save_folder:
        torch.save(encoder.state_dict(), current_encoder_file)
        torch.save(decoder.state_dict(), current_decoder_file)
    return np.mean(acc_val), np.mean(nll_M_val), np.around(np.mean(np.array(acc_blocks_val), axis=0), 4)


def test():
    t = time.time()
    nll_test = []
    nll_var_test = []

    mse_1_test = []
    mse_10_test = []
    mse_20_test = []

    kl_test = []
    kl_list_test = []
    kl_var_list_test = []

    acc_test = []
    acc_var_test = []
    acc_blocks_test = []
    acc_var_blocks_test = []
    perm_test = []

    KLb_test = []
    KLb_blocks_test = []  # KL between blocks list

    nll_M_test = []
    nll_M_var_test = []

    encoder.eval()
    decoder.eval()
    if not args.cuda:
        encoder.load_state_dict(torch.load(encoder_file, map_location='cpu'))
        decoder.load_state_dict(torch.load(decoder_file, map_location='cpu'))
    else:
        encoder.load_state_dict(torch.load(encoder_file))
        decoder.load_state_dict(torch.load(decoder_file))

    for batch_idx, (data, relations) in enumerate(test_loader):
        with torch.no_grad():
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()

            assert (data.size(2) - args.timesteps) >= args.timesteps
            data_encoder = data[:, :, :args.timesteps, :].contiguous()
            data_decoder = data[:, :, -args.timesteps:, :].contiguous()

            # stores the values of the uncertainty. This will be an array of size [batchsize, no. of particles, time,no. of axes (isotropic = 1, anisotropic = 2)]
            # initialise sigma to an array of large negative numbers which become small positive numbers when passted through softplus function.
            sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms, inversesoftplus(pow(args.var,1/2), args.temp_softplus), ani_dims=8)
            if args.cuda:
                sigma = sigma.cuda()
            if args.semi_isotropic:
                sigma = tile(sigma, 3, 2)
            # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
            logits = encoder(data_encoder, rel_rec, rel_send)

            if args.NRI:
                edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
                prob = my_softmax(logits, -1)

                loss_kl = kl_categorical_uniform(prob, args.num_atoms, edge_types)
                loss_kl_split = [loss_kl]
                loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, edge_types)]

                KLb_test.append(0)
                KLb_blocks_test.append([0])

                acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_NRI(logits, relations, args.edge_types_list)

            else:
                logits_split = torch.split(logits, args.edge_types_list, dim=-1)
                edges_split = tuple([gumbel_softmax(logits_i, tau=args.temp, hard=args.hard) for logits_i in logits_split])
                edges = torch.cat(edges_split, dim=-1)
                prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split]

                if args.prior:
                    loss_kl_split = [kl_categorical(prob_split[type_idx], log_prior[type_idx],
                                                    args.num_atoms) for type_idx in range(len(args.edge_types_list))]
                    loss_kl = sum(loss_kl_split)
                else:
                    loss_kl_split = [kl_categorical_uniform(prob_split[type_idx], args.num_atoms,
                                                            args.edge_types_list[type_idx])
                                     for type_idx in range(len(args.edge_types_list))]
                    loss_kl = sum(loss_kl_split)

                    loss_kl_var_split = [kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms,
                                                                    args.edge_types_list[type_idx])
                                         for type_idx in range(len(args.edge_types_list))]

                acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_fNRI(logits_split, relations,
                                                                                              args.edge_types_list, args.skip_first)

                KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                KLb_test.append(sum(KLb_blocks).data.item())
                KLb_blocks_test.append([KL.data.item() for KL in KLb_blocks])
                if args.fixed_var:
                    target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                    output, sigma, accel, vel = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, 1)

                    if args.plot:
                        import matplotlib.pyplot as plt
                        output_plot, sigma_plot, accel_plot, vel_plot = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, 49)

                        output_plot_en, sigma_plot_en, accel_en, vel_en = decoder(data_encoder, edges, rel_rec, rel_send, sigma, False, False,args.temp_softplus, 49)
                        from trajectory_plot import draw_lines

                        if args.NRI:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                             args.edge_types_list)
                        else:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split, relations,
                                                                                              args.edge_types_list)

                        for i in range(args.batch_size):
                            fig = plt.figure(figsize=(7, 7))
                            ax = fig.add_axes([0, 0, 1, 1])
                            xmin_t, ymin_t, xmax_t, ymax_t = draw_lines(target, i, linestyle=':', alpha=0.6)
                            xmin_o, ymin_o, xmax_o, ymax_o = draw_lines(output_plot.detach().cpu().numpy(), i, linestyle='-')

                            ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                            ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                            ax.set_xticks([])
                            ax.set_yticks([])
                            block_names = [str(j) for j in range(len(args.edge_types_list))]
                            acc_text = ['layer ' + block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                                        for j in range(acc_blocks_batch.shape[1])]
                            acc_text = ', '.join(acc_text)
                            plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
                            plt.show()

                    loss_nll = nll_gaussian(output, target, args.var)  # compute the reconstruction loss. nll_gaussian is from utils.py
                    loss_nll_var = nll_gaussian_var(output, target, args.var)

                    output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, args.prediction_steps)
                    loss_nll_M = nll_gaussian(output_M, target, args.var)
                    loss_nll_M_var = nll_gaussian_var(output_M, target, args.var)

                    perm_test.append(perm)
                    acc_test.append(acc_perm)
                    acc_blocks_test.append(acc_blocks)
                    acc_var_test.append(acc_var)
                    acc_var_blocks_test.append(acc_var_blocks)

                    output_10, sigma_10, accel_10, vel_10 = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, 10)
                    output_20, sigma_20, accel_20, vel_20 = decoder(data_decoder, edges, rel_rec, rel_send, sigma, False, False, args.temp_softplus, 20)
                    mse_1_test.append(F.mse_loss(output, target).data.item())
                    mse_10_test.append(F.mse_loss(output_10, target).data.item())
                    mse_20_test.append(F.mse_loss(output_20, target).data.item())

                    nll_test.append(loss_nll.data.item())
                    kl_test.append(loss_kl.data.item())
                    kl_list_test.append([kl_loss.data.item() for kl_loss in loss_kl_split])

                    nll_var_test.append(loss_nll_var.data.item())
                    kl_var_list_test.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                    nll_M_test.append(loss_nll_M.data.item())
                    nll_M_var_test.append(loss_nll_M_var.data.item())
                else:
                    if args.anisotropic:
                        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                        output, sigmaone, accelone, velone = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 1)

                        if args.plot:
                            import matplotlib.pyplot as plt
                            output_plot, sigma_plot, accel_plot, vel_plot = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 49)

                            output_plot_en, sigma_plot_en, accel_plot_en, vel_plot_en = decoder(data_encoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 49)
                            from trajectory_plot import draw_lines

                            if args.NRI:
                                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                                 args.edge_types_list)
                            else:
                                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split, relations,
                                                                                                  args.edge_types_list)

                            from trajectory_plot import draw_lines_anisotropic
                            from matplotlib.patches import Ellipse
                            for i in range(args.batch_size):
                                fig = plt.figure(figsize=(7, 7))
                                ax = fig.add_axes([0, 0, 1, 1])
                                xmin_t, ymin_t, xmax_t, ymax_t = draw_lines_anisotropic(target, i, sigma_plot.detach().cpu().numpy(), vel_plot, ax, linestyle=':', alpha=0.6)
                                xmin_o, ymin_o, xmax_o, ymax_o = draw_lines_anisotropic(output_plot.detach().cpu().numpy(), i, sigma_plot.detach().cpu().numpy(), vel_plot, ax,
                                                                            linestyle='-', plot_ellipses=True)
                                # indices_1 = torch.LongTensor([0, 1])
                                # indices_2 = torch.LongTensor([2, 3])
                                # indices_3 = torch.LongTensor([0])
                                # if args.cuda:
                                #     indices_1, indices_2, indices_3 = indices_1.cuda(), indices_2.cuda(), indices_3.cuda()
                                # positions = torch.index_select(output_plot, 3, indices_1)
                                # sigma_plot_pos = torch.index_select(sigma_plot, 3, indices_1)
                                #
                                # # plots the uncertainty ellipses for gaussian case.
                                # # iterate through each of the atoms
                                # # need to get the angles of the terms to be plotted:
                                # velnorm = vel_plot.norm(p=2, dim=3, keepdim=True)
                                # normalisedvel = vel_plot.div(velnorm.expand_as(vel_plot))
                                # normalisedvel[torch.isnan(normalisedvel)] = np.power(1 / 2, 1 / 2)
                                # # v||.x is just the first term of the tensor
                                # normalisedvelx = torch.index_select(normalisedvel, 3, indices_3)
                                # # angle of rotation is Theta = acos(v||.x) for normalised v|| and x (need angle in degrees not radians)
                                # angle = torch.acos(normalisedvelx).squeeze() * 180 / 3.14159
                                # for j in range(positions.size()[1]):
                                #     # get the first timestep component of (x,y) and angles
                                #     ellipses = []
                                #     ellipses.append(
                                #         Ellipse((positions.tolist()[i][j][0][0], positions.tolist()[i][j][0][1]),
                                #                 width=sigma_plot.tolist()[i][j][0][0],
                                #                 height=sigma_plot.tolist()[i][j][0][1], angle=angle.tolist()[i][j][0]))
                                #     # iterate through each of the atoms
                                #     # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
                                #     # keeps track of current plot value
                                #     l = 0
                                #     for k in range(positions.size()[2] - 1):
                                #         deltar = (torch.from_numpy(
                                #             positions.cpu().numpy()[i][j][k + 1]) - torch.from_numpy(
                                #             positions.numpy()[i][j][l])).norm(p=2, dim=0, keepdim=True)
                                #         deltasigma = (torch.from_numpy(sigma_plot_pos.cpu().numpy()[i][j][l])).norm(p=2,
                                #                                                                                     dim=0,
                                #                                                                                     keepdim=True)
                                #         if (deltar.item() > 2 * deltasigma.item()):
                                #             # check that it is far away from others
                                #             isfarapart = True
                                #             for m in range(positions.size()[1]):
                                #                 for n in range(positions.size()[2]):
                                #                     if (m != j):
                                #                         deltar = (torch.from_numpy(
                                #                             positions.cpu().numpy()[i][m][n]) - torch.from_numpy(
                                #                             positions.numpy()[i][j][k + 1])).norm(p=2, dim=0,
                                #                                                                   keepdim=True)
                                #                         deltasigma = (torch.from_numpy(
                                #                             sigma_plot_pos.cpu().numpy()[i][j][k + 1])).norm(p=2, dim=0,
                                #                                                                              keepdim=True)
                                #                         if (deltar < deltasigma.item()):
                                #                             isfarapart = False
                                #             if isfarapart:
                                #                 ellipses.append(Ellipse(
                                #                     (positions.tolist()[i][j][k + 1][0], positions[i][j][k + 1][1]),
                                #                     width=sigma_plot_pos.tolist()[i][j][k + 1][0],
                                #                     height=sigma_plot_pos.tolist()[i][j][k + 1][0],
                                #                     angle=angle.tolist()[i][j][k + 1]))
                                #                 # updates to new r0 : Deltar = r - r0:
                                #                 l = k
                                #
                                #     # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
                                #     # for k in range(positions.size()[2] - 1):
                                #     #     deltar = (torch.from_numpy(positions.cpu().numpy()[i][j][k + 1]) - torch.from_numpy(
                                #     #         positions.cpu().numpy()[i][j][k])).norm(p=2, dim=0, keepdim=True)
                                #     #     deltasigma = (torch.from_numpy(sigma_plot.cpu().numpy()[i][j][k + 1])).norm(p=2, dim=0,
                                #     #                                                                           keepdim=True)
                                #     #     if (deltar.item() > 2 * deltasigma.item()):
                                #     #         ellipses.append(
                                #     #             Ellipse((positions.tolist()[i][j][k + 1][0], positions[i][j][k + 1][1]),
                                #     #                     width=sigma_plot.tolist()[i][j][k + 1][0],
                                #     #                     height=sigma_plot.tolist()[i][j][k + 1][1], angle=angle.tolist()[i][j][k+1]))
                                #     # fig1, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
                                #     colour = np.random.rand(3)
                                #     for e in ellipses:
                                #         ax.add_artist(e)
                                #         e.set_clip_box(ax.bbox)
                                #         #  e.set_alpha(0.6)
                                #         e.set_facecolor(colour)
                                ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                                ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                                ax.set_xticks([])
                                ax.set_yticks([])
                                block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
                                # block_names = [ 'springs', 'charges' ]
                                acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                                            for j in range(acc_blocks_batch.shape[1])]
                                acc_text = ', '.join(acc_text)
                                plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
                                # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
                                plt.show()

                        loss_nll, loss_1, loss_2 = nll_gaussian_multivariatesigma_withcorrelations(output, target, sigmaone, accelone, velone)  # compute the reconstruction loss. nll_gaussian is from utils.py
                        loss_nll_var = nll_gaussian_var_multivariatesigma_withcorrelations(output, target, sigmaone, accelone, velone)

                        output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, args.prediction_steps)
                        loss_nll_M, loss_1_M, loss_2_M = nll_gaussian_multivariatesigma_withcorrelations(output_M, target, sigma_M, accel_M, vel_M)
                        loss_nll_M_var = nll_gaussian_var_multivariatesigma_withcorrelations(output_M, target, sigma_M, accel_M, vel_M)

                        perm_test.append(perm)
                        acc_test.append(acc_perm)
                        acc_blocks_test.append(acc_blocks)
                        acc_var_test.append(acc_var)
                        acc_var_blocks_test.append(acc_var_blocks)

                        output_10, sigma_10, accel_10, vel_10 = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 10)
                        output_20, sigma_20, accel_20, vel_20 = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, True, args.temp_softplus, 20)
                        mse_1_test.append(F.mse_loss(output, target).data.item())
                        mse_10_test.append(F.mse_loss(output_10, target).data.item())
                        mse_20_test.append(F.mse_loss(output_20, target).data.item())

                        nll_test.append(loss_nll.data.item())
                        kl_test.append(loss_kl.data.item())
                        kl_list_test.append([kl_loss.data.item() for kl_loss in loss_kl_split])

                        nll_var_test.append(loss_nll_var.data.item())
                        kl_var_list_test.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                        nll_M_test.append(loss_nll_M.data.item())
                        nll_M_var_test.append(loss_nll_M_var.data.item())
                        sigma = sigmaone
                    else:
                        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                        output, sigmaone, accelone, velone = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 1)

                        if args.plot:
                            import matplotlib.pyplot as plt
                            output_plot, sigma_plot, accel_plot, vel_plot = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 49)
                            if args.isotropic:
                                sigma_plot = tile(sigma_plot, 3, list(output_plot.size())[3])
                            else:
                                indices_pos = torch.LongTensor([0])
                                indices_vel = torch.LongTensor([1])
                                if args.cuda:
                                    indices_pos = indices_pos.cuda()
                                sigma_plot_pos = torch.index_select(sigma_plot, 3, indices_pos)
                                sigma_plot_vel = torch.index_select(sigma_plot, 3, indices_vel)
                                sigma_plot_pos = tile(sigma_plot_pos, 3, 2)
                                sigma_plot_vel = tile(sigma_plot_vel, 3, 2)
                                sigma_plot = torch.cat((sigma_plot_pos, sigma_plot_vel), 3)
                            output_plot_en, sigma_plot_en, accel_plot_en, vel_en = decoder(data_encoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 49)
                            from trajectory_plot import draw_lines

                            if args.NRI:
                                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                                 args.edge_types_list)
                            else:
                                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split, relations,
                                                                                                  args.edge_types_list)

                            from trajectory_plot import draw_lines_sigma
                            from matplotlib.patches import Ellipse
                            for i in range(args.batch_size):
                                fig = plt.figure(figsize=(7, 7))
                                ax = fig.add_axes([0, 0, 1, 1])
                                xmin_t, ymin_t, xmax_t, ymax_t = draw_lines_sigma(target, i, sigma_plot.detach().cpu().numpy(), ax, linestyle=':', alpha=0.6, plot_ellipses= False)
                                xmin_o, ymin_o, xmax_o, ymax_o = draw_lines_sigma(output_plot.detach().cpu().numpy(), i,sigma_plot.detach().cpu().numpy(), ax,
                                                                            linestyle='-', plot_ellipses=True)
                                # isotropic therefore the ellipses become circles
                                # indices = torch.LongTensor([0, 1])
                                # if args.cuda:
                                #     indices = indices.cuda()
                                # positions = torch.index_select(output_plot, 3, indices)
                                # ellipses = []
                                # # iterate through each of the atoms
                                # for j in range(positions.size()[1]):
                                #     # get the first timestep component of (x,y)
                                #     ellipses.append(
                                #         Ellipse((positions.tolist()[i][j][0][0], positions.tolist()[i][j][0][1]),
                                #                 width=sigma_plot.tolist()[i][j][0][0],
                                #                 height=sigma_plot.tolist()[i][j][0][0], angle=0.0))
                                #     # if Deltax^2+Deltay^2>4*(DeltaSigmax^2+DeltaSigma^2) then plot, else do not plot
                                #     for k in range(positions.size()[2] - 1):
                                #         deltar = (torch.from_numpy(positions.cpu().numpy()[i][j][k + 1]) - torch.from_numpy(
                                #             positions.cpu().numpy()[i][j][k])).norm(p=2, dim=0, keepdim=True)
                                #         deltasigma = (torch.from_numpy(sigma_plot.cpu().numpy()[i][j][k + 1])).norm(p=2,
                                #                                                                               dim=0,
                                #                                                                               keepdim=True)
                                #         if (deltar.item() > 2 * deltasigma.item()):
                                #             ellipses.append(
                                #                 Ellipse((positions.tolist()[i][j][k + 1][0], positions[i][j][k + 1][1]),
                                #                         width=sigma_plot.tolist()[i][j][k + 1][0],
                                #                         height=sigma_plot.tolist()[i][j][k + 1][0], angle=0.0))
                                # fig1, ax1 = plt.subplots(subplot_kw={'aspect': 'equal'})
                                # for e in ellipses:
                                #     ax1.add_artist(e)
                                #     e.set_clip_box(ax1.bbox)
                                #     e.set_alpha(0.6)
                                ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
                                ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
                                ax.set_xticks([])
                                ax.set_yticks([])
                                block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
                                # block_names = [ 'springs', 'charges' ]
                                acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                                            for j in range(acc_blocks_batch.shape[1])]

                                acc_text = ', '.join(acc_text)
                                plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
                                # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
                                plt.show()

                        # in case of isotropic we need to recast sigma to the same shape as output as it is required in the gaussian function
                        if args.isotropic:
                            sigmaone = tile(sigmaone, 3, list(output.size())[3])
                            loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma(output, target, sigmaone, args.epochs, args.temp_sigmoid, args.epochs)  # compute the reconstruction loss. nll_gaussian is from utils.py
                            loss_nll_var = nll_gaussian_var__variablesigma(output, target, sigmaone, args.epochs, args.temp_sigmoid, args.epochs)
                            output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, args.prediction_steps)
                            loss_nll_M, loss_1_M, loss_2_M = nll_gaussian_variablesigma(output_M, target, sigma_M, args.epochs, args.temp_sigmoid, args.epochs)
                            loss_nll_M_var = nll_gaussian_var__variablesigma(output_M, target, sigma_M, args.epochs, args.temp_sigmoid, args.epochs)
                        else:
                            loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma_semiisotropic(output, target, sigmaone, args.epochs,
                                                                                  args.temp_sigmoid,
                                                                                  args.epochs)  # compute the reconstruction loss. nll_gaussian is from utils.py
                            loss_nll_var = nll_gaussian_var__variablesigma_semiisotropic(output, target, sigmaone, args.epochs,
                                                                           args.temp_sigmoid, args.epochs)
                            output_M, sigma_M, accel_M, vel_M = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True,
                                                                 False, args.temp_softplus, args.prediction_steps)
                            loss_nll_M, loss_1_M, loss_2_M = nll_gaussian_variablesigma_semiisotropic(output_M, target, sigma_M,
                                                                                        args.epochs, args.temp_sigmoid,
                                                                                        args.epochs)
                            loss_nll_M_var = nll_gaussian_var__variablesigma_semiisotropic(output_M, target, sigma_M, args.epochs,
                                                                             args.temp_sigmoid, args.epochs)

                        perm_test.append(perm)
                        acc_test.append(acc_perm)
                        acc_blocks_test.append(acc_blocks)
                        acc_var_test.append(acc_var)
                        acc_var_blocks_test.append(acc_var_blocks)

                        output_10, sigma_10, accel_10, vel_10 = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 10)
                        output_20, sigma_20, accel_20, vel_20 = decoder(data_decoder, edges, rel_rec, rel_send, sigma, True, False, args.temp_softplus, 20)
                        mse_1_test.append(F.mse_loss(output, target).data.item())
                        mse_10_test.append(F.mse_loss(output_10, target).data.item())
                        mse_20_test.append(F.mse_loss(output_20, target).data.item())

                        nll_test.append(loss_nll.data.item())
                        kl_test.append(loss_kl.data.item())
                        kl_list_test.append([kl_loss.data.item() for kl_loss in loss_kl_split])

                        nll_var_test.append(loss_nll_var.data.item())
                        kl_var_list_test.append([kl_var.data.item() for kl_var in loss_kl_var_split])

                        nll_M_test.append(loss_nll_M.data.item())
                        nll_M_var_test.append(loss_nll_M_var.data.item())
                        sigma = sigmaone

    print('--------------------------------')
    print('------------Testing-------------')
    print('--------------------------------')
    print('nll_test: {:.2f}'.format(np.mean(nll_test)),
          'nll_M_test: {:.2f}'.format(np.mean(nll_M_test)),
          'kl_test: {:.5f}'.format(np.mean(kl_test)),
          'mse_1_test: {:.10f}'.format(np.mean(mse_1_test)),
          'mse_10_test: {:.10f}'.format(np.mean(mse_10_test)),
          'mse_20_test: {:.10f}'.format(np.mean(mse_20_test)),
          'acc_test: {:.5f}'.format(np.mean(acc_test)),
          'acc_var_test: {:.5f}'.format(np.mean(acc_var_test)),
          'KLb_test: {:.5f}'.format(np.mean(KLb_test)),
          'time: {:.1f}s'.format(time.time() - t))
    print('acc_b_test: ' + str(np.around(np.mean(np.array(acc_blocks_test), axis=0), 4)),
          'acc_var_b: ' + str(np.around(np.mean(np.array(acc_var_blocks_test), axis=0), 4)),
          'kl_test: ' + str(np.around(np.mean(np.array(kl_list_test), axis=0), 4))
          )
    if args.save_folder:
        print('--------------------------------', file=log)
        print('------------Testing-------------', file=log)
        print('--------------------------------', file=log)
        print('nll_test: {:.2f}'.format(np.mean(nll_test)),
              'nll_M_test: {:.2f}'.format(np.mean(nll_M_test)),
              'kl_test: {:.5f}'.format(np.mean(kl_test)),
              'mse_1_test: {:.10f}'.format(np.mean(mse_1_test)),
              'mse_10_test: {:.10f}'.format(np.mean(mse_10_test)),
              'mse_20_test: {:.10f}'.format(np.mean(mse_20_test)),
              'acc_test: {:.5f}'.format(np.mean(acc_test)),
              'acc_var_test: {:.5f}'.format(np.mean(acc_var_test)),
              'KLb_test: {:.5f}'.format(np.mean(KLb_test)),
              'time: {:.1f}s'.format(time.time() - t),
              file=log)
        print('acc_b_test: ' + str(np.around(np.mean(np.array(acc_blocks_test), axis=0), 4)),
              'acc_var_b_test: ' + str(np.around(np.mean(np.array(acc_var_blocks_test), axis=0), 4)),
              'kl_test: ' + str(np.around(np.mean(np.array(kl_list_test), axis=0), 4)),
              file=log)
        log.flush()


# Train model
if not args.test:
    t_total = time.time()
    best_val_loss = np.inf
    best_epoch = 0
    loss_over_epochs = []
    accofencoder_over_epochs = []
    accoflayer1_over_epochs = []
    accoflayer2_over_epochs = []

    labels = ['epoch', 'loss_over_epochs', 'accofencoder', 'accoflayer1', 'accoflayer2']
    for epoch in range(args.epochs):
        acc_val, val_loss, acc_perlayer = train(epoch, best_val_loss)
        loss_over_epochs.append(val_loss)
        accofencoder_over_epochs.append(acc_val * 100)
        accoflayer1_over_epochs.append(acc_perlayer[0] * 100)
        accoflayer2_over_epochs.append(acc_perlayer[1] * 100)
        print('epoch: ' + str(epoch), 'loss_value: ' + str(val_loss), 'accofencoder: ' + str(acc_val), 'accoflayer1: '
            + str(acc_perlayer[0]), 'accoflayer2: ' + str(acc_perlayer[1]), file=loss_data)
        loss_data.flush()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        if epoch - best_epoch > args.patience and epoch > 99:
            break
    import matplotlib.pyplot as plt
    epochs = np.arange(0, len(loss_over_epochs))
    fig, ax = plt.subplots(nrows= 1, ncols= 1)
    ax.plot(epochs, loss_over_epochs, label='loss 1')
    ax.set_xlabel('epoch')
    ax.set_ylabel('Total Loss')
    if args.save_folder:
        fig.savefig(os.path.join(save_folder, 'lossoverepochs.png'))
    plt.close(fig)
    fig, ax = plt.subplots(nrows= 1, ncols= 1)
    ax.plot(epochs, accofencoder_over_epochs, 'g')
    ax.plot(epochs, accoflayer1_over_epochs, 'b')
    ax.plot(epochs, accoflayer2_over_epochs, 'k')
    ax.set_xlabel('epoch')
    ax.set_ylabel('encoder accuracy/ %')
    ax.legend(['Overall accuracy', 'accuracy of layer 1', 'accuracy of layer 2'])
    if args.save_folder:
        fig.savefig(os.path.join(save_folder, 'accuracyoverepochs.png'))
    plt.close(fig)
    print("Optimization Finished!")
    print("Best Epoch: {:04d}".format(best_epoch))
    if args.save_folder:
        print("Best Epoch: {:04d}".format(best_epoch), file=log)
        log.flush()

test()
if log is not None:
    print(save_folder)
    log.close()
    log_csv.close()
    perm_csv.close()
    loss_data.close()