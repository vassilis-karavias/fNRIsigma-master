"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""

import numpy as np
import torch
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable

from itertools import permutations, chain
from math import factorial
import time

from os import path

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input, dim=0) # added dim=0 as implicit choice is deprecated, dim 0 is edgetype due to transpose
    return soft_max_1d.transpose(axis, 0)


def binary_concrete(logits, tau=1, hard=False, eps=1e-10):
    y_soft = binary_concrete_sample(logits, tau=tau, eps=eps)
    if hard:
        y_hard = (y_soft > 0.5).float()
        y = Variable(y_hard.data - y_soft.data) + y_soft
    else:
        y = y_soft
    return y


def binary_concrete_sample(logits, tau=1, eps=1e-10):
    logistic_noise = sample_logistic(logits.size(), eps=eps)
    if logits.is_cuda:
        logistic_noise = logistic_noise.cuda()
    y = logits + Variable(logistic_noise)
    return F.sigmoid(y / tau)


def sample_logistic(shape, eps=1e-10):
    uniform = torch.rand(shape).float()
    return torch.log(uniform + eps) - torch.log(1 - uniform + eps)


def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def my_sigmoid(logits, hard=True, sharpness=1.0):

    edges_soft = 1/(1+torch.exp(-sharpness*logits))
    if hard:
        edges_hard = torch.round(edges_soft)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        if edges_soft.is_cuda:
            edges_hard = edges_hard.cuda()
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        edges = Variable(edges_hard - edges_soft.data) + edges_soft
    else:
        edges = edges_soft
    return edges

def binary_accuracy(output, labels):
    preds = output > 0.5
    correct = preds.type_as(labels).eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def edge_type_encode(edges): # this is used to gives each 'interaction strength' a unique integer = 0, 1, 2 ..
    unique = np.unique(edges)
    encode = np.zeros(edges.shape)
    for i in range(unique.shape[0]):
        encode += np.where( edges == unique[i], i, 0)
    return encode

def loader_edges_encode(edges, num_atoms): 
    edges = np.reshape(edges, [edges.shape[0], edges.shape[1], num_atoms ** 2])
    edges = np.array(edge_type_encode(edges), dtype=np.int64)
    off_diag_idx = np.ravel_multi_index(
        np.where(np.ones((num_atoms, num_atoms)) - np.eye(num_atoms)),
        [num_atoms, num_atoms])
    edges = edges[:,:, off_diag_idx]
    return edges

def loader_combine_edges(edges):
    edge_types_list = [ int(np.max(edges[:,i,:]))+1 for i in range(edges.shape[1]) ]
    assert( edge_types_list == sorted(edge_types_list)[::-1] )
    encoded_target = np.zeros( edges[:,0,:].shape )
    base = 1
    for i in reversed(range(edges.shape[1])):
        encoded_target += base*edges[:,i,:]
        base *= edge_types_list[i]
    return encoded_target.astype('int')

def load_data_NRI(batch_size=1, sim_folder='', shuffle=True, data_folder='data'):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder,sim_folder,'loc_train.npy'))
    vel_train = np.load(path.join(data_folder,sim_folder,'vel_train.npy'))
    edges_train = np.load(path.join(data_folder,sim_folder,'edges_train.npy'))

    loc_valid = np.load(path.join(data_folder,sim_folder,'loc_valid.npy'))
    vel_valid = np.load(path.join(data_folder,sim_folder,'vel_valid.npy'))
    edges_valid = np.load(path.join(data_folder,sim_folder,'edges_valid.npy'))

    loc_test = np.load(path.join(data_folder,sim_folder,'loc_test.npy'))
    vel_test = np.load(path.join(data_folder,sim_folder,'vel_test.npy'))
    edges_test = np.load(path.join(data_folder,sim_folder,'edges_test.npy'))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    edges_train = loader_edges_encode(edges_train, num_atoms)
    edges_valid = loader_edges_encode(edges_valid, num_atoms)
    edges_test = loader_edges_encode(edges_test, num_atoms)

    edges_train = loader_combine_edges(edges_train)
    edges_valid = loader_combine_edges(edges_valid)
    edges_test = loader_combine_edges(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    edges_train = torch.LongTensor(edges_train)
    feat_valid = torch.FloatTensor(feat_valid)
    edges_valid = torch.LongTensor(edges_valid)
    feat_test = torch.FloatTensor(feat_test)
    edges_test = torch.LongTensor(edges_test)

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def load_data_fNRI(batch_size=1, sim_folder='', shuffle=True, data_folder='data'):
    # the edges numpy arrays below are [ num_sims, N, N ]
    loc_train = np.load(path.join(data_folder,sim_folder,'loc_train.npy'))
    vel_train = np.load(path.join(data_folder,sim_folder,'vel_train.npy'))
    edges_train = np.load(path.join(data_folder,sim_folder,'edges_train.npy'))

    loc_valid = np.load(path.join(data_folder,sim_folder,'loc_valid.npy'))
    vel_valid = np.load(path.join(data_folder,sim_folder,'vel_valid.npy'))
    edges_valid = np.load(path.join(data_folder,sim_folder,'edges_valid.npy'))

    loc_test = np.load(path.join(data_folder,sim_folder,'loc_test.npy'))
    vel_test = np.load(path.join(data_folder,sim_folder,'vel_test.npy'))
    edges_test = np.load(path.join(data_folder,sim_folder,'edges_test.npy'))

    # [num_samples, num_timesteps, num_dims, num_atoms]
    num_atoms = loc_train.shape[3]

    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    loc_train = (loc_train - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_train = (vel_train - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_valid = (loc_valid - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_valid = (vel_valid - vel_min) * 2 / (vel_max - vel_min) - 1

    loc_test = (loc_test - loc_min) * 2 / (loc_max - loc_min) - 1
    vel_test = (vel_test - vel_min) * 2 / (vel_max - vel_min) - 1

    # Reshape to: [num_sims, num_atoms, num_timesteps, num_dims]
    loc_train = np.transpose(loc_train, [0, 3, 1, 2])
    vel_train = np.transpose(vel_train, [0, 3, 1, 2])
    feat_train = np.concatenate([loc_train, vel_train], axis=3)

    loc_valid = np.transpose(loc_valid, [0, 3, 1, 2])
    vel_valid = np.transpose(vel_valid, [0, 3, 1, 2])
    feat_valid = np.concatenate([loc_valid, vel_valid], axis=3)

    loc_test = np.transpose(loc_test, [0, 3, 1, 2])
    vel_test = np.transpose(vel_test, [0, 3, 1, 2])
    feat_test = np.concatenate([loc_test, vel_test], axis=3)

    edges_train = loader_edges_encode( edges_train, num_atoms )
    edges_valid = loader_edges_encode( edges_valid, num_atoms )
    edges_test = loader_edges_encode( edges_test, num_atoms )

    edges_train = torch.LongTensor(edges_train)
    edges_valid = torch.LongTensor(edges_valid)
    edges_test =  torch.LongTensor(edges_test)

    feat_train = torch.FloatTensor(feat_train)
    feat_valid = torch.FloatTensor(feat_valid)
    feat_test = torch.FloatTensor(feat_test)

    train_data = TensorDataset(feat_train, edges_train)
    valid_data = TensorDataset(feat_valid, edges_valid)
    test_data = TensorDataset(feat_test, edges_test)

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    valid_data_loader = DataLoader(valid_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    return train_data_loader, valid_data_loader, test_data_loader, loc_max, loc_min, vel_max, vel_min


def to_2d_idx(idx, num_cols):
    idx = np.array(idx, dtype=np.int64)
    y_idx = np.array(np.floor(idx / float(num_cols)), dtype=np.int64)
    x_idx = idx % num_cols
    return x_idx, y_idx


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def get_triu_indices(num_nodes):
    """Linear triu (upper triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    triu_indices = (ones.triu() - eye).nonzero().t()
    triu_indices = triu_indices[0] * num_nodes + triu_indices[1]
    return triu_indices


def get_tril_indices(num_nodes):
    """Linear tril (lower triangular) indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    tril_indices = (ones.tril() - eye).nonzero().t()
    tril_indices = tril_indices[0] * num_nodes + tril_indices[1]
    return tril_indices


def get_offdiag_indices(num_nodes):
    """Linear off-diagonal indices."""
    ones = torch.ones(num_nodes, num_nodes)
    eye = torch.eye(num_nodes, num_nodes)
    offdiag_indices = (ones - eye).nonzero().t()
    offdiag_indices = offdiag_indices[0] * num_nodes + offdiag_indices[1]
    return offdiag_indices


def get_triu_offdiag_indices(num_nodes):
    """Linear triu (upper) indices w.r.t. vector of off-diagonal elements."""
    triu_idx = torch.zeros(num_nodes * num_nodes)
    triu_idx[get_triu_indices(num_nodes)] = 1.
    triu_idx = triu_idx[get_offdiag_indices(num_nodes)]
    return triu_idx.nonzero()


def get_tril_offdiag_indices(num_nodes):
    """Linear tril (lower) indices w.r.t. vector of off-diagonal elements."""
    tril_idx = torch.zeros(num_nodes * num_nodes)
    tril_idx[get_tril_indices(num_nodes)] = 1.
    tril_idx = tril_idx[get_offdiag_indices(num_nodes)]
    return tril_idx.nonzero()


def get_minimum_distance(data):
    data = data[:, :, :, :2].transpose(1, 2)
    data_norm = (data ** 2).sum(-1, keepdim=True)
    dist = data_norm + \
           data_norm.transpose(2, 3) - \
           2 * torch.matmul(data, data.transpose(2, 3))
    min_dist, _ = dist.min(1)
    return min_dist.view(min_dist.size(0), -1)


def get_buckets(dist, num_buckets):
    dist = dist.cpu().data.numpy()

    min_dist = np.min(dist)
    max_dist = np.max(dist)
    bucket_size = (max_dist - min_dist) / num_buckets
    thresholds = bucket_size * np.arange(num_buckets)

    bucket_idx = []
    for i in range(num_buckets):
        if i < num_buckets - 1:
            idx = np.where(np.all(np.vstack((dist > thresholds[i],
                                             dist <= thresholds[i + 1])), 0))[0]
        else:
            idx = np.where(dist > thresholds[i])[0]
        bucket_idx.append(idx)

    return bucket_idx, thresholds


def get_correct_per_bucket(bucket_idx, pred, target):
    pred = pred.cpu().numpy()[:, 0]
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def get_correct_per_bucket_(bucket_idx, pred, target):
    pred = pred.cpu().numpy()
    target = target.cpu().data.numpy()

    correct_per_bucket = []
    for i in range(len(bucket_idx)):
        preds_bucket = pred[bucket_idx[i]]
        target_bucket = target[bucket_idx[i]]
        correct_bucket = np.sum(preds_bucket == target_bucket)
        correct_per_bucket.append(correct_bucket)

    return correct_per_bucket


def kl_categorical(preds, log_prior, num_atoms, eps=1e-16):
    kl_div = preds * (torch.log(preds + eps) - log_prior)
    return kl_div.sum() / (num_atoms * preds.size(0))  # normalisation here is (batch * num atoms)


def kl_categorical_uniform(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return kl_div.sum() / (num_atoms * preds.size(0))

def kl_categorical_uniform_var(preds, num_atoms, num_edge_types, add_const=False,
                           eps=1e-16):
    kl_div = preds * torch.log(preds + eps)
    if add_const:
        const = np.log(num_edge_types)
        kl_div += const
    return (kl_div.sum(dim=1) / num_atoms).var() 


def nll_gaussian(preds, target, variance, add_const=False):
    """
    loss function for fixed variance (log Gaussian)

    :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param variance: fixed value for the variance of the Gaussian. Type float
    :param add_const: True- adds the 1/2 ln(2*pi*variance) term
    :return: value of the loss function normalised by (batch * number of atoms)
    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)

def nll_gaussian_var(preds, target, variance, add_const=False):

    """
    returns the variance over the batch of the reconstruction loss

    :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param variance: fixed value for the variance of the Gaussian. Type float
    :param add_const: True- adds the 1/2 ln(2*pi*variance) term
    :return: variance of the loss function
    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return (neg_log_p.sum(dim=1)/target.size(1)).var()

#
def nll_gaussian_variablesigma(preds, target, sigma, epoch, temperature, total_epochs, add_const=True):
    """
        Loss function for the case of variable sigma, with isotropic gaussian

       :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
       :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
       :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
       :param epoch: value of the current epoch
       :param temperature: temperature used for the softplus for the additional biasing
       :param total_epochs: number of total epochs
       :param add_const: True- adds the 1/2 ln(2*pi*variance) term
       :return: value of the loss function normalised by (batch * number of atoms)
       """
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    # additional terms to add if we want to test how biasing helps
                #+ 0.1* (1-sigmoid(epoch, total_epochs/2, temperature)) * ((preds-target) ** 2 +variance)
    # np.exp(-epoch/temperature) *
    # neg_log_p = ((preds - target) ** 2 / (2 * variance))- 0.0000001/ sigma
    loss_1 = neg_log_p
    loss_2 = 0.0
    if add_const:
        const = (0.5 * torch.log(2*np.pi* variance))
        neg_log_p = neg_log_p + const
        loss_2 += const
    return neg_log_p.sum() / (target.size(0) * target.size(1)), loss_1.sum() / (target.size(0) * target.size(1)) , loss_2.sum() / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)


def nll_gaussian_var__variablesigma(preds, target, sigma, epoch, temperature, total_epochs, add_const=True):
    """
           Loss function for the case of variable sigma, with isotropic gaussian
          returns the variance over the batch of the reconstruction loss

          :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
          :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
          :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
          :param epoch: value of the current epoch
          :param temperature: temperature used for the softplus for the additional biasing
          :param total_epochs: number of total epochs
          :param add_const: True- adds the 1/2 ln(2*pi*variance) term
          :return: variation of the loss function
          """
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    # additional terms to add if we want to test how biasing helps
    # + 0.1 * (1-sigmoid(epoch, total_epochs/2, temperature)) * ((preds-target) ** 2 +variance)
    # np.exp(-epoch/temperature) *
    #neg_log_p = ((preds - target) ** 2 / (2 * variance))- 0.0000001/ sigma
    if add_const:
        const = (0.5 * torch.log(2*np.pi* variance))
        neg_log_p = neg_log_p + const
    return (neg_log_p.sum(dim=1)/target.size(1)).var()

def nll_gaussian_variablesigma_semiisotropic(preds, target, sigma, epoch, temperature, total_epochs, add_const=True):
    """
             Loss function for the case of variable sigma- semiisotropic => isotropic in (x,y) and (vx,vy)
             returns the variance over the batch of the reconstruction loss

             :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
             :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
             :param sigma: tensor of sigma values size [batch, particles, timesteps, 2]- 1 is (x,y) and 1 is (v_x,v_y)
             :param epoch: value of the current epoch
             :param temperature: temperature used for the softplus for the additional biasing
             :param total_epochs: number of total epochs
             :param add_const: True- adds the 1/2 ln(2*pi*variance) term
             :return: value of the loss function normalised by (batch * number of atoms)
             """
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    # select the positions for coords (2D) for 3D coords go to (0,1,2), (3,4,5) same coordes for position
    indices_pos = torch.LongTensor([0, 1])
    indices_vel = torch.LongTensor([2, 3])
    indices_pos_var = torch.LongTensor([0])
    indices_vel_var = torch.LongTensor([1])
    if preds.is_cuda:
        indices_pos, indices_vel, indices_pos_var, indices_vel_var = indices_pos.cuda(), indices_vel.cuda(), indices_pos_var.cuda(), indices_vel_var.cuda()
    positions = torch.index_select(preds, 3, indices_pos)
    velocities = torch.index_select(preds, 3, indices_vel)
    pos_targets = torch.index_select(target, 3, indices_pos)
    vel_targets = torch.index_select(target, 3, indices_vel)
    pos_var = torch.index_select(variance, 3, indices_pos_var)
    vel_var = torch.index_select(variance, 3, indices_vel_var)
    # recast the positions to the correct size
    pos_var = tile(pos_var, 3, list(positions.size())[3])
    vel_var = tile(vel_var, 3, list(velocities.size())[3])
    # gets the value of the loss
    neg_log_p = ((positions- pos_targets) ** 2 / (2 * pos_var)) + ((velocities - vel_targets) ** 2 / (2 * vel_var))
    # additional terms to add if we want to test how biasing helps
    # + 0.1* (1-sigmoid(epoch, total_epochs/2, temperature)) * ((preds-target) ** 2 +variance)
    # np.exp(-epoch/temperature) *
    # neg_log_p = ((preds - target) ** 2 / (2 * variance))- 0.0000001/ sigma
    # determinant of the covariance matrix with diagonal terms
    determinant = torch.prod(variance, 3).unsqueeze(3)
    loss_1 = neg_log_p
    loss_2 = 0.0
    if add_const:
        const = (0.5 * torch.log(2*np.pi* determinant))
        neg_log_p = neg_log_p + const
        loss_2 += const
    return neg_log_p.sum() / (target.size(0) * target.size(1)), loss_1.sum() / (target.size(0) * target.size(1)) , loss_2.sum() / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)


def nll_gaussian_var__variablesigma_semiisotropic(preds, target, sigma, epoch, temperature, total_epochs, add_const=True):
    """
               Loss function for the case of variable sigma- semiisotropic => isotropic in (x,y) and (vx,vy)
               returns the variance over the batch of the reconstruction loss

               :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
               :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
               :param sigma: tensor of sigma values size [batch, particles, timesteps, 2]- 1 is (x,y) and 1 is (v_x,v_y)
               :param epoch: value of the current epoch
               :param temperature: temperature used for the softplus for the additional biasing
               :param total_epochs: number of total epochs
               :param add_const: True- adds the 1/2 ln(2*pi*variance) term
               :return: variation of the loss function
               """
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    # select the positions for coords (2D) for 3D coords go to (0,1,2), (3,4,5) same coordes for position
    indices_pos = torch.LongTensor([0, 1])
    indices_vel = torch.LongTensor([2, 3])
    indices_pos_var = torch.LongTensor([0])
    indices_vel_var = torch.LongTensor([1])
    if preds.is_cuda:
        indices_pos, indices_vel, indices_pos_var, indices_vel_var = indices_pos.cuda(), indices_vel.cuda(), indices_pos_var.cuda(), indices_vel_var.cuda()
    positions = torch.index_select(preds, 3, indices_pos)
    velocities = torch.index_select(preds, 3, indices_vel)
    pos_targets = torch.index_select(target, 3, indices_pos)
    vel_targets = torch.index_select(target, 3, indices_vel)
    pos_var = torch.index_select(variance, 3, indices_pos_var)
    vel_var = torch.index_select(variance, 3, indices_vel_var)
    # recast the positions to the correct size
    pos_var = tile(pos_var, 3, list(positions.size())[3])
    vel_var = tile(vel_var, 3, list(velocities.size())[3])
    # gets the value of the loss
    neg_log_p = ((positions - pos_targets) ** 2 / (2 * pos_var)) + ((velocities - vel_targets) ** 2 / (2 * vel_var))
    # additional terms to add if we want to test how biasing helps
    # + 0.1* (1-sigmoid(epoch, total_epochs/2, temperature)) * ((preds-target) ** 2 +variance)
    # np.exp(-epoch/temperature) *
    # neg_log_p = ((preds - target) ** 2 / (2 * variance))- 0.0000001/ sigma
    # determinant of the covariance matrix with diagonal terms
    determinant = torch.prod(variance, 3).unsqueeze(3)
    loss_1 = neg_log_p
    loss_2 = 0.0
    if add_const:
        const = (0.5 * torch.log(2 * np.pi * determinant))
        neg_log_p = neg_log_p + const
        loss_2 += const
    return (neg_log_p.sum(dim=1)/target.size(1)).var()


def nll_lorentzian(preds, target, gamma):
    """
    Isotropic lorentzian loss function

    :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param gamma: The tensor for the FWHM of the distribution of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :return:  value of the loss function normalised by (batch * number of atoms)
    """
    gammasquared = gamma ** 2
    neg_log_p = torch.log(1+((preds - target) ** 2 / (gammasquared)))
    neg_log_p += torch.log(gamma)
    return neg_log_p.sum() / (target.size(0) * target.size(1))

def nll_lorentzian_var(preds, target, gamma):
    """
        Isotropic lorentzian loss function

        :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
        :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
        :param gamma: The tensor for the FWHM of the distribution of size [batch, particles, timesteps, (x,y,v_x,v_y)]
        :return: variance of the loss function normalised by (batch * number of atoms)
        """
    gammasquared = gamma ** 2
    neg_log_p = torch.log(1+((preds - target) ** 2 / (gammasquared)))
    neg_log_p += torch.log(gamma)
    return (neg_log_p.sum(dim=1)/target.size(1)).var()

def nll_gaussian_multivariatesigma_efficient(preds, target, sigma, accel, vel, add_const=True):
    """
    Loss function for the case of variable sigma multivariate normal case

    :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param accel: gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
    :param vel: gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
    :param add_const: True- adds the 1/2 ln(2*pi*variance) term
    :return: value of the loss function normalised by (batch * number of atoms), value for loss of each term
    """
    # get normalised vectors for acceleration and velocities v|| and a||
    # t = time.time()
    velnorm = vel.norm(p=2, dim = 3, keepdim = True)
    normalisedvel = vel.div(velnorm.expand_as(vel))
    # 1/sqrt(2) - isotropic => direction unimportant. chosen here to improve efficiency
    normalisedvel[torch.isnan(normalisedvel)] = np.power(1/2, 1/2)
    accelnorm = accel.norm(p=2, dim = 3, keepdim = True)
    normalisedaccel = accel.div(accelnorm.expand_as(accel))
    normalisedaccel[torch.isnan(normalisedaccel)] = np.power(1 / 2, 1 / 2)
    # print('extractdata: {:.1f}s'.format(time.time() - t))
    # get perpendicular components to the accelerations and velocities accelperp, velperp
    # # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
    # tim = time.time()
    # velperp = torch.zeros(normalisedvel.size()[0], normalisedvel.size()[1], normalisedvel.size()[2], normalisedvel.size()[3])
    # accelperp = torch.zeros(accelnorm.size()[0], accelnorm.size()[1], accelnorm.size()[2], normalisedvel.size()[3])
    # for i in range(normalisedvel.size()[0]):
    #     for j in range(normalisedvel[i].size()[0]):
    #         for k in range(normalisedvel[i][j].size()[0]):
    #             velperp[i][j][k][0] = -normalisedvel[i][j][k][1]
    #             velperp[i][j][k][1] = normalisedvel[i][j][k][0]
    #             accelperp[i][j][k][0] = -normalisedaccel[i][j][k][1]
    #             accelperp[i][j][k][1] = normalisedaccel[i][j][k][0]
    # if preds.is_cuda:
    #    velperp, accelperp = velperp.cuda(), accelperp.cuda()
    # print('getperp: {:.1f}s'.format(time.time() - tim))

    # need Sigma=Sigma^2, Sigma^-2 and det(Sigma)
    # ti = time.time()
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    determinant = torch.prod(variance, 3).unsqueeze(3)
    inversevariance = variance ** -1
    # need position and velocity differences in (x,y) coordinates
    differences = preds-target
    indices_pos = torch.LongTensor([0,1])
    indices_vel = torch.LongTensor([2,3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    position_differences = torch.index_select(differences, 3, indices_pos)
    velocity_differences = torch.index_select(differences, 3, indices_vel)
    position_differences = position_differences.unsqueeze(4)
    velocity_differences = velocity_differences.unsqueeze(4)# (x-mu)
    # print('getdifferences: {:.1f}s'.format(time.time() - ti))
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    # multime = time.time()
    # surprisingly it is more efficient to calculate the perpendicular term by considering
    # (position_differences - (position_differences.v||)v||).vperp to get the position differences in the perpendicular
    # direction than using rotation (x,y) -> (-y,x) as the triple for loop is inefficient. about 100x faster this way
    # and almost as fast as isotropic
    errorvectorparalleltov = torch.matmul(normalisedvel.unsqueeze(3), position_differences)
    parallelterm = torch.matmul(normalisedvel.unsqueeze(4), errorvectorparalleltov)
    perpterm = (position_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim = 3, keepdim = True)
    # NaN can occur when dividing by 0 (see comment below) but the problem with replacing NaN after the division is that
    # the NaN carries through anyway - the function that the system is backtracking through keeps the NaN =
    # therefore leads to NaN errors on the second pass of the function - replacing the 0's before division solves this
    # issue.
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN can occur when perpterm is 0, this means that preds-true = (preds-true).v|| v||
    # i.e. error entirely in parallel direction and no error perpendicular: so we set these terms to 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    # gets the error vectors
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()
    errorvectorparalleltoa = torch.matmul(normalisedaccel.unsqueeze(3), velocity_differences)
    parallelterm = torch.matmul(normalisedaccel.unsqueeze(4), errorvectorparalleltoa)
    perpterm = (velocity_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN when preds-target is entirely in the v parallel direction. This means the error in the perpendiular
    # direction is 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptoa = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltoa = errorvectorparalleltoa.squeeze()
    # errorvectorperptoa = torch.matmul(accelperp.unsqueeze(3), velocity_differences).squeeze()
    indices_vpar = torch.LongTensor([0])
    indices_vperp = torch.LongTensor([1])
    indices_apar = torch.LongTensor([2])
    indices_aperp = torch.LongTensor([3])
    #print('matrixmult: {:.1f}s'.format(time.time() - multime))
    if preds.is_cuda:
        indices_vpar, indices_vperp, indices_apar, indices_aperp = indices_vpar.cuda(), indices_vperp.cuda(), indices_apar.cuda(), indices_aperp.cuda()
    # t = time.time()
    # gets the loss components
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(inversevariance, 3, indices_vpar).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(inversevariance, 3, indices_vperp).squeeze()
    losscomponentparalleltoa = (errorvectorparalleltoa ** 2) * torch.index_select(inversevariance, 3, indices_apar).squeeze()
    losscomponentperptoa = (errorvectorperptoa ** 2) * torch.index_select(inversevariance, 3, indices_aperp).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov + losscomponentparalleltoa + losscomponentperptoa
    loss_1 = neg_log_loss
    loss_2 = 0.0
    # print('getlosscomponents: {:.1f}s'.format(time.time() - t))
    if add_const:
        const = (0.5 * torch.log(2*np.pi* determinant))
        neg_log_loss += const.squeeze()
        loss_2 += const.squeeze()
    return (neg_log_loss).sum() / (target.size(0) * target.size(1)), loss_1.sum() / (target.size(0) * target.size(1)) , loss_2 / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)

def nll_gaussian_var_multivariatesigma_efficient(preds, target, sigma, accel, vel, add_const=True):
    """
        Loss function for the case of variable sigma multivariate normal case

        :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
        :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
        :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
        :param accel: gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
        :param vel: gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
        :param add_const: True- adds the 1/2 ln(2*pi*variance) term
        :return: variance of the loss function
        """
    # get normalised vectors for acceleration and velocities v|| and a||
    velnorm = vel.norm(p=2, dim=3, keepdim=True)
    normalisedvel = vel.div(velnorm.expand_as(vel))
    normalisedvel[torch.isnan(normalisedvel)] = np.power(1 / 2, 1 / 2)
    accelnorm = accel.norm(p=2, dim=3, keepdim=True)
    normalisedaccel = accel.div(accelnorm.expand_as(accel))
    # get perpendicular components to the accelerations and velocities accelperp, velperp
    # # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
    # velperp = torch.zeros(normalisedvel.size()[0], normalisedvel.size()[1], normalisedvel.size()[2],
    #                       normalisedvel.size()[3])
    # accelperp = torch.zeros(accelnorm.size()[0], accelnorm.size()[1], accelnorm.size()[2], normalisedvel.size()[3])
    # for i in range(normalisedvel.size()[0]):
    #     for j in range(normalisedvel[i].size()[0]):
    #         for k in range(normalisedvel[i][j].size()[0]):
    #             velperp[i][j][k][0] = -normalisedvel[i][j][k][1]
    #             velperp[i][j][k][1] = normalisedvel[i][j][k][0]
    #             accelperp[i][j][k][0] = -normalisedaccel[i][j][k][1]
    #             accelperp[i][j][k][1] = normalisedaccel[i][j][k][0]
    # if preds.is_cuda:
    #    velperp, accelperp = velperp.cuda(), accelperp.cuda()
    # need Sigma=Sigma^2, Sigma^-2 and det(Sigma)
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    determinant = torch.prod(variance, 3).unsqueeze(3)
    inversevariance = variance ** -1
    # need position and velocity differences in (x,y) coordinates
    differences = preds - target
    indices_pos = torch.LongTensor([0, 1])
    indices_vel = torch.LongTensor([2, 3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    position_differences = torch.index_select(differences, 3, indices_pos)
    velocity_differences = torch.index_select(differences, 3, indices_vel)
    position_differences = position_differences.unsqueeze(4)
    velocity_differences = velocity_differences.unsqueeze(4)  # (x-mu)
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    errorvectorparalleltov = torch.matmul(normalisedvel.unsqueeze(3), position_differences)
    parallelterm = torch.matmul(normalisedvel.unsqueeze(4), errorvectorparalleltov)
    perpterm = (position_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()
    errorvectorparalleltoa = torch.matmul(normalisedaccel.unsqueeze(3), velocity_differences)
    parallelterm = torch.matmul(normalisedaccel.unsqueeze(4), errorvectorparalleltoa)
    perpterm = (velocity_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptoa = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltoa = errorvectorparalleltoa.squeeze()
    indices_vpar = torch.LongTensor([0])
    indices_vperp = torch.LongTensor([1])
    indices_apar = torch.LongTensor([2])
    indices_aperp = torch.LongTensor([3])
    if preds.is_cuda:
        indices_vpar, indices_vperp, indices_apar, indices_aperp = indices_vpar.cuda(), indices_vperp.cuda(), indices_apar.cuda(), indices_aperp.cuda()
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(inversevariance, 3,
                                                                                  indices_vpar).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(inversevariance, 3, indices_vperp).squeeze()
    losscomponentparalleltoa = (errorvectorparalleltoa ** 2) * torch.index_select(inversevariance, 3,
                                                                                  indices_apar).squeeze()
    losscomponentperptoa = (errorvectorperptoa ** 2) * torch.index_select(inversevariance, 3, indices_aperp).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov + losscomponentparalleltoa + losscomponentperptoa
    loss_1 = neg_log_loss
    loss_2 = 0.0
    if add_const:
        const = (0.5 * torch.log(2 * np.pi * determinant))
        neg_log_loss += const.squeeze()
        loss_2 += const.squeeze()
    return ((neg_log_loss).sum(dim=1)/target.size(1)).var()


def nll_gaussian_multivariatesigma_convexified(preds, target, sigma, accel, vel, sigma_prev, preds_prev, vvec, sigmavec, alpha, add_const=True):
    """
    Loss function for the case of variable sigma multivariate normal case with added convexification. The Algorithm
    follows that suggested by Edoardo Calvello

    :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param accel: gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
    :param sigma_prev: previous prediction of sigma
    :param preds_prev: previous prediction of the position and velocity space
    :param vel: gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
    :param vvec: vector that is used to provide the point of convexification from previous iteration. Size [batch, particles, timesteps, 4]
    :param sigmavec: same as vvec but for sigma parameters. Size [batch, particles, timesteps, 4]
    :param alpha: scale of the convexification. Float.
    :param add_const: True- adds the 1/2 ln(2*pi*variance) term
    :return: value of the loss function normalised by (batch * number of atoms), value for loss of each term
    """
    # according to algorithm, we want to convexify about yk= alphak vk-1 +(1-alphak)xk-1
    yphasespace = alpha * vvec + (1-alpha) * preds_prev
    ysigmaterm = alpha * sigmavec + (1-alpha) * sigma_prev
    # get normalised vectors for acceleration and velocities v|| and a||
    # t = time.time()
    velnorm = vel.norm(p=2, dim = 3, keepdim = True)
    normalisedvel = vel.div(velnorm.expand_as(vel))
    # 1/sqrt(2) - isotropic => direction unimportant. chosen here to improve efficiency
    normalisedvel[torch.isnan(normalisedvel)] = np.power(1/2, 1/2)
    accelnorm = accel.norm(p=2, dim = 3, keepdim = True)
    normalisedaccel = accel.div(accelnorm.expand_as(accel))
    normalisedaccel[torch.isnan(normalisedaccel)] = np.power(1 / 2, 1 / 2)
    # print('extractdata: {:.1f}s'.format(time.time() - t))
    # get perpendicular components to the accelerations and velocities accelperp, velperp
    # # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
    # need Sigma=Sigma^2, Sigma^-2 and det(Sigma)
    # ti = time.time()
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    determinant = torch.prod(variance, 3).unsqueeze(3)
    inversevariance = variance ** -1
    # need position and velocity differences in (x,y) coordinates
    differences = preds-target
    indices_pos = torch.LongTensor([0,1])
    indices_vel = torch.LongTensor([2,3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    position_differences = torch.index_select(differences, 3, indices_pos)
    velocity_differences = torch.index_select(differences, 3, indices_vel)
    position_differences = position_differences.unsqueeze(4)
    velocity_differences = velocity_differences.unsqueeze(4)# (x-mu)
    # print('getdifferences: {:.1f}s'.format(time.time() - ti))
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    # multime = time.time()
    # surprisingly it is more efficient to calculate the perpendicular term by considering
    # (position_differences - (position_differences.v||)v||).vperp to get the position differences in the perpendicular
    # direction than using rotation (x,y) -> (-y,x) as the triple for loop is inefficient. about 100x faster this way
    # and almost as fast as isotropic
    errorvectorparalleltov = torch.matmul(normalisedvel.unsqueeze(3), position_differences)
    parallelterm = torch.matmul(normalisedvel.unsqueeze(4), errorvectorparalleltov)
    perpterm = (position_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim = 3, keepdim = True)
    # NaN can occur when dividing by 0 (see comment below) but the problem with replacing NaN after the division is that
    # the NaN carries through anyway - the function that the system is backtracking through keeps the NaN =
    # therefore leads to NaN errors on the second pass of the function - replacing the 0's before division solves this
    # issue.
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN can occur when perpterm is 0, this means that preds-true = (preds-true).v|| v||
    # i.e. error entirely in parallel direction and no error perpendicular: so we set these terms to 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    # gets the error vectors
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()
    errorvectorparalleltoa = torch.matmul(normalisedaccel.unsqueeze(3), velocity_differences)
    parallelterm = torch.matmul(normalisedaccel.unsqueeze(4), errorvectorparalleltoa)
    perpterm = (velocity_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN when preds-target is entirely in the v parallel direction. This means the error in the perpendiular
    # direction is 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptoa = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltoa = errorvectorparalleltoa.squeeze()
    # errorvectorperptoa = torch.matmul(accelperp.unsqueeze(3), velocity_differences).squeeze()
    indices_vpar = torch.LongTensor([0])
    indices_vperp = torch.LongTensor([1])
    indices_apar = torch.LongTensor([2])
    indices_aperp = torch.LongTensor([3])
    #print('matrixmult: {:.1f}s'.format(time.time() - multime))
    if preds.is_cuda:
        indices_vpar, indices_vperp, indices_apar, indices_aperp = indices_vpar.cuda(), indices_vperp.cuda(), indices_apar.cuda(), indices_aperp.cuda()
    # t = time.time()
    # gets the loss components
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(inversevariance, 3, indices_vpar).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(inversevariance, 3, indices_vperp).squeeze()
    losscomponentparalleltoa = (errorvectorparalleltoa ** 2) * torch.index_select(inversevariance, 3, indices_apar).squeeze()
    losscomponentperptoa = (errorvectorperptoa ** 2) * torch.index_select(inversevariance, 3, indices_aperp).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov + losscomponentparalleltoa + losscomponentperptoa
    loss_1 = neg_log_loss
    loss_2 = 0.0
    # print('getlosscomponents: {:.1f}s'.format(time.time() - t))
    # convexifying term is 0.1 * ||x-y||^2 according to algorithm by Edoardo Calvello. lambda is chosen as 0.1 here
    convterm = 0.1 * ((preds-target) - yphasespace) ** 2 + 0.1 * (sigma - ysigmaterm) ** 2
    neg_log_loss += convterm.sum(dim = 3)
    if add_const:
        const = (0.5 * torch.log(2*np.pi* determinant))
        neg_log_loss += const.squeeze()
        loss_2 += const.squeeze()
    return (neg_log_loss).sum() / (target.size(0) * target.size(1)), loss_1.sum() / (target.size(0) * target.size(1)) , loss_2 / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)

def nll_gaussian_multivariatesigma_var_convexified(preds, target, sigma, accel, vel, sigma_prev, preds_prev, vvec, sigmavec, alpha, add_const=True):
    """
    Loss function for the case of variable sigma multivariate normal case with added convexification. The Algorithm
    follows that suggested by Edoardo Calvello

    :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param accel: gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
    :param sigma_prev: previous prediction of sigma
    :param preds_prev: previous prediction of the position and velocity space
    :param vel: gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
    :param vvec: vector that is used to provide the point of convexification from previous iteration. Size [batch, particles, timesteps, 4]
    :param sigmavec: same as vvec but for sigma parameters. Size [batch, particles, timesteps, 4]
    :param alpha: scale of the convexification. Float.
    :param add_const: True- adds the 1/2 ln(2*pi*variance) term
    :return: value of the loss function normalised by (batch * number of atoms), value for loss of each term
    """
    # according to algorithm, we want to convexify about yk= alphak vk-1 +(1-alphak)xk-1
    yphasespace = alpha * vvec + (1-alpha) * preds_prev
    ysigmaterm = alpha * sigmavec + (1-alpha) * sigma_prev
    # get normalised vectors for acceleration and velocities v|| and a||
    # t = time.time()
    velnorm = vel.norm(p=2, dim = 3, keepdim = True)
    normalisedvel = vel.div(velnorm.expand_as(vel))
    # 1/sqrt(2) - isotropic => direction unimportant. chosen here to improve efficiency
    normalisedvel[torch.isnan(normalisedvel)] = np.power(1/2, 1/2)
    accelnorm = accel.norm(p=2, dim = 3, keepdim = True)
    normalisedaccel = accel.div(accelnorm.expand_as(accel))
    normalisedaccel[torch.isnan(normalisedaccel)] = np.power(1 / 2, 1 / 2)
    # print('extractdata: {:.1f}s'.format(time.time() - t))
    # get perpendicular components to the accelerations and velocities accelperp, velperp
    # # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
    # need Sigma=Sigma^2, Sigma^-2 and det(Sigma)
    # ti = time.time()
    variance = sigma ** 2
    # ensures variance does not go to 0
    if (torch.min(variance) < pow(10, -10)):
        accuracy = np.full((variance.size(0), variance.size(1), variance.size(2), variance.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        variance = torch.max(variance, accuracy)
    determinant = torch.prod(variance, 3).unsqueeze(3)
    inversevariance = variance ** -1
    # need position and velocity differences in (x,y) coordinates
    differences = preds-target
    indices_pos = torch.LongTensor([0,1])
    indices_vel = torch.LongTensor([2,3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    position_differences = torch.index_select(differences, 3, indices_pos)
    velocity_differences = torch.index_select(differences, 3, indices_vel)
    position_differences = position_differences.unsqueeze(4)
    velocity_differences = velocity_differences.unsqueeze(4)# (x-mu)
    # print('getdifferences: {:.1f}s'.format(time.time() - ti))
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    # multime = time.time()
    # surprisingly it is more efficient to calculate the perpendicular term by considering
    # (position_differences - (position_differences.v||)v||).vperp to get the position differences in the perpendicular
    # direction than using rotation (x,y) -> (-y,x) as the triple for loop is inefficient. about 100x faster this way
    # and almost as fast as isotropic
    errorvectorparalleltov = torch.matmul(normalisedvel.unsqueeze(3), position_differences)
    parallelterm = torch.matmul(normalisedvel.unsqueeze(4), errorvectorparalleltov)
    perpterm = (position_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim = 3, keepdim = True)
    # NaN can occur when dividing by 0 (see comment below) but the problem with replacing NaN after the division is that
    # the NaN carries through anyway - the function that the system is backtracking through keeps the NaN =
    # therefore leads to NaN errors on the second pass of the function - replacing the 0's before division solves this
    # issue.
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN can occur when perpterm is 0, this means that preds-true = (preds-true).v|| v||
    # i.e. error entirely in parallel direction and no error perpendicular: so we set these terms to 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    # gets the error vectors
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()
    errorvectorparalleltoa = torch.matmul(normalisedaccel.unsqueeze(3), velocity_differences)
    parallelterm = torch.matmul(normalisedaccel.unsqueeze(4), errorvectorparalleltoa)
    perpterm = (velocity_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN when preds-target is entirely in the v parallel direction. This means the error in the perpendiular
    # direction is 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptoa = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltoa = errorvectorparalleltoa.squeeze()
    # errorvectorperptoa = torch.matmul(accelperp.unsqueeze(3), velocity_differences).squeeze()
    indices_vpar = torch.LongTensor([0])
    indices_vperp = torch.LongTensor([1])
    indices_apar = torch.LongTensor([2])
    indices_aperp = torch.LongTensor([3])
    #print('matrixmult: {:.1f}s'.format(time.time() - multime))
    if preds.is_cuda:
        indices_vpar, indices_vperp, indices_apar, indices_aperp = indices_vpar.cuda(), indices_vperp.cuda(), indices_apar.cuda(), indices_aperp.cuda()
    # t = time.time()
    # gets the loss components
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(inversevariance, 3, indices_vpar).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(inversevariance, 3, indices_vperp).squeeze()
    losscomponentparalleltoa = (errorvectorparalleltoa ** 2) * torch.index_select(inversevariance, 3, indices_apar).squeeze()
    losscomponentperptoa = (errorvectorperptoa ** 2) * torch.index_select(inversevariance, 3, indices_aperp).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov + losscomponentparalleltoa + losscomponentperptoa
    loss_1 = neg_log_loss
    loss_2 = 0.0
    # print('getlosscomponents: {:.1f}s'.format(time.time() - t))
    # convexifying term is 0.1 * ||x-y||^2 according to algorithm by Edoardo Calvello. lambda is chosen as 0.1 here
    convterm = 0.1 * ((preds-target) - yphasespace) ** 2 + 0.1 * (sigma - ysigmaterm) ** 2
    neg_log_loss += convterm.sum(dim = 3)
    if add_const:
        const = (0.5 * torch.log(2*np.pi* determinant))
        neg_log_loss += const.squeeze()
        loss_2 += const.squeeze()
    return ((neg_log_loss).sum(dim=1)/target.size(1)).var()


def nll_gaussian_var_multivariatesigma_withcorrelations(preds, target, sigma, accel, vel, eps= 1e-3, alpha = 0.05, add_const=True):
    """
            Loss function for the case of variable sigma multivariate normal case with correlations between coordinates.
            Implemented based on arXiv:1910.14215 [cs.LG] R.L. Russell et al findings
            sigma has shape [batchsize, no.ofparticles, times, (s11,rho12,rho21,s22,s33,rho34,rho43,s44)]
            :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
            :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
            :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
            :param accel: gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
            :param vel: gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
            :param eps: small term to ensure Pearson correlation coefficients are not close to 1: see arXiv:1910.14215 [cs.LG] R.L. Russell et al
            :param alpha: term that ensures Pearson correlation coefficients do not saturate quickly: see arXiv:1910.14215 [cs.LG] R.L. Russell et al
            :param add_const: True- adds the 1/2 ln(2*pi*variance) term
            :return: variance of the loss function
            """
    # get normalised vectors for acceleration and velocities v|| and a||
    # t = time.time()
    velnorm = vel.norm(p=2, dim=3, keepdim=True)
    normalisedvel = vel.div(velnorm.expand_as(vel))
    # 1/sqrt(2) - isotropic when NaN => direction unimportant. chosen here to improve efficiency
    normalisedvel[torch.isnan(normalisedvel)] = np.power(1 / 2, 1 / 2)
    accelnorm = accel.norm(p=2, dim=3, keepdim=True)
    normalisedaccel = accel.div(accelnorm.expand_as(accel))
    normalisedaccel[torch.isnan(normalisedaccel)] = np.power(1 / 2, 1 / 2)

    # Pearson correlation coeffns activation function to ensure they are within (-1,1) and 1-eps to ensure they are not
    # close to 1 and alpha to ensure they don't saturate quickly: see arXiv:1910.14215 [cs.LG] R.L. Russell et al
    indices_pos = torch.LongTensor([1, 2])
    indices_vel = torch.LongTensor([5, 6])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    # extract pearson coeffns output from NN
    rho_pos = torch.index_select(sigma, 3, indices_pos)
    rho_vel = torch.index_select(sigma, 3, indices_vel)
    # rescale pearson coeffns
    rho_pos = (1 - eps) * torch.tanh(alpha * rho_pos)
    rho_vel = (1 - eps) * torch.tanh(alpha * rho_vel)
    # extract each of the sigma terms for position and velocity
    indices_pos_1 = torch.LongTensor([0])
    indices_pos_2 = torch.LongTensor([3])
    indices_vel_1 = torch.LongTensor([4])
    indices_vel_2 = torch.LongTensor([7])
    if preds.is_cuda:
        indices_pos_1, indices_pos_2, indices_vel_1, indices_vel_2 = indices_pos_1.cuda(), indices_pos_2.cuda(), indices_vel_1.cuda(), indices_vel_2.cuda()
    sigma_pos_1 = torch.index_select(sigma, 3, indices_pos_1)
    sigma_pos_2 = torch.index_select(sigma, 3, indices_pos_2)
    sigma_vel_1 = torch.index_select(sigma, 3, indices_vel_1)
    sigma_vel_2 = torch.index_select(sigma, 3, indices_vel_2)
    # off diagonal terms given by rho sigma1 sigma2
    sigma_term_pos = torch.sqrt((sigma_pos_1 * sigma_pos_2))
    sigma_term_vel = torch.sqrt((sigma_vel_1 * sigma_vel_2))
    offdiagsigma_pos = tile(sigma_term_pos, 3, rho_pos.size(3))
    offdiagsigma_vel = tile(sigma_term_vel, 3, rho_vel.size(3))
    sigmaoffdiag_pos = rho_pos * offdiagsigma_pos
    sigmaoffdiag_vel = rho_vel * offdiagsigma_vel
    # need Sigma=Sigma^2, Sigma^-2 and det(Sigma)
    # ti = time.time()
    # reconstruct sigma from position and velocity
    indices_pos_1 = torch.LongTensor([0])
    indices_pos_2 = torch.LongTensor([3])
    indices_vel_1 = torch.LongTensor([4])
    indices_vel_2 = torch.LongTensor([7])
    if preds.is_cuda:
        indices_pos_1, indices_pos_2, indices_vel_1, indices_vel_2 = indices_pos_1.cuda(), indices_pos_2.cuda(), indices_vel_1.cuda(), indices_vel_2.cuda()
    sigma_pos = torch.cat((torch.cat((torch.index_select(sigma, 3, indices_pos_1), sigmaoffdiag_pos), 3),
                           torch.index_select(sigma, 3, indices_pos_2)), 3)
    sigma_vel = torch.cat((torch.cat((torch.index_select(sigma, 3, indices_vel_1), sigmaoffdiag_vel), 3),
                           torch.index_select(sigma, 3, indices_vel_2)), 3)
    sigma_pos = sigma_pos.reshape(sigma.size(0), sigma.size(1), sigma.size(2), 2, 2)
    sigma_vel = sigma_vel.reshape(sigma.size(0), sigma.size(1), sigma.size(2), 2, 2)
    # get sigma^2 for pos and vel
    variance_pos = torch.matmul(sigma_pos, sigma_pos)
    variance_vel = torch.matmul(sigma_vel, sigma_vel)
    # reshape to desired shape for use
    variance_pos = variance_pos.reshape(variance_pos.size(0), variance_pos.size(1), variance_pos.size(2), 4)
    variance_vel = variance_vel.reshape(variance_vel.size(0), variance_vel.size(1), variance_vel.size(2), 4)
    indices_sigma = torch.LongTensor([0, 3])
    indices_diag_1 = torch.LongTensor([1, 2])
    if preds.is_cuda:
        indices_sigma, indices_diag_1 = indices_sigma.cuda(), indices_diag_1.cuda()
    # extract variance
    var_pos = torch.index_select(variance_pos, 3, indices_sigma)
    var_vel = torch.index_select(variance_vel, 3, indices_sigma)
    offdiag_pos = torch.index_select(variance_pos, 3, indices_diag_1)
    offdiag_vel = torch.index_select(variance_vel, 3, indices_diag_1)
    # ensures variance does not go to 0
    if (torch.min(var_pos) < pow(10, -14)) or (torch.min(var_vel) < pow(10, -14)):
        accuracy = np.full((var_pos.size(0), var_pos.size(1), var_pos.size(2), var_pos.size(3)),
                           pow(10, -14), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        var_pos = torch.max(var_pos, accuracy)
        var_vel = torch.max(var_vel, accuracy)
    indices_1 = torch.LongTensor([0])
    indices_2 = torch.LongTensor([1])
    if preds.is_cuda:
        indices_1, indices_2 = indices_1.cuda(), indices_2.cuda()
    # recasts the variance into desired form
    variance_pos = torch.cat((torch.cat((torch.index_select(var_pos, 3, indices_1), offdiag_pos), 3),
                              torch.index_select(var_pos, 3, indices_2)), 3)
    variance_vel = torch.cat((torch.cat((torch.index_select(var_vel, 3, indices_1), offdiag_vel), 3),
                              torch.index_select(var_vel, 3, indices_2)), 3)
    variance_pos = variance_pos.reshape(variance_pos.size(0), variance_pos.size(1), variance_pos.size(2), 2, 2)
    variance_vel = variance_vel.reshape(variance_vel.size(0), variance_vel.size(1), variance_vel.size(2), 2, 2)
    # determinant of block diagonal matrix = product of submatrices determinants
    determinant_pos = variance_pos.det()
    determinant_vel = variance_vel.det()
    determinant = determinant_vel * determinant_pos
    # Matrix not invertable iff sigma1 or sigma2 == 0 or Pearson correlation coeffs are 1 (we ensure this is not the
    # case above)
    # of form 1/(1-rho^2)  (1/sigma1^2,      -rho/sigma1sigma2
    #                       -rho/sigma1sigma2       1/sigma2^2)
    inversevariance_pos = torch.inverse(variance_pos)
    inversevariance_vel = torch.inverse(variance_vel)
    # recasts inverse variance into desired shape
    inversevariance_pos = inversevariance_pos.reshape(inversevariance_pos.size(0), inversevariance_pos.size(1),
                                                      inversevariance_pos.size(2), 4)
    inversevariance_vel = inversevariance_vel.reshape(inversevariance_vel.size(0), inversevariance_vel.size(1),
                                                      inversevariance_vel.size(2), 4)
    # if np.isnan(np.sum(inversevariance.cpu().detach().numpy())):
    #     print("Some values from variance are nan")
    # need position and velocity differences in (x,y) coordinates
    differences = preds - target
    indices_pos = torch.LongTensor([0, 1])
    indices_vel = torch.LongTensor([2, 3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    position_differences = torch.index_select(differences, 3, indices_pos)
    velocity_differences = torch.index_select(differences, 3, indices_vel)
    position_differences = position_differences.unsqueeze(4)
    velocity_differences = velocity_differences.unsqueeze(4)  # (x-mu)
    # print('getdifferences: {:.1f}s'.format(time.time() - ti))
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    # multime = time.time()
    # surprisingly it is more efficient to calculate the perpendicular term by considering
    # (position_differences - (position_differences.v||)v||).vperp to get the position differences in the perpendicular
    # direction than using rotation (x,y) -> (-y,x) as the triple for loop is inefficient. about 100x faster this way
    # and almost as fast as isotropic
    errorvectorparalleltov = torch.matmul(normalisedvel.unsqueeze(3), position_differences)
    parallelterm = torch.matmul(normalisedvel.unsqueeze(4), errorvectorparalleltov)
    perpterm = (position_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    # NaN can occur when dividing by 0 (see comment below) but the problem with replacing NaN after the division is that
    # the NaN carries through anyway - the function that the system is backtracking through keeps the NaN =
    # therefore leads to NaN errors on the second pass of the function - replacing the 0's before division solves this
    # issue.
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN can occur when perpterm is 0, this means that preds-true = (preds-true).v|| v||
    # i.e. error entirely in parallel direction and no error perpendicular: so we set these terms to 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()
    errorvectorparalleltoa = torch.matmul(normalisedaccel.unsqueeze(3), velocity_differences)
    parallelterm = torch.matmul(normalisedaccel.unsqueeze(4), errorvectorparalleltoa)
    perpterm = (velocity_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN when preds-target is entirely in the v parallel direction. This means the error in the perpendiular
    # direction is 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptoa = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltoa = errorvectorparalleltoa.squeeze()
    # errorvectorperptoa = torch.matmul(accelperp.unsqueeze(3), velocity_differences).squeeze()
    indices_par = torch.LongTensor([0])
    indices_perp = torch.LongTensor([3])
    indices_rho12 = torch.LongTensor([1])
    indices_rho21 = torch.LongTensor([2])
    # print('matrixmult: {:.1f}s'.format(time.time() - multime))
    if preds.is_cuda:
        indices_par, indices_perp, indices_rho12, indices_rho21 = indices_par.cuda(), indices_perp.cuda(), indices_rho12.cuda(), indices_rho21.cuda()
    # t = time.time()
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(inversevariance_pos, 3,
                                                                                  indices_par).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(inversevariance_pos, 3,
                                                                          indices_perp).squeeze()
    losscomponentparalleltoa = (errorvectorparalleltoa ** 2) * torch.index_select(inversevariance_vel, 3,
                                                                                  indices_par).squeeze()
    losscomponentperptoa = (errorvectorperptoa ** 2) * torch.index_select(inversevariance_vel, 3,
                                                                          indices_perp).squeeze()
    losscomponentoffdiagv = (errorvectorperptov * errorvectorparalleltov) * (
                torch.index_select(inversevariance_pos, 3, indices_rho12) + torch.index_select(inversevariance_pos, 3,
                                                                                               indices_rho21)).squeeze()
    losscomponentoffdiaga = (errorvectorperptoa * errorvectorparalleltoa) * (
                torch.index_select(inversevariance_vel, 3, indices_rho12) + torch.index_select(inversevariance_vel, 3,
                                                                                               indices_rho21)).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov + losscomponentparalleltoa + losscomponentperptoa + losscomponentoffdiagv + losscomponentoffdiaga
    loss_1 = neg_log_loss
    loss_2 = 0.0
    # print('getlosscomponents: {:.1f}s'.format(time.time() - t))
    if add_const:
        const = (0.5 * torch.log(2 * np.pi * determinant))
        neg_log_loss += const.squeeze()
        loss_2 += const.squeeze()
    return ((neg_log_loss).sum(dim=1)/target.size(1)).var()

def nll_gaussian_multivariatesigma_withcorrelations(preds, target, sigma, accel, vel, eps= 1e-3, alpha = 0.2, add_const=True):
    """
               Loss function for the case of variable sigma multivariate normal case with correlations between coordinates.
               Implemented based on arXiv:1910.14215 [cs.LG] R.L. Russell et al findings
               sigma has shape [batchsize, no.ofparticles, times, (s11,rho12,rho21,s22,s33,rho34,rho43,s44)]
               :param preds: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
               :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
               :param sigma: tensor of sigma values size [batch, particles, timesteps, (x,y,v_x,v_y)]
               :param accel: gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
               :param vel: gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
               :param eps: small term to ensure Pearson correlation coefficients are not close to 1: see arXiv:1910.14215 [cs.LG] R.L. Russell et al
               :param alpha: term that ensures Pearson correlation coefficients do not saturate quickly: see arXiv:1910.14215 [cs.LG] R.L. Russell et al
               :param add_const: True- adds the 1/2 ln(2*pi*variance) term
               :return: value of the loss function normalised by (batch * number of atoms), value for loss of each term
               """
    # get normalised vectors for acceleration and velocities v|| and a||
    # t = time.time()
    velnorm = vel.norm(p=2, dim=3, keepdim=True)
    normalisedvel = vel.div(velnorm.expand_as(vel))
    # 1/sqrt(2) - isotropic when NaN => direction unimportant. chosen here to improve efficiency
    normalisedvel[torch.isnan(normalisedvel)] = np.power(1 / 2, 1 / 2)
    accelnorm = accel.norm(p=2, dim=3, keepdim=True)
    normalisedaccel = accel.div(accelnorm.expand_as(accel))
    normalisedaccel[torch.isnan(normalisedaccel)] = np.power(1 / 2, 1 / 2)

    # Pearson correlation coeffns activation function to ensure they are within (-1,1) and 1-eps to ensure they are not
    # close to 1 and alpha to ensure they don't saturate quickly: see arXiv:1910.14215 [cs.LG] R.L. Russell et al
    indices_pos = torch.LongTensor([1, 2])
    indices_vel = torch.LongTensor([5, 6])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    # extract pearson coeffns output from NN
    rho_pos = torch.index_select(sigma, 3, indices_pos)
    rho_vel = torch.index_select(sigma, 3, indices_vel)
    # rescale pearson coeffns
    rho_pos = (1 - eps) * torch.tanh(alpha * rho_pos)
    rho_vel = (1 - eps) * torch.tanh(alpha * rho_vel)
    # extract each of the sigma terms for position and velocity
    indices_pos_1 = torch.LongTensor([0])
    indices_pos_2 = torch.LongTensor([3])
    indices_vel_1 = torch.LongTensor([4])
    indices_vel_2 = torch.LongTensor([7])
    if preds.is_cuda:
        indices_pos_1, indices_pos_2, indices_vel_1, indices_vel_2 = indices_pos_1.cuda(), indices_pos_2.cuda(), indices_vel_1.cuda(), indices_vel_2.cuda()
    sigma_pos_1 = torch.index_select(sigma, 3, indices_pos_1)
    sigma_pos_2 = torch.index_select(sigma, 3, indices_pos_2)
    sigma_vel_1 = torch.index_select(sigma, 3, indices_vel_1)
    sigma_vel_2 = torch.index_select(sigma, 3, indices_vel_2)
    # off diagonal terms given by rho sigma1 sigma2
    sigma_term_pos = torch.sqrt((sigma_pos_1 * sigma_pos_2))
    sigma_term_vel = torch.sqrt((sigma_vel_1 * sigma_vel_2))
    offdiagsigma_pos = tile(sigma_term_pos, 3, rho_pos.size(3))
    offdiagsigma_vel = tile(sigma_term_vel, 3, rho_vel.size(3))
    sigmaoffdiag_pos = rho_pos * offdiagsigma_pos
    sigmaoffdiag_vel = rho_vel * offdiagsigma_vel
    # need Sigma=Sigma^2, Sigma^-2 and det(Sigma)
    # ti = time.time()
    # reconstruct sigma from position and velocity
    indices_pos_1 = torch.LongTensor([0])
    indices_pos_2 = torch.LongTensor([3])
    indices_vel_1 = torch.LongTensor([4])
    indices_vel_2 = torch.LongTensor([7])
    if preds.is_cuda:
        indices_pos_1, indices_pos_2, indices_vel_1, indices_vel_2 = indices_pos_1.cuda(), indices_pos_2.cuda(), indices_vel_1.cuda(), indices_vel_2.cuda()
    sigma_pos = torch.cat((torch.cat((torch.index_select(sigma, 3, indices_pos_1), sigmaoffdiag_pos), 3),
                           torch.index_select(sigma, 3, indices_pos_2)), 3)
    sigma_vel = torch.cat((torch.cat((torch.index_select(sigma, 3, indices_vel_1), sigmaoffdiag_vel), 3),
                           torch.index_select(sigma, 3, indices_vel_2)), 3)
    sigma_pos = sigma_pos.reshape(sigma.size(0), sigma.size(1), sigma.size(2), 2, 2)
    sigma_vel = sigma_vel.reshape(sigma.size(0), sigma.size(1), sigma.size(2), 2, 2)
    # get sigma^2 for pos and vel
    variance_pos = torch.matmul(sigma_pos, sigma_pos)
    variance_vel = torch.matmul(sigma_vel, sigma_vel)
    # reshape to desired shape for use
    variance_pos = variance_pos.reshape(variance_pos.size(0), variance_pos.size(1), variance_pos.size(2), 4)
    variance_vel = variance_vel.reshape(variance_vel.size(0), variance_vel.size(1), variance_vel.size(2), 4)
    indices_sigma = torch.LongTensor([0, 3])
    indices_diag_1 = torch.LongTensor([1, 2])
    if preds.is_cuda:
        indices_sigma, indices_diag_1 = indices_sigma.cuda(), indices_diag_1.cuda()
    # extract variance
    var_pos = torch.index_select(variance_pos, 3, indices_sigma)
    var_vel = torch.index_select(variance_vel, 3, indices_sigma)
    offdiag_pos = torch.index_select(variance_pos, 3, indices_diag_1)
    offdiag_vel = torch.index_select(variance_vel, 3, indices_diag_1)
    # ensures variance does not go to 0
    if (torch.min(var_pos) < pow(10, -14)) or (torch.min(var_vel) < pow(10, -14)):
        accuracy = np.full((var_pos.size(0), var_pos.size(1), var_pos.size(2), var_pos.size(3)),
                           pow(10, -14), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        var_pos = torch.max(var_pos, accuracy)
        var_vel = torch.max(var_vel, accuracy)
    indices_1 = torch.LongTensor([0])
    indices_2 = torch.LongTensor([1])
    if preds.is_cuda:
        indices_1, indices_2 = indices_1.cuda(), indices_2.cuda()
    # recasts the variance into desired form
    variance_pos = torch.cat((torch.cat((torch.index_select(var_pos, 3, indices_1), offdiag_pos), 3),
                              torch.index_select(var_pos, 3, indices_2)), 3)
    variance_vel = torch.cat((torch.cat((torch.index_select(var_vel, 3, indices_1), offdiag_vel), 3),
                              torch.index_select(var_vel, 3, indices_2)), 3)
    variance_pos = variance_pos.reshape(variance_pos.size(0), variance_pos.size(1), variance_pos.size(2), 2, 2)
    variance_vel = variance_vel.reshape(variance_vel.size(0), variance_vel.size(1), variance_vel.size(2), 2, 2)
    # determinant of block diagonal matrix = product of submatrices determinants
    determinant_pos = variance_pos.det()
    determinant_vel = variance_vel.det()
    determinant = determinant_vel * determinant_pos
    # Matrix not invertable iff sigma1 or sigma2 == 0 or Pearson correlation coeffs are 1 (we ensure this is not the
    # case above)
    # of form 1/(1-rho^2)  (1/sigma1^2,      -rho/sigma1sigma2
    #                       -rho/sigma1sigma2       1/sigma2^2)
    inversevariance_pos = torch.inverse(variance_pos)
    inversevariance_vel = torch.inverse(variance_vel)
    inversevariance_pos = inversevariance_pos.reshape(inversevariance_pos.size(0), inversevariance_pos.size(1), inversevariance_pos.size(2), 4)
    inversevariance_vel = inversevariance_vel.reshape(inversevariance_vel.size(0), inversevariance_vel.size(1),
                                                   inversevariance_vel.size(2), 4)
    # if np.isnan(np.sum(inversevariance.cpu().detach().numpy())):
    #     print("Some values from variance are nan")
    # need position and velocity differences in (x,y) coordinates
    differences = preds-target
    indices_pos = torch.LongTensor([0,1])
    indices_vel = torch.LongTensor([2,3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    position_differences = torch.index_select(differences, 3, indices_pos)
    velocity_differences = torch.index_select(differences, 3, indices_vel)
    position_differences = position_differences.unsqueeze(4)
    velocity_differences = velocity_differences.unsqueeze(4)# (x-mu)
    # print('getdifferences: {:.1f}s'.format(time.time() - ti))
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    # multime = time.time()
    # surprisingly it is more efficient to calculate the perpendicular term by considering
    # (position_differences - (position_differences.v||)v||).vperp to get the position differences in the perpendicular
    # direction than using rotation (x,y) -> (-y,x) as the triple for loop is inefficient. about 100x faster this way
    # and almost as fast as isotropic
    errorvectorparalleltov = torch.matmul(normalisedvel.unsqueeze(3), position_differences)
    parallelterm = torch.matmul(normalisedvel.unsqueeze(4), errorvectorparalleltov)
    perpterm = (position_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim = 3, keepdim = True)
    # NaN can occur when dividing by 0 (see comment below) but the problem with replacing NaN after the division is that
    # the NaN carries through anyway - the function that the system is backtracking through keeps the NaN =
    # therefore leads to NaN errors on the second pass of the function - replacing the 0's before division solves this
    # issue.
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN can occur when perpterm is 0, this means that preds-true = (preds-true).v|| v||
    # i.e. error entirely in parallel direction and no error perpendicular: so we set these terms to 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()
    errorvectorparalleltoa = torch.matmul(normalisedaccel.unsqueeze(3), velocity_differences)
    parallelterm = torch.matmul(normalisedaccel.unsqueeze(4), errorvectorparalleltoa)
    perpterm = (velocity_differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN when preds-target is entirely in the v parallel direction. This means the error in the perpendiular
    # direction is 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptoa = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltoa = errorvectorparalleltoa.squeeze()
    # errorvectorperptoa = torch.matmul(accelperp.unsqueeze(3), velocity_differences).squeeze()
    indices_par = torch.LongTensor([0])
    indices_perp = torch.LongTensor([3])
    indices_rho12 = torch.LongTensor([1])
    indices_rho21 = torch.LongTensor([2])
    #print('matrixmult: {:.1f}s'.format(time.time() - multime))
    if preds.is_cuda:
        indices_par, indices_perp, indices_rho12, indices_rho21 = indices_par.cuda(), indices_perp.cuda(), indices_rho12.cuda(), indices_rho21.cuda()
    # t = time.time()
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(inversevariance_pos, 3, indices_par).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(inversevariance_pos, 3, indices_perp).squeeze()
    losscomponentparalleltoa = (errorvectorparalleltoa ** 2) * torch.index_select(inversevariance_vel, 3, indices_par).squeeze()
    losscomponentperptoa = (errorvectorperptoa ** 2) * torch.index_select(inversevariance_vel, 3, indices_perp).squeeze()
    losscomponentoffdiagv = (errorvectorperptov *errorvectorparalleltov) * (torch.index_select(inversevariance_pos, 3, indices_rho12) + torch.index_select(inversevariance_pos, 3, indices_rho21)).squeeze()
    losscomponentoffdiaga = (errorvectorperptoa *errorvectorparalleltoa) * (torch.index_select(inversevariance_vel, 3, indices_rho12) + torch.index_select(inversevariance_vel, 3, indices_rho21)).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov + losscomponentparalleltoa + losscomponentperptoa+ losscomponentoffdiagv + losscomponentoffdiaga
    loss_1 = neg_log_loss
    loss_2 = 0.0
    # print('getlosscomponents: {:.1f}s'.format(time.time() - t))
    if add_const:
        const = (0.5 * torch.log(2*np.pi* determinant))
        neg_log_loss += const.squeeze()
        loss_2 += const.squeeze()
    return (neg_log_loss).sum() / (target.size(0) * target.size(1)), loss_1.sum() / (target.size(0) * target.size(1)) , loss_2 / (target.size(0) * target.size(1)) # normalisation here is (batch * num atoms)

def true_flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def trace(A):
    """
    taken from https://github.com/pytorch/pytorch/issues/7500
    Takes the trace of the matrix
    :param A: Tensor of at least dimension [1,1]. Takes trace of last two dimensions
    """
    return A.diagonal(dim1=-2, dim2=-1).sum(-1)

def KL_output_multivariate(output, sigma, target, sigma_target, eps=1e-20):
    """
    KL term for the multivariate Gaussian distribution. Trying to compare the output distribution to a prior Gaussian
    distribution.

    :param output: prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param sigma: tensor of sigma values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param target: target data of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param sigma_target: tensor of sigma values from the prior of size [batch, particles, timesteps, (x,y,v_x,v_y)]
    :param eps: small term to ensure that the logarithm doesn't become 0
    :return: KL term normalised by batch size and no. of particles.
    """
    # variance and target variance
    variance = sigma ** 2
    variance_target = sigma_target[:,:,:sigma.size(2),:] ** 2
    # ensures the inverse will not yield NaN
    if (torch.min(variance_target) < pow(10, -10)):
        accuracy = np.full((variance_target.size(0), variance_target.size(1), variance_target.size(2), variance_target.size(3)),
                           pow(10, -10), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if output.is_cuda:
            accuracy = accuracy.cuda()
        variance_target = torch.max(variance_target, accuracy)
    inversevariance_target = variance_target ** -1
    trace_term = torch.sum(inversevariance_target * variance, dim = 3)

    errorvect = (target[:,:,:,:] - output[:,:,:,:]) ** 2
    error_term = errorvect * inversevariance_target

    determinant_variance = torch.prod(variance, dim =3)
    determinant_variance_target = torch.prod(variance_target, dim =3)
    logterm = torch.log((determinant_variance_target+eps)/(determinant_variance+eps))
    # add all 3 contributions
    KL_term = 1/2*(trace_term + error_term.sum(dim = 3)+logterm )
    return (KL_term).sum() / (target.size(0) * target.size(1))

def get_deltax0(target):
    """
    Gets the value of the mean change in position and velocity in the first timestep

    :param target: tensor of all data points from simulation
    target has dimensions [batch, particle, timestep, state]
    :return: mean change in position and velocity over the first timestep
    """
    # separate out the velocity and position terms
    indices = torch.LongTensor([0,1])
    indices_vel = torch.LongTensor([2,3])
    if target.is_cuda:
        indices, indices_vel = indices.cuda(), indices_vel.cuda()
    target_pos = torch.index_select(target, 3, indices)
    # calculate the magnitude in change in displacement
    deltax0 = (target_pos[:,:,1,:]-target_pos[:,:,0,:]).squeeze()
    deltax0 = deltax0.norm(p=2 , dim = 2, keepdim=True)
    target_vel = torch.index_select(target, 3, indices_vel)
    # calculate the magnitude in the change in velocity
    deltav0 = (target_vel[:, :, 1, :] - target_vel[:, :, 0, :]).squeeze()
    deltav0 = deltav0.norm(p=2, dim=2, keepdim=True)
    return deltax0.mean(), deltav0.mean()

def get_errorarray(phys_error_folder, comp_error_folder,data_folder = 'data', sim_folder = ''):
    """

    :param phys_error_folder: folder containing the theoretical values of the physical error
    :param comp_error_folder: folder containing the theoretical values of the computational error
    :param data_folder: folder containing the data
    :param sim_folder: folder containing the simulation data
    :return: the array for the contribution to sigma prior due to computational and physical errors for position and velocity
    """
    # phys_errors has shape [different_sigma, timestep]. Get them from their respective files
    phys_errors_pos = np.load(path.join(data_folder, phys_error_folder, 'mse_model_pos.npy'))
    phys_errors_vel = np.load(path.join(data_folder, phys_error_folder, 'mse_model_vel.npy'))
    # array of sigma values used in phys_errors- different terms in dim 1
    sigma = np.load(path.join(data_folder, phys_error_folder, 'sigma.npy'))
    # comp_errors has shape [timestep] - we know what the comp error should follow
    comp_errors_pos = np.load(path.join(data_folder, comp_error_folder, 'mse_model_pos.npy'))
    comp_errors_vel = np.load(path.join(data_folder, comp_error_folder, 'mse_model_pos.npy'))
    # index of current sigma: starts at smallest input value
    sigma_current_pos = 0
    sigma_current_vel = 0
    # the data here is to recast the values calculated by the simulator into the range of the data in the model
    loc_train = np.load(path.join(data_folder, sim_folder, 'loc_train.npy'))
    vel_train = np.load(path.join(data_folder, sim_folder, 'vel_train.npy'))
    loc_max = loc_train.max()
    loc_min = loc_train.min()
    vel_max = vel_train.max()
    vel_min = vel_train.min()

    # Normalize to [-1, 1]
    phys_errors_vel = (phys_errors_vel - vel_min) * 2 / (vel_max - vel_min) - 1
    comp_errors_vel = (comp_errors_vel - vel_min) * 2 / (vel_max - vel_min) - 1
    phys_errors_pos = (phys_errors_pos - loc_min) * 2 / (loc_max - loc_min) - 1
    comp_errors_pos = (comp_errors_pos - loc_min) * 2 / (loc_max - loc_min) - 1
    sigma = (sigma - loc_min) * 2 / (loc_max - loc_min) - 1
    delta_x_sqrd_array = []
    delta_v_sqrd_array = []
    offset_pos = 0
    offset_vel = 0
    # recursively build the array
    for i in range(len(comp_errors_pos)):
        delta_x_sqrd = comp_errors_pos[i] ** 2 + phys_errors_pos[sigma_current_pos, i-offset_pos] ** 2
        delta_v_sqrd = comp_errors_vel[i] ** 2 + phys_errors_vel[sigma_current_vel, i - offset_vel] ** 2
        # we have the max sigma so just use that
        if (sigma_current_pos == len(sigma)-1):
            delta_x_sqrd_array.append(delta_x_sqrd)
        else:
            # if the error is greater than sigma_0 we use this as the new sigma value for physical errors and start
            # from the begining
            if (delta_x_sqrd > sigma[sigma_current_pos+1] ** 2):
                sigma_current_pos = sigma_current_pos + 1
                delta_x_sqrd = comp_errors_pos[i] ** 2 + phys_errors_pos[sigma_current_pos, 0] ** 2
                delta_x_sqrd_array.append(delta_x_sqrd)
                offset_pos = i
            else:
                # in the case we do not need to use new sigma value just append value to array.
                delta_x_sqrd_array.append(delta_x_sqrd)
                # we have the max sigma so just use that
        if (sigma_current_vel == len(sigma) - 1):
            delta_v_sqrd_array.append(delta_v_sqrd)
        else:
            # if the error is greater than sigma_0 we use this as the new sigma value for physical errors and start
            # from the begining
            if (delta_v_sqrd > sigma[sigma_current_vel + 1] ** 2):
                sigma_current_vel = sigma_current_vel + 1
                delta_v_sqrd = comp_errors_vel[i] ** 2 + phys_errors_vel[sigma_current_vel, 0] ** 2
                delta_v_sqrd_array.append(delta_v_sqrd)
                offset_vel = i
            else:
                # in the case we do not need to use new sigma value just append value to array.
                delta_v_sqrd_array.append(delta_v_sqrd)
                # we have the max sigma so just use that
    delta_x_sqrd_array = torch.FloatTensor(delta_x_sqrd_array)
    delta_v_sqrd_array = torch.FloatTensor(delta_v_sqrd_array)
    return delta_x_sqrd_array, delta_v_sqrd_array


def getsigma_target(target, phys_error_folder, comp_error_folder, data_folder = 'data', sim_folder = ''):
    """

    :param target: tensor of all data points from simulation
    target has dimensions [batch, particle, timestep, state]
    :param phys_error_folder: folder containing the theoretical values of the physical error
    :param comp_error_folder: folder containing the theoretical values of the computational error
    :param data_folder: folder containing the data
    :param sim_folder: folder containing the simulation data
    :return: the array for the prior sigma tensor
    """
    # gets the terms for the mean shift in position and velocity at the 1st timestep
    deltax_0, deltav_0 = get_deltax0(target)
    # gets the contribution due to errors
    delta_x_error_array, delta_v_error_array = get_errorarray( phys_error_folder, comp_error_folder,data_folder, sim_folder)
    if target.is_cuda:
        delta_x_error_array, delta_v_error_array = delta_x_error_array.cuda(), delta_v_error_array.cuda()
    delta_x_error_array, delta_v_error_array = Variable(delta_x_error_array), Variable(delta_v_error_array)
    # deltax^2 = deltax_0 ^2 + delta_x from error considerations
    delta_x_array = delta_x_error_array + deltax_0 ** 2
    delta_v_array = delta_v_error_array + deltav_0 ** 2
    delta_x_array = tile(delta_x_array.unsqueeze(1), 1, 2)
    delta_v_array = tile(delta_v_array.unsqueeze(1), 1, 2)
    # output is of shape [timestep, (x,y, vx, vy)] needs to be recast into correct shape before use
    return torch.sqrt(torch.cat((delta_x_array, delta_v_array), dim = 1))

def KL_between_blocks(prob_list, num_atoms, eps=1e-16):
    # Return a list of the mutual information between every block pair
    KL_list = []
    for i in range(len(prob_list)):
        for j in range(len(prob_list)):
            if i != j:
                KL = prob_list[i] *( torch.log(prob_list[i] + eps) - torch.log(prob_list[j] + eps) )
                KL_list.append( KL.sum() / (num_atoms * prob_list[i].size(0)) )
                KL = prob_list[i] *( torch.log(prob_list[i] + eps) - torch.log( true_flip(prob_list[j],-1) + eps) )
                KL_list.append( KL.sum() / (num_atoms * prob_list[i].size(0)) )  
    return KL_list


def decode_target( target, num_edge_types_list ):
    target_list = []
    base = np.prod(num_edge_types_list)
    for i in range(len(num_edge_types_list)):
        base /= num_edge_types_list[i]
        target_list.append( target//base )
        target = target % base
    return target_list

def encode_target_list( target_list, edge_types_list ):
    encoded_target = np.zeros( target_list[0].shape )
    base = 1
    for i in reversed(range(len(target_list))):
        encoded_target += base*np.array(target_list[i])
        base *= edge_types_list[i]
    return encoded_target.astype('int')

def edge_accuracy_perm_NRI_batch(preds, target, num_edge_types_list):
    # permutation edge accuracy calculator for the standard NRI model
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graphs 

    _, preds = preds.max(-1)    # returns index of max in each z_ij to reduce dim by 1

    num_edge_types = np.prod(num_edge_types_list)
    preds = np.eye(num_edge_types)[np.array(preds.cpu())]  # this is nice way to turn integers into one-hot vectors
    target = np.array(target.cpu())

    perms = [p for p in permutations(range(num_edge_types))] # list of edge type permutations
    # in the below, for each permutation of edge-types, permute preds, then take argmax to go from one-hot to integers
    # then compare to target, compute accuracy
    acc = np.array([np.mean(np.equal(target, np.argmax(preds[:,:,p], axis=-1),dtype=object)) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)
    preds_deperm = np.argmax(preds[:,:,perms[idx]], axis=-1)

    target_list = decode_target( target, num_edge_types_list )
    preds_deperm_list = decode_target( preds_deperm, num_edge_types_list )

    blocks_acc = [ np.mean(np.equal(target_list[i], preds_deperm_list[i], dtype=object),axis=-1)
                   for i in range(len(target_list)) ]
    acc = np.mean(np.equal(target, preds_deperm ,dtype=object), axis=-1)
    blocks_acc = np.swapaxes(np.array(blocks_acc),0,1)

    idx_onehot = np.eye(len(perms))[np.array(idx)]
    return acc, idx_onehot, blocks_acc

def edge_accuracy_perm_NRI(preds, targets, num_edge_types_list):
    acc_batch, perm_code_onehot, acc_blocks_batch = edge_accuracy_perm_NRI_batch(preds, targets, num_edge_types_list)
    
    acc = np.mean(acc_batch)
    acc_var = np.var(acc_batch)
    acc_blocks = np.mean(acc_blocks_batch, axis=0)
    acc_var_blocks = np.var(acc_blocks_batch, axis=0)

    return acc, perm_code_onehot, acc_blocks, acc_var, acc_var_blocks


def edge_accuracy_perm_fNRI_batch(preds_list, targets, num_edge_types_list):
    # permutation edge accuracy calculator for the fNRI model
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graphs

    target_list = [ targets[:,i,:].cpu() for i in range(targets.shape[1])]
    preds_list = [ pred.max(-1)[1].cpu() for pred in preds_list]
    preds = encode_target_list(preds_list, num_edge_types_list)
    target = encode_target_list(target_list, num_edge_types_list)

    target_list = [ np.array(t.cpu()).astype('int') for t in target_list ]

    num_edge_types = np.prod(num_edge_types_list)
    preds = np.eye(num_edge_types)[preds]     # this is nice way to turn integers into one-hot vectors

    perms = [p for p in permutations(range(num_edge_types))] # list of edge type permutations
    
    # in the below, for each permutation of edge-types, permute preds, then take argmax to go from one-hot to integers
    # then compare to target to compute accuracy
    acc = np.array([np.mean(np.equal(target, np.argmax(preds[:,:,p], axis=-1),dtype=object)) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)

    preds_deperm = np.argmax(preds[:,:,perms[idx]], axis=-1)
    preds_deperm_list = decode_target( preds_deperm, num_edge_types_list )

    blocks_acc = [ np.mean(np.equal(target_list[i], preds_deperm_list[i], dtype=object),axis=-1) 
                   for i in range(len(target_list)) ]
    acc = np.mean(np.equal(target, preds_deperm ,dtype=object), axis=-1)
    blocks_acc = np.swapaxes(np.array(blocks_acc),0,1)

    idx_onehot = np.array([0])#np.eye(len(perms))[np.array(idx)]

    return acc, idx_onehot, blocks_acc

def edge_accuracy_perm_fNRI_batch_skipfirst(preds_list, targets, num_factors):
    # permutation edge accuracy calculator for the fNRI model when using skip-first argument 
    # and all factor graphs have two edge types
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graphs

    targets = np.swapaxes(np.array(targets.cpu()),1,2)
    preds = torch.cat( [ torch.unsqueeze(pred.max(-1)[1],-1) for pred in preds_list], -1 )
    preds = np.array(preds.cpu())
    perms = [p for p in permutations(range(num_factors))]

    acc = np.array([np.mean(  np.sum(np.equal(targets, preds[:,:,p],dtype=object),axis=-1)==num_factors  ) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)

    preds_deperm = preds[:,:,perms[idx]]
    blocks_acc = np.mean(np.equal(targets, preds_deperm, dtype=object),axis=1)
    acc = np.mean(  np.sum(np.equal(targets, preds_deperm,dtype=object),axis=-1)==num_factors, axis=-1)

    idx_onehot = np.eye(len(perms))[np.array(idx)]

    return acc, idx_onehot, blocks_acc


def edge_accuracy_perm_fNRI(preds_list, targets, num_edge_types_list, skip_first=False):

    if skip_first and all(e == 2 for e in num_edge_types_list):
        acc_batch, perm_code_onehot, acc_blocks_batch = edge_accuracy_perm_fNRI_batch_skipfirst(preds_list, targets, len(num_edge_types_list))
    else:
        acc_batch, perm_code_onehot, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(preds_list, targets, num_edge_types_list)
    
    acc = np.mean(acc_batch)
    acc_var = np.var(acc_batch)
    acc_blocks = np.mean(acc_blocks_batch, axis=0)
    acc_var_blocks = np.var(acc_blocks_batch, axis=0)

    return acc, perm_code_onehot, acc_blocks, acc_var, acc_var_blocks

def edge_accuracy_perm_sigmoid_batch(preds, targets):
    # permutation edge accuracy calculator for the sigmoid model
    # return the maximum accuracy of the batch over the permutations of the edge labels
    # also returns a one-hot encoding of the number which represents this permutation
    # also returns the accuracies for the individual factor graph_list

    targets = np.swapaxes(np.array(targets.cpu()),1,2)
    preds = np.array(preds.cpu().detach())
    preds = np.rint(preds).astype('int')
    num_factors = targets.shape[-1]
    perms = [p for p in permutations(range(num_factors))] # list of edge type permutations

    # in the below, for each permutation of edge-types, permute preds, then take argmax to go from one-hot to integers
    # then compare to target to compute accuracy
    acc = np.array([np.mean(  np.sum(np.equal(targets, preds[:,:,p],dtype=object),axis=-1)==num_factors  ) for p in perms])
    max_acc, idx = np.amax(acc), np.argmax(acc)

    preds_deperm = preds[:,:,perms[idx]]
    blocks_acc = np.mean(np.equal(targets, preds_deperm, dtype=object),axis=1)
    acc = np.mean( np.sum(np.equal(targets, preds_deperm,dtype=object),axis=-1)==num_factors, axis=-1)

    idx_onehot = np.eye(len(perms))[np.array(idx)]
    return acc, idx_onehot, blocks_acc


def edge_accuracy_perm_sigmoid(preds, targets):
    acc_batch, perm_code_onehot, acc_blocks_batch= edge_accuracy_perm_sigmoid_batch(preds, targets)
    
    acc = np.mean(acc_batch)
    acc_var = np.var(acc_batch)
    acc_blocks = np.mean(acc_blocks_batch, axis=0)
    acc_var_blocks = np.var(acc_blocks_batch, axis=0)

    return acc, perm_code_onehot, acc_blocks, acc_var, acc_var_blocks


def initsigma(batchsize, time, anisotropic, noofparticles, initvar, ani_dims = 4):
    """
    initialises a Tensor of sigma values of size [batchsize, no. of particles, time,no. of axes (isotropic = 1,
    anisotropic = 4 (or 2 for semiisotropic))]
    :param batchsize: size of the batch dimension. Int
    :param time: size of the timestep dimension. Int
    :param anisotropic: if it is anisotropic or not. Boolean
    :param noofparticles: size of the particles dimension. Int
    :param initvar: value of the initial variance. Float
    :param ani_dims: dimensions that the anisotropic should have (default = 4)
    :return: tensor of dimension [batchsize, noofparticles, time, ani_dims(or 1 if anisotropic = False)] with initvar at
    each point
    """
    if anisotropic:
        ani = ani_dims
    else:
        ani = 1
    # create numpy array of appropriate size
    sigma = np.zeros((batchsize, noofparticles, time, ani), dtype = np.float32)
    for i in range(len(sigma)):
        for j in range(len(sigma[i])):
            for l in range(len(sigma[i][j])):
                for m in range(len(sigma[i][j][l])):
                    sigma[i][j][l][m] = np.float32(initvar)
    return torch.from_numpy(sigma)

def tile(a, dim, n_tile):
    """"
    Taken from: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3

    tiles the data along dimension dim
    :param a: tensor to be tiled
    :param dim: dimension along which the tiling is to be done
    :param n_tile: number of times the tiling should be done along dimension dim
    :returns: tiled tensor
    """
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    if a.is_cuda:
        order_index = order_index.cuda()
    return torch.index_select(a, dim, order_index)

# takes a tensor and applies a softplus (y=ln(1+e^beta*x)/beta) function to each of the components
def softplus(tensor, beta = 1.0):
    return F.softplus(Variable(tensor * beta)).data / beta

# inverse of temperature dependent softplus function above
def inversesoftplus(x, beta = 1.0):
    intermediate = abs(1-np.exp(beta * x))
    return np.log(intermediate) / beta

# returns a gaussian with mean and sigma
def gaussian(x, amplitude,  mean, sigma):
    return amplitude * np.exp(-(x-mean) ** 2 / (2 * sigma ** 2))

# returns a lorentzian
def lorentzian(x, amplitude, mean, gamma):
    return amplitude * gamma ** 2 / (gamma ** 2 + (x - mean) ** 2)

# calculates sigmoid
def sigmoid(epochs, epochs_mid, temperature):
    return 1/(1+np.exp(-(epochs-epochs_mid)/temperature))

# calculates exponential
def exp(x, amp, alpha, const):
    return amp*0.0000001 * np.exp(alpha * x) + const


class NormalInverseWishart(object):
    """implementation based on formulae found in:
    https://www.cs.cmu.edu/~epxing/Class/10701-12f/recitation/mle_map_examples.pdf
    Note that in this implementation 1/beta -> beta as the formulae are easier with this change
    the Normal Inverse Wishart is the conjugate prior to the multivariate
    Normal distribution for unknown mean and covariance matrix.
    Parameters:
    :param mu: tensor of coords: [batchsize, particle, timestep, (x,y) or (v_x,v_y)]
    :param beta: no. of samples to get mean
    :param nu: no. of samples to get covariance matrix: must be >d-1 where d = dim(mu(3))
    :param Psi: tensor of dimensionality [batchsize, particle, timestep, 2, 2]
    """

    def __init__(self, mu, beta, nu, psi):
        self.mu = mu
        self.beta = beta
        self.nu = nu
        self.psi = psi
        self.inv_psi = torch.inverse(psi)

    def getterms(self):
        """
        :return: all the parameters of the distribution
        """
        return(self.mu , self.beta, self.nu, self.psi)


    def posterior(self, observation):
        """
        :param observation: must have the same dimensions as mu except in the timestep dimension. The sampled
        observation of the distribution. Valid for all slicing except dont_split_data slicing.
        :return: The posterior distribution using the current distribution as a prior and observation as the values of
        of the observed distribution
        """
        # data is a single vector => n =1
        timesteps = observation.shape[2]
        muprime = (self.beta * self.mu[:,:, -timesteps:, :] + observation) / (self.beta + 1)
        betaprime = self.beta + 1
        nuprime = self.nu + 2
        mean_error = observation - self.mu[:,:,-timesteps:,:]
        mean_error_T = mean_error.unsqueeze(4)
        mean_error = mean_error.unsqueeze(3)
        psiprime = self.psi[:,:,-timesteps:,:,:] + (self.beta * torch.matmul(mean_error_T, mean_error)) / (self.beta + 1)
        return NormalInverseWishart(muprime, betaprime, nuprime, psiprime)


def batch_diagonal(input):
    # Taken from https://github.com/pytorch/pytorch/issues/12160


    # idea from here: https://discuss.pytorch.org/t/batch-of-diagonal-matrix/13560
    # batches a stack of vectors (batch x N) -> a stack of diagonal matrices (batch x N x N)
    # works in  2D -> 3D, should also work in higher dimensions
    # make a zero matrix, which duplicates the last dim of input
    dims = [input.size(i) for i in torch.arange(input.dim())]
    dims.append(dims[-1])
    output = torch.zeros(dims)
    # stride across the first dimensions, add one to get the diagonal of the last dimension
    strides = [output.stride(i) for i in torch.arange(input.dim() - 1 )]
    strides.append(output.size(-1) + 1)
    # stride and copy the imput to the diagonal
    output.as_strided(input.size(), strides ).copy_(input)
    return output


def getpriorcovmat(target, sigmatarget, nu = 6):
    """

    :param target: target data of size [batch, particles, timesteps, 2]
    :param sigmatarget: prior sigma tensor of dimensions [1, 1, timesteps, 2]
    :param nu: number of samples used to get an estimate for the covariance matrix (default =6). Should be > 1 - same
    as nu hyperparameter in normal-Inverse-Wishart distribution
    :return: estimate for the prior covariance matrix
    """
    # covariance matrix
    covmat = torch.matmul(batch_diagonal(sigmatarget), batch_diagonal(sigmatarget))
    # sample nu times from the distribution
    sample = np.empty((target.size(0), target.size(1), target.size(2), nu, target.size(3)))
    for i in range(target.size(0)):
        for j in range(target.size(1)):
            for k in range(target.size(2)):
                samples = np.random.multivariate_normal(target.detach().cpu().numpy()[i][j][k], covmat.detach().cpu().numpy()[0][0][k], size = nu)
                sample[i][j][k] = samples
    sample = sample.astype(np.single)
    sample = torch.from_numpy(sample)
    if target.is_cuda:
        sample = sample.cuda()
    target = tile(target.unsqueeze(dim = 3), dim = 3, n_tile = nu)
    # get a measure for the covariance matrix from the samples as 1/nu-1 * sum((x_i-xbar)^T(x_i-xbar))
    covmatapprox = torch.matmul((sample - target).unsqueeze(5), (sample - target).unsqueeze(4)).sum(dim = 3)/(nu -1)
    return covmatapprox

def getpriordist(target, sigmatarget, nu = 6):
    """

    :param target: target data of size [batch, particles, timesteps, 2]
    :param sigmatarget: prior sigma tensor of dimensions [batch, particles, timesteps, 2]
    :param nu: number of samples used to get an estimate for the covariance matrix (default =6). Should be > 1 - same
    as nu hyperparameter in normal-Inverse-Wishart distribution
    :return: Prior distribution for this batch
    """
    convmat = getpriorcovmat(target, sigmatarget, nu)
    psi = nu * convmat
    # beta = no. of samples to get the mean. In our case this is always 1
    beta = 1
    return NormalInverseWishart(target, beta, nu, psi)


def nll_second_term_loss(dim_preds, dim_target, dim_covmat, dim_direction, beta):
    """
    :param dim_preds: The predictions along the dimension of interest, output of NN. Size [batch, particles, timesteps, 2]
    :param dim_target: The target along the dimension of interest, mu_dim. Size [batch, particles, timesteps, 2]
    :param dim_covmat: The covariance matrix along the dimensions of interest. Size [batch, particles, timesteps, 2, 2]
    :param dim_direction: The velocity/acceleration direction. Size [batch, particles, timesteps, 2]
    :param beta: the value of beta of the posterior distribution. Type float and beta > 0
    :return: neg_log_loss: loss term for (x-mu)^T(1/beta Sigma ^-1)(x-mu) term
    """
    # t = time.time()
    dimnorm = dim_direction.norm(p=2, dim=3, keepdim=True)
    normaliseddim = dim_direction.div(dimnorm.expand_as(dim_direction))
    # 1/sqrt(2) - isotropic => direction unimportant. chosen here to improve efficiency
    normaliseddim[torch.isnan(normaliseddim)] = np.power(1 / 2, 1 / 2)
    # ti = time.time()
    if beta < pow(10, -3):
        beta = pow(10, -3)
    # gets scaled covariance matrix
    dim_covmat = dim_covmat / beta
    dim_covmat = dim_covmat.reshape(dim_covmat.size(0), dim_covmat.size(1), dim_covmat.size(2), 4)
    indices_sigma = torch.LongTensor([0, 3])
    indices_diag_1 = torch.LongTensor([1, 2])
    if dim_preds.is_cuda:
        indices_sigma, indices_diag_1 = indices_sigma.cuda(), indices_diag_1.cuda()
    # extract variance
    var_pos = torch.index_select(dim_covmat, 3, indices_sigma)
    offdiag_pos = torch.index_select(dim_covmat, 3, indices_diag_1)
    # ensures variance does not go to 0
    if (torch.min(var_pos) < pow(10, -14)):
        accuracy = np.full((var_pos.size(0), var_pos.size(1), var_pos.size(2), var_pos.size(3)),
                           pow(10, -14), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if dim_preds.is_cuda:
            accuracy = accuracy.cuda()
        var_pos = torch.max(var_pos, accuracy)
    indices_1 = torch.LongTensor([0])
    indices_2 = torch.LongTensor([1])
    if dim_preds.is_cuda:
        indices_1, indices_2 = indices_1.cuda(), indices_2.cuda()
    # recasts the variance into desired form
    variance_pos = torch.cat((torch.cat((torch.index_select(var_pos, 3, indices_1), offdiag_pos), 3),
                              torch.index_select(var_pos, 3, indices_2)), 3)
    dim_covmat = variance_pos.reshape(variance_pos.size(0), variance_pos.size(1), variance_pos.size(2), 2, 2)
    # inverse of the covariance matrix
    inversevariance = dim_covmat.inverse()
    # if np.isnan(np.sum(inversevariance.cpu().detach().numpy())):
    #     print("Some values from variance are nan")
    # need position and velocity differences in (x,y) coordinates
    differences = dim_preds - dim_target
    differences = differences.unsqueeze(4)
    # print('getdifferences: {:.1f}s'.format(time.time() - ti))
    # the matrix multiplication for multivariate case can be thought of as taking a projection of the error vector
    # along the parallel and perpendicular velocity/acceleration directions and multiplying by 1/sigma^2 along that
    # direction. This follows directly from the fact the rotation matrix is orthogonal.
    # multime = time.time()
    # surprisingly it is more efficient to calculate the perpendicular term by considering
    # (position_differences - (position_differences.v||)v||).vperp to get the position differences in the perpendicular
    # direction than using rotation (x,y) -> (-y,x) as the triple for loop is inefficient. about 100x faster this way
    # and almost as fast as isotropic
    errorvectorparalleltov = torch.matmul(normaliseddim.unsqueeze(3), differences)
    parallelterm = torch.matmul(normaliseddim.unsqueeze(4), errorvectorparalleltov)
    perpterm = (differences - parallelterm).squeeze()
    perpnorm = perpterm.norm(p=2, dim=3, keepdim=True)
    # NaN can occur when dividing by 0 (see comment below) but the problem with replacing NaN after the division is that
    # the NaN carries through anyway - the function that the system is backtracking through keeps the NaN =
    # therefore leads to NaN errors on the second pass of the function - replacing the 0's before division solves this
    # issue.
    if (torch.min(perpnorm) < pow(10, -7)):
        accuracy = np.full((perpnorm.size(0), perpnorm.size(1), perpnorm.size(2), perpnorm.size(3)),
                           pow(10, -7), dtype=np.float32)
        accuracy = torch.from_numpy(accuracy)
        if dim_preds.is_cuda:
            accuracy = accuracy.cuda()
        perpnorm = torch.max(perpnorm, accuracy)
    normalisedperp = perpterm.div(perpnorm.expand_as(perpterm))
    # NaN can occur when perpterm is 0, this means that preds-true = (preds-true).v|| v||
    # i.e. error entirely in parallel direction and no error perpendicular: so we set these terms to 0
    # normalisedperp[torch.isnan(normalisedperp)] = 0
    errorvectorperptov = torch.matmul(perpterm.unsqueeze(3), normalisedperp.unsqueeze(4)).squeeze()
    errorvectorparalleltov = errorvectorparalleltov.squeeze()
    # errorvectorperptov = torch.matmul(velperp.unsqueeze(3), position_differences).squeeze()

    indices_vpar = torch.LongTensor([0])
    indices_vperp = torch.LongTensor([1])
    # print('matrixmult: {:.1f}s'.format(time.time() - multime))
    if dim_preds.is_cuda:
        indices_vpar, indices_vperp = indices_vpar.cuda(), indices_vperp.cuda()
    # t = time.time()
    losscomponentparalleltov = (errorvectorparalleltov ** 2) * torch.index_select(
        torch.index_select(inversevariance, 3, indices_vpar), 4, indices_vpar).squeeze()
    losscomponentperptov = (errorvectorperptov ** 2) * torch.index_select(
        torch.index_select(inversevariance, 3, indices_vperp), 4, indices_vperp).squeeze()
    neg_log_loss = losscomponentparalleltov + losscomponentperptov
    return neg_log_loss


def nll_Normal_Inverse_WishartLoss(preds, sigma, accel, vel, prior_pos, prior_vel):
    """
    Loss function derived: https://www.cs.cmu.edu/~epxing/Class/10701-12f/recitation/mle_map_examples.pdf
    The posterior distribution is used to find a loss function that needs to be minimised

    Parameters:
        preds = prediction values from NN of size [batch, particles, timesteps, (x,y,v_x,v_y)]
        sigma = values of uncertainty of size [batch, particles, timesteps, 4]
        accel = gives direction of acceleration of each prediction data point. Size [batch, particles, timesteps, 2]
        vel = gives direction of velocity of each prediction data point. Size [batch, particles, timesteps, 2]
        prior_pos = The prior distribution on the positions. Here assumed to be NormalInverseWishart
        prior_vel = The prior distribution on the velocities. Here assumed to be NormalInverseWishart
        target is implicitly in prior
    """
    # 2 dimensional terms for (x,y) and (vx,vy)
    d = 2
    # separate the positions and velocities
    indices_pos = torch.LongTensor([0,1])
    indices_vel = torch.LongTensor([2,3])
    if preds.is_cuda:
        indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
    pos_preds = torch.index_select(preds, 3, indices_pos)
    vel_preds = torch.index_select(preds, 3, indices_vel)
    pos_sigma = torch.index_select(sigma, 3, indices_pos)
    vel_sigma = torch.index_select(sigma, 3, indices_vel)
    # get the posterior distribution
    pos_posterior = prior_pos.posterior(pos_preds)
    vel_posterior = prior_vel.posterior(vel_preds)

    mu_pos, beta_pos, nu_pos, psi_pos = pos_posterior.getterms()
    mu_vel, beta_vel, nu_vel, psi_vel = vel_posterior.getterms()
    # get the covariance matrices from the NN output
    pos_covmat = torch.matmul(batch_diagonal(pos_sigma), batch_diagonal(pos_sigma))
    vel_covmat = torch.matmul(batch_diagonal(vel_sigma), batch_diagonal(vel_sigma))

    if preds.is_cuda:
        pos_covmat , vel_covmat = pos_covmat.cuda(), vel_covmat.cuda()

    # calculate the loss function given in the reference
    loss_term_1_pos = (nu_pos + d + 2) * torch.log(pos_covmat.det())
    loss_term_1_vel = (nu_vel + d + 2) * torch.log(vel_covmat.det())

    inv_pos_covmat = torch.inverse(pos_covmat)
    inv_vel_covmat = torch.inverse(vel_covmat)

    #  to do- there must be a better way to batch trace
    loss_term_3_pos = torch.matmul(psi_pos, inv_pos_covmat)
    loss_term_3_pos = loss_term_3_pos[:,:,:,0,0] + loss_term_3_pos[:,:,:,1,1]
    loss_term_3_vel = torch.matmul(psi_vel, inv_vel_covmat)
    loss_term_3_vel = loss_term_3_vel[:,:,:,0,0] + loss_term_3_vel[:,:,:,1,1]

    loss_term_2_pos = nll_second_term_loss(pos_preds, mu_pos, pos_covmat, vel, beta_pos)
    loss_term_2_vel = nll_second_term_loss(vel_preds, mu_vel, vel_covmat, accel, beta_vel)

    loss = loss_term_1_pos + loss_term_1_vel + loss_term_2_pos + loss_term_2_vel + loss_term_3_pos + loss_term_3_vel
    return loss.sum() / (preds.size(0) * preds.size(1)), ((loss).sum(dim=1)/preds.size(1)).var()

