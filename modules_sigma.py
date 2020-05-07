'''
This code is based on https://github.com/ekwebb/fNRI which in turn is based on https://github.com/ethanfetaya/NRI
(MIT licence)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import time
import utils

from torch.autograd import Variable
from utils import my_softmax, get_offdiag_indices, gumbel_softmax, softplus

_EPS = 1e-10

class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):

        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLPDecoder_multi(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, edge_types_list, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, init_type='default'):
        super(MLPDecoder_multi, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first = skip_first
        self.edge_types = edge_types
        self.edge_types_list = edge_types_list

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'default' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for decoder weight initialization')

        if self.init_type != 'default':
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data,gain=0.000001)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data,gain=0.000001)
                #m.bias.data.fill_(0.1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        # size of [batchsize, no. of particles (N), N(N-1), no. of phase space components (x,y,vx,vy, sigma)]
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        # size of [batchsize, no. of particles (N), N(N-1), 2 * no. of phase space components (x,y,vx,vy, sigma)], concatinates the components [x_i||x_j]
        pre_msg = torch.cat([receivers, senders], dim=-1)
        # size of [batchsize, no. of particles (N), N(N-1), no. of hidden layers]
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        # non_null_idxs = list of indices of edge types which as non null (i.e. edges over which messages can be passed)
        non_null_idxs = list(range(self.edge_types))
        if self.skip_first:
            # if skip_first is True, the first edge type in each factor block is null
            edge = 0
            for k in self.edge_types_list:
                non_null_idxs.remove(edge)
                edge += k

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        # f^k_e
        for i in non_null_idxs:
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
        # Predict position/velocity difference
        return single_timestep_inputs + pred, pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, sigma, sigmavariable, anisotropic, beta ,pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        if sigmavariable:
            # concatinate the sigma component to the tensor making each point have components (x,y,vx,vy,{sigma})
            inputs = torch.cat((inputs,sigma), dim = 3)
            inputs = inputs.transpose(1, 2).contiguous()

            sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                     rel_type.size(2)]
            rel_type = rel_type.unsqueeze(1).expand(sizes)

            time_steps = inputs.size(1)
            assert (pred_steps <= time_steps)
            preds = []
            accelerations = []
            velocities = []
            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred = inputs[:, 0::pred_steps, :, :]
            curr_rel_type = rel_type[:, 0::pred_steps, :, :]
            # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

            # Run n prediction steps, gets last predictions and the changes in values- will be used to calculate the acceleration
            for step in range(0, pred_steps):
                last_pred, differences = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                     curr_rel_type)
                preds.append(last_pred)
                # index = torch.LongTensor([2,3])
                # index_vel = torch.LongTensor([0,1])
                # if inputs.is_cuda:
                #     index, index_vel = index.cuda(), index_vel.cuda()
                # acceleration = torch.index_select(differences, 3, index)
                # accelerations.append(acceleration)
                # velocity = torch.index_select(differences, 3, index_vel)
                # velocities.append(velocity)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]

            # accsizes = [accelerations[0].size(0), accelerations[0].size(1) * pred_steps,
            #          accelerations[0].size(2), accelerations[0].size(3)]
            #
            # velsizes = [velocities[0].size(0), velocities[0].size(1) * pred_steps,
            #             velocities[0].size(2), velocities[0].size(3)]

            output = Variable(torch.zeros(sizes))
            # get acceleration direction in (x,y) basis
            # acc = Variable(torch.zeros(accsizes))
            # vel = Variable(torch.zeros(velsizes))
            if inputs.is_cuda:
                output = output.cuda()
                # acc = acc.cuda()
                # vel  =vel.cuda()
            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
                # acc[:, i::pred_steps, :, :] = accelerations[i]
                # vel[:, i::pred_steps, :, :] = velocities[i]'
            # here will need to take out the new predicted sigma values from the tensor.
            # t = time.time()
            future = output[:,1:, :,:]
            current = output[:,:output.size()[1]-1, :, :]
            acc = future[:,:,:,2:4]- current[:,:,:,2:4]
            vel = future[:,:,:,0:2]- current[:,:,:,0:2]
            accelzero = torch.zeros(acc.size()[0], 1, acc.size()[2],acc.size()[3], dtype = torch.float)
            velzero = torch.zeros(vel.size()[0], 1, vel.size()[2],vel.size()[3], dtype = torch.float)
            if inputs.is_cuda:
                accelzero, velzero = accelzero.cuda(), velzero.cuda()
            # print('arraygenerationtime: {:.1f}s'.format(time.time() - t))
            # t = time.time()
            acc = torch.cat((accelzero, acc), dim = 1)
            vel = torch.cat((velzero, vel), dim = 1)
            # print('arrayconcattime: {:.1f}s'.format(time.time() - t))
            pred_all = output[:, :(inputs.size(1) - 1), :, :]
            accel = acc[:, :(inputs.size(1)-1), :, :]
            velocity = vel[:, :(inputs.size(1) - 1), :, :]
            indices = (torch.from_numpy(np.arange(4,list(pred_all.size())[3]))).type(torch.LongTensor)
            if inputs.is_cuda:
                indices = indices.cuda()
            sigma_1 = torch.index_select(pred_all, 3, indices)
            sigma_1 = sigma_1.transpose(1, 2).contiguous()
            # sigma must be >=0 therefore use a softplus function to confine values to positive  values
            sigma_1 = softplus(sigma_1, beta)
            indices = torch.tensor([0,1,2,3])
            if inputs.is_cuda:
                indices = indices.cuda()
            pred_all = torch.index_select(pred_all, 3, indices)
        else:
            inputs = inputs.transpose(1, 2).contiguous()

            sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                     rel_type.size(2)]
            rel_type = rel_type.unsqueeze(1).expand(sizes)

            time_steps = inputs.size(1)
            assert (pred_steps <= time_steps)
            preds = []

            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred = inputs[:, 0::pred_steps, :, :]
            curr_rel_type = rel_type[:, 0::pred_steps, :, :]
            # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

            # Run n prediction steps
            for step in range(0, pred_steps):
                last_pred, differences = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                     curr_rel_type)
                preds.append(last_pred)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]

            output = Variable(torch.zeros(sizes))
            if inputs.is_cuda:
                output = output.cuda()

            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
            # no need for accel here
            accel = torch.ones(1,1,1)
            velocity = torch.ones(1,1,1)
            if inputs.is_cuda:
                accel, velocity = accel.cuda(), velocity.cuda()
            pred_all = output[:, :(inputs.size(1) - 1), :, :]
            sigma_1 = sigma

        return pred_all.transpose(1, 2).contiguous(), sigma_1, accel.transpose(1,2).contiguous(), velocity.transpose(1,2).contiguous()

# 3 layer decoder instead of 2
class MLPDecoder_multi_threelayers(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, edge_types, edge_types_list, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, init_type='default'):
        super(MLPDecoder_multi_threelayers, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_hid) for _ in range(edge_types)])
        self.msg_fc3 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first = skip_first
        self.edge_types = edge_types
        self.edge_types_list = edge_types_list

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid,n_hid)
        self.out_fc4 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'default' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for decoder weight initialization')

        if self.init_type != 'default':
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data,gain=0.000001)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data,gain=0.000001)
                #m.bias.data.fill_(0.1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        # size of [batchsize, no. of particles (N), N(N-1), no. of phase space components (x,y,vx,vy, sigma)]
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        # size of [batchsize, no. of particles (N), N(N-1), 2 * no. of phase space components (x,y,vx,vy, sigma)], concatinates the components [x_i||x_j]
        pre_msg = torch.cat([receivers, senders], dim=-1)
        # size of [batchsize, no. of particles (N), N(N-1), no. of hidden layers]
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        # non_null_idxs = list of indices of edge types which as non null (i.e. edges over which messages can be passed)
        non_null_idxs = list(range(self.edge_types))
        if self.skip_first:
            # if skip_first is True, the first edge type in each factor block is null
            edge = 0
            for k in self.edge_types_list:
                non_null_idxs.remove(edge)
                edge += k

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        # f^k_e
        for i in non_null_idxs:
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc3[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc3(pred)), p=self.dropout_prob)
        pred = self.out_fc4(pred)
        # Predict position/velocity difference
        return single_timestep_inputs + pred, pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, sigma, sigmavariable, anisotropic, beta ,pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        if sigmavariable:
            # concatinate the sigma component to the tensor making each point have components (x,y,vx,vy,{sigma})

            inputs = torch.cat((inputs,sigma), dim = 3)
            inputs = inputs.transpose(1, 2).contiguous()

            sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                     rel_type.size(2)]
            rel_type = rel_type.unsqueeze(1).expand(sizes)

            time_steps = inputs.size(1)
            assert (pred_steps <= time_steps)
            preds = []
            accelerations = []
            velocities = []
            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred = inputs[:, 0::pred_steps, :, :]
            curr_rel_type = rel_type[:, 0::pred_steps, :, :]
            # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

            # Run n prediction steps, gets last predictions and the changes in values- will be used to calculate the acceleration
            for step in range(0, pred_steps):
                last_pred, differences = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                     curr_rel_type)
                preds.append(last_pred)
                # index = torch.LongTensor([2,3])
                # index_vel = torch.LongTensor([0,1])
                # if inputs.is_cuda:
                #     index, index_vel = index.cuda(), index_vel.cuda()
                # acceleration = torch.index_select(differences, 3, index)
                # accelerations.append(acceleration)
                # velocity = torch.index_select(differences, 3, index_vel)
                # velocities.append(velocity)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]

            # accsizes = [accelerations[0].size(0), accelerations[0].size(1) * pred_steps,
            #          accelerations[0].size(2), accelerations[0].size(3)]
            #
            # velsizes = [velocities[0].size(0), velocities[0].size(1) * pred_steps,
            #             velocities[0].size(2), velocities[0].size(3)]

            output = Variable(torch.zeros(sizes))
            # get acceleration direction in (x,y) basis
            # acc = Variable(torch.zeros(accsizes))
            # vel = Variable(torch.zeros(velsizes))
            if inputs.is_cuda:
                output = output.cuda()
                # acc = acc.cuda()
                # vel  =vel.cuda()
            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
                # acc[:, i::pred_steps, :, :] = accelerations[i]
                # vel[:, i::pred_steps, :, :] = velocities[i]'
            # here will need to take out the new predicted sigma values from the tensor.
            # t = time.time()
            future = output[:,1:, :,:]
            current = output[:,:output.size()[1]-1, :, :]
            acc = future[:,:,:,2:4]- current[:,:,:,2:4]
            vel = future[:,:,:,0:2]- current[:,:,:,0:2]
            accelzero = torch.zeros(acc.size()[0], 1, acc.size()[2],acc.size()[3], dtype = torch.float)
            velzero = torch.zeros(vel.size()[0], 1, vel.size()[2],vel.size()[3], dtype = torch.float)
            if inputs.is_cuda:
                accelzero, velzero = accelzero.cuda(), velzero.cuda()
            # print('arraygenerationtime: {:.1f}s'.format(time.time() - t))
            # t = time.time()
            acc = torch.cat((accelzero, acc), dim = 1)
            vel = torch.cat((velzero, vel), dim = 1)
            # print('arrayconcattime: {:.1f}s'.format(time.time() - t))
            pred_all = output[:, :(inputs.size(1) - 1), :, :]
            accel = acc[:, :(inputs.size(1)-1), :, :]
            velocity = vel[:, :(inputs.size(1) - 1), :, :]
            indices = (torch.from_numpy(np.arange(4,list(pred_all.size())[3]))).type(torch.LongTensor)
            if inputs.is_cuda:
                indices = indices.cuda()
            sigma_1 = torch.index_select(pred_all, 3, indices)
            sigma_1 = sigma_1.transpose(1, 2).contiguous()
            # sigma must be >=0 therefore use a softplus function to confine values to positive  values
            sigma_1 = softplus(sigma_1, beta)
            indices = torch.tensor([0,1,2,3])
            if inputs.is_cuda:
                indices = indices.cuda()
            pred_all = torch.index_select(pred_all, 3, indices)
        else:
            inputs = inputs.transpose(1, 2).contiguous()

            sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                     rel_type.size(2)]
            rel_type = rel_type.unsqueeze(1).expand(sizes)

            time_steps = inputs.size(1)
            assert (pred_steps <= time_steps)
            preds = []

            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred = inputs[:, 0::pred_steps, :, :]
            curr_rel_type = rel_type[:, 0::pred_steps, :, :]
            # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

            # Run n prediction steps
            for step in range(0, pred_steps):
                last_pred, differences = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                     curr_rel_type)
                preds.append(last_pred)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]

            output = Variable(torch.zeros(sizes))
            if inputs.is_cuda:
                output = output.cuda()

            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
            # no need for accel here
            accel = torch.ones(1,1,1)
            velocity = torch.ones(1,1,1)
            if inputs.is_cuda:
                accel, velocity = accel.cuda(), velocity.cuda()
            pred_all = output[:, :(inputs.size(1) - 1), :, :]
            sigma_1 = sigma

        return pred_all.transpose(1, 2).contiguous(), sigma_1, accel.transpose(1,2).contiguous(), velocity.transpose(1,2).contiguous()


class MLPDecoder_multi_randomfeatures(nn.Module):
    """MLP decoder module.
    With added random features as suggested in  arXiv:2002.03155 [cs.LG] this could help with structural issues as
    suggested in that paper"""

    def __init__(self, n_in_node, edge_types, edge_types_list, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, init_type='default'):
        super(MLPDecoder_multi_randomfeatures, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(edge_types)])
        self.msg_out_shape = msg_out
        self.skip_first = skip_first
        self.edge_types = edge_types
        self.edge_types_list = edge_types_list

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'default' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for decoder weight initialization')

        if self.init_type != 'default':
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data,gain=0.000001)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data,gain=0.000001)
                #m.bias.data.fill_(0.1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        # size of [batchsize, no. of particles (N), N(N-1), no. of phase space components (x,y,vx,vy, sigma)]
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        # size of [batchsize, no. of particles (N), N(N-1), 2 * no. of phase space components (x,y,vx,vy, sigma)], concatinates the components [x_i||x_j]
        pre_msg = torch.cat([receivers, senders], dim=-1)
        # size of [batchsize, no. of particles (N), N(N-1), no. of hidden layers]
        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        # non_null_idxs = list of indices of edge types which as non null (i.e. edges over which messages can be passed)
        non_null_idxs = list(range(self.edge_types))
        if self.skip_first:
            # if skip_first is True, the first edge type in each factor block is null
            edge = 0
            for k in self.edge_types_list:
                non_null_idxs.remove(edge)
                edge += k

        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        # f^k_e
        for i in non_null_idxs:
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)
        # Predict position/velocity difference
        return single_timestep_inputs + pred, pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, sigma, sigmavariable, anisotropic, beta ,pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.
        if sigmavariable:
            # concatinate the sigma component to the tensor making each point have components (x,y,vx,vy,{sigma})
            inputs = torch.cat((inputs,sigma), dim = 3)
            #### generate and concatinate a random tensor to the inputs- this will act as the label of the nodes ####
            #### from the  	arXiv:2002.03155 [cs.LG] paper we will implement Algorithm 1 using the normal dist   ####
            #### This will generate a rGIN as they suggested                                                     ####
            random_feature_label = torch.normal(0,0.5,size=(sigma.size(0), sigma.size(1), sigma.size(2),1))
            if inputs.is_cuda:
                random_feature_label = random_feature_label.cuda()
            inputs = torch.cat((inputs, random_feature_label), dim=3)

            inputs = inputs.transpose(1, 2).contiguous()

            sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                     rel_type.size(2)]
            rel_type = rel_type.unsqueeze(1).expand(sizes)

            time_steps = inputs.size(1)
            assert (pred_steps <= time_steps)
            preds = []
            accelerations = []
            velocities = []
            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred = inputs[:, 0::pred_steps, :, :]
            curr_rel_type = rel_type[:, 0::pred_steps, :, :]
            # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

            # Run n prediction steps, gets last predictions and the changes in values- will be used to calculate the acceleration
            for step in range(0, pred_steps):
                last_pred, differences = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                     curr_rel_type)
                preds.append(last_pred)
                # index = torch.LongTensor([2,3])
                # index_vel = torch.LongTensor([0,1])
                # if inputs.is_cuda:
                #     index, index_vel = index.cuda(), index_vel.cuda()
                # acceleration = torch.index_select(differences, 3, index)
                # accelerations.append(acceleration)
                # velocity = torch.index_select(differences, 3, index_vel)
                # velocities.append(velocity)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]

            # accsizes = [accelerations[0].size(0), accelerations[0].size(1) * pred_steps,
            #          accelerations[0].size(2), accelerations[0].size(3)]
            #
            # velsizes = [velocities[0].size(0), velocities[0].size(1) * pred_steps,
            #             velocities[0].size(2), velocities[0].size(3)]

            output = Variable(torch.zeros(sizes))
            # get acceleration direction in (x,y) basis
            # acc = Variable(torch.zeros(accsizes))
            # vel = Variable(torch.zeros(velsizes))
            if inputs.is_cuda:
                output = output.cuda()
                # acc = acc.cuda()
                # vel  =vel.cuda()
            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
                # acc[:, i::pred_steps, :, :] = accelerations[i]
                # vel[:, i::pred_steps, :, :] = velocities[i]'
            # here will need to take out the new predicted sigma values from the tensor.
            # t = time.time()
            future = output[:,1:, :,:]
            current = output[:,:output.size()[1]-1, :, :]
            acc = future[:,:,:,2:4]- current[:,:,:,2:4]
            vel = future[:,:,:,0:2]- current[:,:,:,0:2]
            accelzero = torch.zeros(acc.size()[0], 1, acc.size()[2],acc.size()[3], dtype = torch.float)
            velzero = torch.zeros(vel.size()[0], 1, vel.size()[2],vel.size()[3], dtype = torch.float)
            if inputs.is_cuda:
                accelzero, velzero = accelzero.cuda(), velzero.cuda()
            # print('arraygenerationtime: {:.1f}s'.format(time.time() - t))
            # t = time.time()
            acc = torch.cat((accelzero, acc), dim = 1)
            vel = torch.cat((velzero, vel), dim = 1)
            # print('arrayconcattime: {:.1f}s'.format(time.time() - t))
            pred_all = output[:, :(inputs.size(1) - 1), :, :]
            accel = acc[:, :(inputs.size(1)-1), :, :]
            velocity = vel[:, :(inputs.size(1) - 1), :, :]
            # the sigma here runs from 4 -> len-1th term -> last term is just the labels
            indices = (torch.from_numpy(np.arange(4,list(pred_all.size())[3]-1))).type(torch.LongTensor)
            if inputs.is_cuda:
                indices = indices.cuda()
            sigma_1 = torch.index_select(pred_all, 3, indices)
            sigma_1 = sigma_1.transpose(1, 2).contiguous()
            # sigma must be >=0 therefore use a softplus function to confine values to positive  values
            sigma_1 = softplus(sigma_1, beta)
            indices = torch.tensor([0,1,2,3])
            if inputs.is_cuda:
                indices = indices.cuda()
            pred_all = torch.index_select(pred_all, 3, indices)
        else:
            inputs = inputs.transpose(1, 2).contiguous()

            sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                     rel_type.size(2)]
            rel_type = rel_type.unsqueeze(1).expand(sizes)

            time_steps = inputs.size(1)
            assert (pred_steps <= time_steps)
            preds = []

            # Only take n-th timesteps as starting points (n: pred_steps)
            last_pred = inputs[:, 0::pred_steps, :, :]
            curr_rel_type = rel_type[:, 0::pred_steps, :, :]
            # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

            # Run n prediction steps
            for step in range(0, pred_steps):
                last_pred, differences = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                     curr_rel_type)
                preds.append(last_pred)

            sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                     preds[0].size(2), preds[0].size(3)]

            output = Variable(torch.zeros(sizes))
            if inputs.is_cuda:
                output = output.cuda()

            # Re-assemble correct timeline
            for i in range(len(preds)):
                output[:, i::pred_steps, :, :] = preds[i]
            # no need for accel here
            accel = torch.ones(1,1,1)
            velocity = torch.ones(1,1,1)
            if inputs.is_cuda:
                accel, velocity = accel.cuda(), velocity.cuda()
            pred_all = output[:, :(inputs.size(1) - 1), :, :]
            sigma_1 = sigma

        return pred_all.transpose(1, 2).contiguous(), sigma_1, accel.transpose(1,2).contiguous(), velocity.transpose(1,2).contiguous()


class MLPDecoder_sigmoid(nn.Module):
    """MLP decoder module."""

    def __init__(self, n_in_node, num_factors, msg_hid, msg_out, n_hid,
                 do_prob=0., skip_first=False, init_type='default'):
        super(MLPDecoder_sigmoid, self).__init__()
        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_in_node, msg_hid) for _ in range(num_factors)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(msg_hid, msg_out) for _ in range(num_factors)])
        self.msg_out_shape = msg_out
        self.num_factors = num_factors

        self.out_fc1 = nn.Linear(n_in_node + msg_out, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        print('Using learned interaction net decoder.')

        self.dropout_prob = do_prob

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'default' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for decoder weight initialization')

        if self.init_type != 'default':
            self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data,gain=0.000001)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data,gain=0.000001)
                #m.bias.data.fill_(0.1)

    def single_step_forward(self, single_timestep_inputs, rel_rec, rel_send,
                            single_timestep_rel_type):

        # single_timestep_inputs has shape
        # [batch_size, num_timesteps, num_atoms, num_dims]

        # single_timestep_rel_type has shape:
        # [batch_size, num_timesteps, num_atoms*(num_atoms-1), num_edge_types]

        # Node2edge
        receivers = torch.matmul(rel_rec, single_timestep_inputs)
        senders = torch.matmul(rel_send, single_timestep_inputs)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        pre_msg.size(2), self.msg_out_shape))
        if single_timestep_inputs.is_cuda:
            all_msgs = all_msgs.cuda()


        # Run separate MLP for every edge type
        # NOTE: To exlude one edge type, simply offset range by 1
        for i in range(self.num_factors):
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * single_timestep_rel_type[:, :, :, i:i + 1]
            all_msgs += msg

        # Aggregate all msgs to receiver
        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        # Skip connection
        aug_inputs = torch.cat([single_timestep_inputs, agg_msgs], dim=-1)

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(aug_inputs)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        return single_timestep_inputs + pred

    def forward(self, inputs, rel_type, rel_rec, rel_send, pred_steps=1):
        # NOTE: Assumes that we have the same graph across all samples.

        inputs = inputs.transpose(1, 2).contiguous()

        sizes = [rel_type.size(0), inputs.size(1), rel_type.size(1),
                 rel_type.size(2)]
        rel_type = rel_type.unsqueeze(1).expand(sizes)

        time_steps = inputs.size(1)
        assert (pred_steps <= time_steps)
        preds = []

        # Only take n-th timesteps as starting points (n: pred_steps)
        last_pred = inputs[:, 0::pred_steps, :, :]
        curr_rel_type = rel_type[:, 0::pred_steps, :, :]
        # NOTE: Assumes rel_type is constant (i.e. same across all time steps).

        # Run n prediction steps
        for step in range(0, pred_steps):
            last_pred = self.single_step_forward(last_pred, rel_rec, rel_send,
                                                 curr_rel_type)
            preds.append(last_pred)

        sizes = [preds[0].size(0), preds[0].size(1) * pred_steps,
                 preds[0].size(2), preds[0].size(3)]

        output = Variable(torch.zeros(sizes))
        if inputs.is_cuda:
            output = output.cuda()

        # Re-assemble correct timeline
        for i in range(len(preds)):
            output[:, i::pred_steps, :, :] = preds[i]

        pred_all = output[:, :(inputs.size(1) - 1), :, :]

        return pred_all.transpose(1, 2).contiguous()

class MLPEncoder_multi(nn.Module):
    def __init__(self, n_in, n_hid, edge_types_list, do_prob=0., split_point=1,
                 init_type='xavier_normal', bias_init=0.0):
        super(MLPEncoder_multi, self).__init__()

        self.edge_types_list = edge_types_list
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        #print(self.mlp1.fc1.weight[0][0:5])
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        self.init_type = init_type
        if self.init_type not in [ 'xavier_normal', 'orthogonal', 'sparse' ]:
            raise ValueError('This initialization type has not been coded')
        #print('Using '+self.init_type+' for encoder weight initialization')
        self.bias_init = bias_init

        self.split_point = split_point
        if split_point == 0:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = nn.ModuleList([nn.Linear(n_hid, sum(edge_types_list))])
        elif split_point == 1:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, K) for K in edge_types_list])
        elif split_point == 2:
            self.mlp3 = nn.ModuleList([MLP(n_hid, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in edge_types_list])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, K) for K in edge_types_list])
        else:
            raise ValueError('Split point is not valid, must be 0, 1, or 2')

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif self.init_type == 'sparse':
                    nn.init.sparse_(m.weight.data, sparsity=0.1)

                if not math.isclose(self.bias_init, 0, rel_tol=1e-9):
                    m.bias.data.fill_(self.bias_init)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        if self.split_point == 0:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            return self.fc_out[0](x)
        elif self.split_point == 1:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp4[i](x)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)
        elif self.split_point == 2:
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp3[i](x)
                y = self.node2edge(y, rel_rec, rel_send)
                y = torch.cat((y, x_skip), dim=2)  # Skip connection
                y = self.mlp4[i](y)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)

class MLPEncoder_sigmoid(nn.Module):
    def __init__(self, n_in, n_hid, num_factors, do_prob=0., split_point=1):
        super(MLPEncoder_sigmoid, self).__init__()

        self.num_factors = num_factors
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)

        self.split_point = split_point
        if split_point == 0:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = nn.Linear(n_hid, num_factors)
        elif split_point == 1:
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in range(num_factors)])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, 1) for i in range(num_factors)])
        elif split_point == 2:
            self.mlp3 = nn.ModuleList([MLP(n_hid, n_hid, n_hid, do_prob) for _ in range(num_factors)])
            self.mlp4 = nn.ModuleList([MLP(n_hid * 3, n_hid, n_hid, do_prob) for _ in range(num_factors)])
            self.fc_out =  nn.ModuleList([nn.Linear(n_hid, 1) for i in range(num_factors)])
        else:
            raise ValueError('Split point is not valid, must be 0, 1, or 2')

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        if self.split_point == 0:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            return self.fc_out(x)
        elif self.split_point == 1:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            y_list = []
            for i in range(self.num_factors):
                y = self.mlp4[i](x)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)
        elif self.split_point == 2:
            y_list = []
            for i in range(self.num_factors):
                y = self.mlp3[i](x)
                y = self.node2edge(y, rel_rec, rel_send)
                y = torch.cat((y, x_skip), dim=2)  # Skip connection
                y = self.mlp4[i](y)
                y_list.append( self.fc_out[i](y) )
            return torch.cat(y_list,dim=-1)

class KalmanFilter(object):
    """Kalman filter to work on top of NN. Based on implementation in:
    https://arxiv.org/pdf/1910.14215.pdf
    R. L. Russell and C. Reale
    """
    def __init__(self, sigma_prior):
        """

        :param sigma_prior: the prior sigma value for timestep 0. [batchsize, particles, coords]
        """
        # only observe the position coordinates so the observation matrix is (I_2  0_2
        #                                                                     0_2  0_2) in 2D
        self.H = torch.eye(4)
        self.H[2,2] = 0
        self.H[3,3] = 0

        # constant velocity approximation model
        deltaT = 0.1
        self.F = torch.eye(4)
        self.F[0,2] = deltaT
        self.F[1,3] = deltaT
        if sigma_prior.is_cuda:
            self.H, self.F = self.H.cuda(), self.F.cuda()
        self.H, self.F = Variable(self.H), Variable(self.F)
        self.P_0 = utils.batch_diagonal(sigma_prior)
        self.P_0 = torch.matmul(self.P_0, self.P_0)
        if sigma_prior.is_cuda:
            self.P_0 = self.P_0.cuda()
        self.P_0 = Variable(self.P_0)

    def single_timestep(self, state_prev, pred, sigma, covar, timestep):
        common_term = torch.matmul(self.H, self.F).transpose(dim0=2, dim1=3)
        common_term = torch.matmul(covar, common_term)
        common_term = torch.matmul(self.F, common_term)
        inn_part = utils.batch_diagonal(sigma[:,:,timestep+1,:])
        if state_prev.is_cuda:
            inn_part = inn_part.cuda()
        inn_part = Variable(inn_part)
        innovation_covar = torch.matmul(self.H, common_term) + inn_part
        if (torch.min(innovation_covar) < pow(10, -14)):
            accuracy = np.full((innovation_covar.size(0), innovation_covar.size(1), innovation_covar.size(2), innovation_covar.size(3)),
                               pow(10, -14), dtype=np.float32)
            accuracy = torch.from_numpy(accuracy)
            if state_prev.is_cuda:
                accuracy = accuracy.cuda()
            accuracy = Variable(accuracy)
            innovation_covar = torch.max(innovation_covar, accuracy)
        kalman_gain = torch.matmul(common_term, torch.inverse(innovation_covar))
        update_state = pred[:,:,timestep,:] - torch.matmul(self.H, torch.matmul(self.F, state_prev.unsqueeze(-1))).squeeze()
        update_state = torch.matmul(self.F, state_prev.unsqueeze(-1)).squeeze() + torch.matmul(kalman_gain, update_state.unsqueeze(-1)).squeeze()
        update_cov = torch.matmul(self.F, torch.matmul(covar, self.F.transpose(dim0 = 2, dim1 = 3)))
        eye = torch.eye(4)
        if state_prev.is_cuda:
            eye = eye.cuda()
        eye = Variable(eye)
        update_cov = torch.matmul(eye - torch.matmul(kalman_gain, self.H),update_cov)
        return update_state, update_cov

    def kalman_filter_steps(self, target, preds, sigma):
        state = target[:,:,0,:]
        covariance = torch.empty((0))
        if target.is_cuda:
            covariance = covariance.cuda()
        covariance = Variable(covariance)
        covariance = torch.cat((covariance,self.P_0.unsqueeze(2)), dim = 2)
        state_array = torch.empty((0))
        if target.is_cuda:
            state_array = state_array.cuda()
        state_array = Variable(state_array)
        state_array = torch.cat((state_array, state.unsqueeze(2)),dim = 2)
        self.H = utils.tile(self.H.unsqueeze(0), 0, target.size(0))
        self.H = utils.tile(self.H.unsqueeze(1), 1, target.size(1))
        self.F = utils.tile(self.F.unsqueeze(0), 0, target.size(0))
        self.F = utils.tile(self.F.unsqueeze(1), 1, target.size(1))
        covmat = self.P_0
        for i in range(target.size(2)-1):
            state, covmat = self.single_timestep(state, preds, sigma, covmat, i)
            covariance = torch.cat((covariance,covmat.unsqueeze(2)), dim = 2)
            state_array = torch.cat((state_array, state.unsqueeze(2)), dim=2)
        return state_array, covariance

