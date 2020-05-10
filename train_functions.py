'''
This code is based on https://github.com/ekwebb/fNRI which in turn is based on https://github.com/ethanfetaya/NRI
(MIT licence)
'''

import argparse
import csv
import datetime
import os
import pickle
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from modules_sigma import *
from utils import *
import math



class Model(object):
    def __init__(self, args, edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler):
        # initialises states
        super(Model, self).__init__()
        self.edge_types = edge_types
        self.log_prior = log_prior
        self.encoder_file = encoder_file
        self.decoder_file = decoder_file
        self.current_encoder_file = current_encoder_file
        self.current_decoder_file = current_decoder_file
        self.log = log
        self.loss_data = loss_data
        self.csv_writer = csv_writer
        self.perm_writer = perm_writer
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        if args.NRI:
            self.train_loader, self.valid_loader, self.test_loader, self.loc_max, self.loc_min, self.vel_max, \
            self.vel_min = load_data_NRI(args.batch_size, args.sim_folder, shuffle=True, data_folder=args.data_folder)
        else:
            self.train_loader, self.valid_loader, self.test_loader, self.loc_max, self.loc_min, self.vel_max,\
            self.vel_min = load_data_fNRI(args.batch_size, args.sim_folder, shuffle=True, data_folder=args.data_folder)

        self.datatensor = torch.FloatTensor([])
        if args.cuda:
            self.datatensor = self.datatensor.cuda()
        self.datatensor = Variable(self.datatensor)
        for batch_idx, (data, relations) in enumerate(self.train_loader):
            if args.cuda:
                data, relations = data.cuda(), relations.cuda()
            data, relations = Variable(data), Variable(relations)
            self.datatensor = torch.cat((self.datatensor, data), dim=0)

        # get the prior for sigma
        self.sigma_target = getsigma_target(self.datatensor, phys_error_folder=args.phys_folder,
                                            comp_error_folder=args.comp_folder, data_folder=args.data_folder,
                                            sim_folder=args.sim_folder)
        self.sigma_target = self.sigma_target.unsqueeze(dim=0)
        self.sigma_target = self.sigma_target.unsqueeze(dim=1)
        if args.cuda:
            self.sigma_target = self.sigma_target.cuda()

        # Generate off-diagonal interaction graph
        self.off_diag = np.ones([args.num_atoms, args.num_atoms]) - np.eye(args.num_atoms)
        self.rel_rec = np.array(encode_onehot(np.where(self.off_diag)[1]), dtype=np.float32)
        self.rel_send = np.array(encode_onehot(np.where(self.off_diag)[0]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(self.rel_rec)
        self.rel_send = torch.FloatTensor(self.rel_send)

        # initialise parameters needed for convexification
        self.alpha = 1
        self.preds_prev = torch.zeros((args.num_atoms, args.timesteps-1, 4))
        self.sigma_prev = torch.zeros((args.num_atoms, args.timesteps-1, 4))
        if args.cuda:
            self.preds_prev, self.sigma_prev = self.preds_prev.cuda(), self.sigma_prev.cuda()
        self.preds_prev, self.sigma_prev = Variable(self.preds_prev), Variable(self.sigma_prev)

        if args.cuda:
            self.rel_rec = self.rel_rec.cuda()
            self.rel_send = self.rel_send.cuda()

        self.rel_rec = Variable(self.rel_rec)
        self.rel_send = Variable(self.rel_send)




    def close_log(self, save_folder, log_csv, perm_csv):
        # close the log files
        if self.log is not None:
            print(save_folder)
            self.log.close()
            log_csv.close()
            perm_csv.close()
            self.loss_data.close()



    def loss_fixed(self, data_decoder, edges, sigma, args, pred_steps=1, use_onepred=False):
        # calculate the loss for the fixed variance case. Also returns the output of the NN.
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, False,
                                                     False, args.temp_softplus, pred_steps)
        else:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, False,
                                                     False, args.temp_softplus, args.prediction_steps)
        # calculate loss function
        loss_nll = nll_gaussian(output, target, args.var)
        loss_nll_var = nll_gaussian_var(output, target, args.var)
        return loss_nll, loss_nll_var, output

    def loss_anisotropic(self, data_decoder, edges, sigma, args, pred_steps=1, use_onepred=False):
        # calculate the loss for the anisotropic case. Also returns the output of the NN
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, pred_steps)
        else:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, args.prediction_steps)
        # calculate loss function
        loss_nll, loss_1, loss_2 = nll_gaussian_multivariatesigma_efficient(output, target, sigma, accel, vel)
        loss_nll_var = nll_gaussian_var_multivariatesigma_efficient(output, target, sigma, accel, vel)
        return loss_nll, loss_nll_var, output

    def loss_KL(self, data_decoder, edges, sigma, args, pred_steps=1, use_onepred=False):
        # calculate the loss for the anisotropic case with a KL divergence to account for prior. Also returns the output of the NN
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # recast target sigma to correct shape.
        sigma_target_1 = tile(self.sigma_target, 0, sigma.size(0))
        sigma_target_1 = tile(sigma_target_1, 1, sigma.size(1))
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, pred_steps)
        else:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, args.prediction_steps)
        # calculate the loss function
        loss_nll, loss_1, loss_2 = nll_gaussian_multivariatesigma_efficient(output, target, sigma, accel, vel)
        loss_nll_var = nll_gaussian_var_multivariatesigma_efficient(output, target, sigma, accel, vel)
        loss_kl_decoder = KL_output_multivariate(output, sigma, target, sigma_target_1)
        return loss_nll + args.beta * loss_kl_decoder, loss_nll_var, output

    def loss_normalinversewishart(self, data_decoder, edges, sigma, args, batch_idx, settouse, pred_steps=1, use_onepred=False):
        # calculate the loss for the anisotropic Normal-Inverse-Wishart distribution. Also returns the output of the NN.
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, pred_steps)
        else:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, args.prediction_steps)
        # get the appropriate prior.
        if settouse.lower() == 'train':
            prior_pos = self.prior_pos_tensor_train[batch_idx]
            prior_vel = self.prior_vel_tensor_train[batch_idx]
        elif settouse.lower() == 'validation':
            prior_pos = self.prior_pos_tensor_valid[batch_idx]
            prior_vel = self.prior_vel_tensor_valid[batch_idx]
        elif settouse.lower() == 'test':
            prior_pos = self.prior_pos_tensor_test[batch_idx]
            prior_vel = self.prior_vel_tensor_test[batch_idx]
        else:
            print("The set to use parameter must be one of 'train', 'validation' or 'test'.")
        # calculate the loss
        loss_nll, loss_nll_var = nll_Normal_Inverse_WishartLoss(output, sigma, accel, vel, prior_pos, prior_vel)
        return loss_nll, loss_nll_var, output

    def loss_kalmanfilter(self, data_decoder, edges, sigma, args, pred_steps=1, use_onepred=False):
        # calculate the loss for the anisotropic Normal distribution with a Kalman filter envelope. Also returns the output of the NN.
        # Note that this follows the suggestion given in: Multivariate Uncertainty in Deep Learning,
        #                                                 Rebecca L. Russell and Christopher Reale,
        #                                                 2019,  	arXiv:1910.14215 [cs.LG]
        target = data_decoder[:, :, 1:, :]
        # recast target sigma to correct shape
        sigma_target_1 = tile(self.sigma_target, 0, sigma.size(0))
        sigma_target_1 = tile(sigma_target_1, 1, sigma.size(1))
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, pred_steps)
        else:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, args.prediction_steps)
        # calculates the output after passing through a kalman filter
        kalmanfiler = KalmanFilter(sigma_target_1[:output.size(0), :, 0, :])
        output, covmat = kalmanfiler.kalman_filter_steps(target, output, sigma)
        # loss here is the MSE.
        loss_nll = F.mse_loss(output, target)
        loss_nll_var = nll_gaussian_var(output, target, args.var)
        return loss_nll, loss_nll_var, output

    def loss_isotropic(self, data_decoder, edges, sigma, args, epoch, pred_steps=1, use_onepred=False):
        # calculates the loss for the isotropic model. Also returns the output of the NN
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma_1, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       False, args.temp_softplus, pred_steps)
        else:
            output, sigma_1, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       False, args.temp_softplus, args.prediction_steps)
        # in case of isotropic we need to recast sigma to the same shape as output as it is required in the gaussian loss calculation
        sigma_1 = tile(sigma_1, 3, list(output.size())[3])
        # calculates loss function
        loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma(output, target, sigma_1, epoch, args.temp_sigmoid,
                                                              args.epochs)
        loss_nll_var = nll_gaussian_var__variablesigma(output, target, sigma_1, epoch, args.temp_sigmoid, args.epochs)
        return loss_nll, loss_nll_var, output

    def loss_lorentzian(self, data_decoder, edges, sigma, args, pred_steps=1, use_onepred=False):
        # calculates the loss for the Lorentzian model. Also returns the output of the NN
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma_1, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       False, args.temp_softplus, pred_steps)
        else:
            output, sigma_1, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       False, args.temp_softplus, args.prediction_steps)
        # in case of isotropic we need to recast sigma to the same shape as output as it is required in the gaussian loss calculation
        sigma_1 = tile(sigma_1, 3, list(output.size())[3])
        # calculates loss function
        loss_nll = nll_lorentzian(output, target, sigma_1)
        loss_nll_var = nll_lorentzian_var(output, target, sigma_1)
        return loss_nll, loss_nll_var, output

    def loss_semi_isotropic(self, data_decoder, edges, sigma, args, epoch, pred_steps=1, use_onepred=False):
        # calculates the loss for the semi-isotropic Gaussian model. Also returns the output of the NN
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       False, args.temp_softplus, pred_steps)
        else:
            output, sigma, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       False, args.temp_softplus, args.prediction_steps)
        # calculates loss function
        loss_nll, loss_1, loss_2 = nll_gaussian_variablesigma_semiisotropic(output, target, sigma, epoch,
                                                                            args.temp_sigmoid, args.epochs)
        loss_nll_var = nll_gaussian_var__variablesigma_semiisotropic(output, target, sigma, epoch, args.temp_sigmoid,
                                                                     args.epochs)
        return loss_nll, loss_nll_var, output

    def loss_anisotropic_withconvex(self, data_decoder, edges, sigma, args, vvec, sigmavec, pred_steps=1, use_onepred= False):
        # calculates the loss for the anisotropic Gaussian model with convexification. Also returns the output of the NN
        # The algorithm was designed by Edoardo Calvello
        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
        # ensures alpha is not too small
        if (abs(self.alpha) > 1e-16):
            self.alpha = 1e-16
        # forward() in decoder called here - carries out decoding step
        if use_onepred:
            output, sigma_1, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, pred_steps)
        else:
            output, sigma_1, accel, vel = self.decoder(data_decoder, edges, self.rel_rec, self.rel_send, sigma, True,
                                                       True, args.temp_softplus, args.prediction_steps)
        sigma_prev = tile(self.sigma_prev.unsqueeze(0), 0, target.size(0))
        preds_prev = tile(self.preds_prev.unsqueeze(0), 0, target.size(0))
        # calculates the loss
        loss_nll, loss_1, loss_2 = nll_gaussian_multivariatesigma_convexified(output, target, sigma_1, accel, vel, sigma_prev, preds_prev,  vvec, sigmavec, self.alpha)
        loss_nll_var = nll_gaussian_multivariatesigma_var_convexified(output, target, sigma_1, accel, vel, sigma_prev, preds_prev, vvec, sigmavec, self.alpha)
        # update step 3 of algorithm by Edoardo Calvello
        vvec_new = preds_prev + (output-preds_prev) /self.alpha
        sigmavec_new = sigma_prev + (sigma_1-sigma_prev) / self.alpha
        self.alpha = (np.sqrt(pow(self.alpha,4) + 4 * pow(self.alpha, 2)) - pow(self.alpha, 2)) / 2
        return loss_nll, loss_nll_var, output, vvec_new, sigmavec_new

    def fixed_var_plot(self, args, acc_blocks_batch, target, output_plot):
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from trajectory_plot import draw_lines
        # plots trajectories over timesteps - Plot the trajectories of the output of the NN and the target
        for i in range(args.batch_size):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            # ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            xmin_t, ymin_t, xmax_t, ymax_t = draw_lines(target, i, linestyle=':', alpha=0.6)
            xmin_o, ymin_o, xmax_o, ymax_o = draw_lines(output_plot.detach().cpu().numpy(), i,
                                                        linestyle='-')
            ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
            ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
            rect = patches.Rectangle((-1,-1),2,2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            # ax.set_xticks(np.linspace(math.ceil(min(xmin_t, xmin_o)*10)/10, math.floor(max(xmax_t, xmax_o)*10)/10,2))
            # ax.set_yticks(np.linspace(math.ceil(min(ymin_t, ymin_o)*10)/10, math.floor(max(ymax_t, ymax_o)*10)/10,2))
            block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
            # block_names = [ 'springs', 'charges' ]
            acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                        for j in range(acc_blocks_batch.shape[1])]
            acc_text = ', '.join(acc_text)
            plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
            plt.show()

    def isotropic_plot(self, args, data_decoder, edges, sigma, logits, logits_split, relations, target, zscorelist_x,
                       zscorelist_y):
        # plots the graphs for the isotropic model. returns the z-score values parallel and perpendicular to the velocity.
        import matplotlib.pyplot as plt
        # for plotting
        output_plot, sigma_plot, accel_plot, vel_plot = self.decoder(data_decoder, edges, self.rel_rec,
                                                                     self.rel_send, sigma, True, False,
                                                                     args.temp_softplus, 49)
        if args.loss_type.lower() == 'isotropic' or args.loss_type.lower() == 'lorentzian':
            # put sigma_plot in correct form for isotropic case
            sigma_plot = tile(sigma_plot, 3, list(output_plot.size())[3])
        else:
            # semi-isotropic need to select the position coords and velocity coords and convert from
            # [batch, particle, time , 1] -> [batch, particle, time, 2 (3 for 3D)] then reconcatinate to
            # put sigma_plot in correct form
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
            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split,
                                                                              relations,
                                                                              args.edge_types_list)
        # plot the mean value of sigma over timestep:
        timestep = np.arange(0.1, 0.1 * (args.timesteps - 1), 0.1)
        sigma_mean = sigma_plot.mean(dim=0)
        sigma_mean = sigma_mean.mean(dim=0)
        sigma_mean = sigma_mean.mean(dim=1)
        sigma_mean = sigma_mean.detach().cpu().numpy()
        fig = plt.figure()
        plt.plot(timestep, sigma_mean, label='raw data')
        plt.ylabel('Sigma Value')
        plt.xlabel('Time along Trajectory')
        plt.show()
        from trajectory_plot import draw_lines_sigma
        from matplotlib.patches import Ellipse, Rectangle
        # plotting graphs for isotropic/anisotropic case - here we plot the values of sigma using
        # Ellipses which are same colour as plots (see draw_lines_sigma in trajectory_plot)
        for i in range(args.batch_size):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            # ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            xmin_t, ymin_t, xmax_t, ymax_t = -1, -1, 1, 1
            xmin_o, ymin_o, xmax_o, ymax_o = -0.5, -0.5, 0.5, 0.5
            xmin_t, ymin_t, xmax_t, ymax_t = draw_lines_sigma(target, i, sigma_plot.detach().cpu().numpy(), ax,
                                                              linestyle=':', alpha=0.6)
            xmin_o, ymin_o, xmax_o, ymax_o = draw_lines_sigma(output_plot.detach().cpu().numpy(), i,
                                                              sigma_plot.detach().cpu().numpy(), ax, linestyle='-',
                                                              plot_ellipses=True)
            ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
            ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
            rect = Rectangle((-1, -1), 2, 2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
            # block_names = [ 'springs', 'charges' ]
            acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                        for j in range(acc_blocks_batch.shape[1])]
            acc_text = ', '.join(acc_text)
            plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
            plt.show()
        # z-score calcualtion
        if (torch.min(sigma_plot) < pow(10, -7)):
            accuracy = np.full((sigma_plot.size(0), sigma_plot.size(1), sigma_plot.size(2),
                                sigma_plot.size(3)), pow(10, -7), dtype=np.float32)
            accuracy = torch.from_numpy(accuracy)
            if args.cuda:
                accuracy = accuracy.cuda()
            output_plot = torch.max(output_plot, accuracy)
        # z = (y-yhat)/sigma
        zscore = (output_plot - target) / (sigma_plot)
        # select out velocity coords to get direction parallel and perpendicular to velocity- need this
        # to find z-score valuesalong these directions
        indices = torch.LongTensor([2, 3])
        if args.cuda:
            indices = indices.cuda()
        velocities = torch.index_select(output_plot, 3, indices)
        # abs(v)
        velnorm = velocities.norm(p=2, dim=3, keepdim=True)
        # vhat = v/abs(v)
        normalisedvel = velocities.div(velnorm.expand_as(velocities))
        accelnorm = accel_plot.norm(p=2, dim=3, keepdim=True)
        normalisedaccel = accel_plot.div(accelnorm.expand_as(accel_plot))
        # get perpendicular components to the accelerations and velocities accelperp, velperp
        # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
        rotationmatrix = np.zeros(
            (velocities.size(0), velocities.size(1), velocities.size(2), 2, 2),
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
        indices = torch.LongTensor([0, 1])
        if args.cuda:
            indices = indices.cuda()
        # zscore along the position axes
        zscore = torch.index_select(zscore, 3, indices)
        # zscore parallel to velocity
        zscore_x = torch.matmul(zscore.unsqueeze(3), normalisedvel.unsqueeze(4))
        # zscore perp to velocity
        zscore_y = torch.matmul(zscore.unsqueeze(3), velperp.unsqueeze(4))
        zscorelist_x.append(zscore_x)
        zscorelist_y.append(zscore_y)
        return zscorelist_x, zscorelist_y

    def anisotropic_plot(self, args, data_decoder, edges, sigma, logits, logits_split, relations, target, zscorelist_x,
                         zscorelist_y):
        # plots the graphs for the anisotropic model. returns the z-score values parallel and perpendicular to the velocity.
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        output_plot, sigma_plot, accel_plot, vel_plot = self.decoder(data_decoder, edges, self.rel_rec,
                                                                     self.rel_send, sigma, True, True,
                                                                     args.temp_softplus, 49)
        # plot the MSE over time to investigate chaos theory.
        loc = target[:, :, :, 0:2].detach().numpy()
        loc_new = output_plot[:, :, :, 0:2].detach().numpy()
        mse_loc = ((loc_new - loc) ** 2).mean(axis=3) / args.num_atoms
        mse_loc = mse_loc.mean(axis=1)
        mse_loc = mse_loc.mean(axis=0)
        deltaT = 0.1
        T = np.arange(0, deltaT * (len(mse_loc) - 1 / 2), 0.1)
        optimised_params_x, pcov = curve_fit(exp, T, mse_loc,
                                             p0=[1, 5, -1], maxfev=1000000)
        fig = plt.figure()
        plt.plot(T, np.log(exp(T, *optimised_params_x)), label='fit')
        plt.plot(T, np.log(mse_loc), label='raw data')
        plt.ylabel('log(Mean Square Error)')
        plt.xlabel('Time along Trajectory')
        plt.legend(loc='best')
        plt.show()
        # plot the mean value of sigma over timestep:
        timestep = np.arange(0.1, 0.1 * (args.timesteps - 1), 0.1)
        sigma_mean = sigma_plot.mean(dim=0)
        sigma_mean = sigma_mean.mean(dim=0)
        sigma_mean = sigma_mean.mean(dim=1)
        sigma_mean = sigma_mean.detach().cpu().numpy()
        optimised_params_x, pcov = curve_fit(exp, timestep, sigma_mean,
                                             p0=[1, 5, -1], maxfev=1000000)
        fig = plt.figure()
        plt.plot(T, exp(T, *optimised_params_x), label='fit')
        plt.plot(timestep, sigma_mean, label='raw data')
        plt.ylabel('Sigma Value')
        plt.xlabel('Time along Trajectory')
        plt.legend(loc='best')
        plt.show()
        fig = plt.figure()
        plt.plot(T, np.log(exp(T, *optimised_params_x)), label='fit')
        plt.plot(timestep, np.log(sigma_mean), label='raw data')
        plt.ylabel('log(Sigma Value)')
        plt.xlabel('Time along Trajectory')
        plt.legend(loc='best')
        plt.show()
        if args.NRI:
            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                             args.edge_types_list)
        else:
            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split,
                                                                              relations,
                                                                              args.edge_types_list)
        # plot trajectories and sigma's - use ellipses to plot anisotropy - see draw_lines_anisotropic
        from trajectory_plot import draw_lines_anisotropic
        from matplotlib.patches import Rectangle
        for i in range(args.batch_size):
            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111)
            # ax = fig.add_axes([0, 0, 1, 1])
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            xmin_t, ymin_t, xmax_t, ymax_t = draw_lines_anisotropic(target, i,
                                                                    sigma_plot.detach().cpu().numpy(),
                                                                    vel_plot, ax, linestyle=':',
                                                                    alpha=0.6)
            xmin_o, ymin_o, xmax_o, ymax_o = draw_lines_anisotropic(
                output_plot.detach().cpu().numpy(), i, sigma_plot.detach().cpu().numpy(),
                vel_plot, ax, linestyle='-', plot_ellipses=True)
            ax.set_xlim([min(xmin_t, xmin_o), max(xmax_t, xmax_o)])
            ax.set_ylim([min(ymin_t, ymin_o), max(ymax_t, ymax_o)])
            rect = Rectangle((-1, -1), 2, 2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            block_names = ['layer ' + str(j) for j in range(len(args.edge_types_list))]
            # block_names = [ 'springs', 'charges' ]
            acc_text = [block_names[j] + ' acc: {:02.0f}%'.format(100 * acc_blocks_batch[i, j])
                        for j in range(acc_blocks_batch.shape[1])]
            acc_text = ', '.join(acc_text)
            plt.text(0.5, 0.95, acc_text, horizontalalignment='center', transform=ax.transAxes)
            ax.xaxis.set_visible(True)
            ax.yaxis.set_visible(True)
            plt.xlabel('x')
            plt.ylabel('y')
            # plt.savefig(os.path.join(args.load_folder,str(i)+'_pred_and_true.png'), dpi=300)
            plt.show()
        # for z score
        # make sure we aren't dividing by 0
        if (torch.min(sigma_plot) < pow(10, -7)):
            accuracy = np.full((sigma_plot.size(0), sigma_plot.size(1), sigma_plot.size(2),
                                sigma_plot.size(3)), pow(10, -7), dtype=np.float32)
            accuracy = torch.from_numpy(accuracy)
            if args.cuda:
                accuracy = accuracy.cuda()
            output_plot = torch.max(output_plot, accuracy)
        zscore = (output_plot - target) / (sigma_plot)
        # select out velocity coords to get direction parallel and perpendicular to velocity- need this
        # to find z-score values along these directions
        velnorm = vel_plot.norm(p=2, dim=3, keepdim=True)
        normalisedvel = vel_plot.div(velnorm.expand_as(vel_plot))
        accelnorm = accel_plot.norm(p=2, dim=3, keepdim=True)
        normalisedaccel = accel_plot.div(accelnorm.expand_as(accel_plot))
        # get perpendicular components to the accelerations and velocities accelperp, velperp
        # note in 2D perpendicular vector is just rotation by pi/2 about origin (x,y) -> (-y,x)
        rotationmatrix = np.zeros(
            (normalisedvel.size(0), normalisedvel.size(1), normalisedvel.size(2), 2, 2),
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
        indices = torch.LongTensor([0, 1])
        if args.cuda:
            indices = indices.cuda()
        zscore = torch.index_select(zscore, 3, indices)
        zscore_x = torch.matmul(zscore.unsqueeze(3), normalisedvel.unsqueeze(4))
        zscore_y = torch.matmul(zscore.unsqueeze(3), velperp.unsqueeze(4))
        zscorelist_x.append(zscore_x)
        zscorelist_y.append(zscore_y)
        return zscorelist_x, zscorelist_y

    def zscore_plot(self, zscorelist_x, zscorelist_y):
        # plots the z-score graphs
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        # average over all timesteps
        zscorelistintx = np.empty((0))
        zscorelistinty = np.empty((0))
        for i in range(len(zscorelist_x)):
            zscorelistintx = np.append(zscorelistintx, zscorelist_x[i].numpy())
            zscorelistinty = np.append(zscorelistinty, zscorelist_y[i].numpy())
        bins = np.arange(-4, 4.1, 0.1)
        # get histogram distribution for terms parallel to the velocity
        histdatax, bin_edges, patches = plt.hist(zscorelistintx, bins, density=True)
        # take the histdata point to be at the centre of the bin_edges:
        # Gaussian fit- we expect a good model to give mean = 0 and sigma = 1
        # for fit to full graph. NOTE USE histdatax instead of histdatax_new
        xcoords = np.empty(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            xcoords[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        numberofpoints = len(xcoords)

        # for fit to all points except central peak. NOTE USE histdatax_new instead of histdatax
        # xcoords = np.empty(0)
        # histdatax_new = np.empty(0)
        # xcoords_small = np.empty(0)
        # histdatax_small = np.empty(0)
        # for i in range(len(bin_edges) - 1):
        #     if (abs(bin_edges[i])>0.5):
        #         xcoords = np.append(xcoords, [(bin_edges[i] + bin_edges[i + 1])/ 2])
        #         histdatax_new = np.append(histdatax_new, [histdatax[i]])
        #     else:
        #         xcoords_small = np.append(xcoords_small, [(bin_edges[i] + bin_edges[i + 1])/ 2])
        #         histdatax_small = np.append(histdatax_small, [histdatax[i]])
        # numberofpoints = len(xcoords)
        # mean is 1/N SUM(xy)
        mean_gaussian_x = np.sum(xcoords * histdatax) / numberofpoints
        # var = 1/N SUM(y*(x-mean) ** 2)
        sigma_x = np.sqrt(np.sum(histdatax * (xcoords - mean_gaussian_x) ** 2) / numberofpoints)
        # Fit to Gaussian
        # optimised_params_x, pcov = curve_fit(gaussian, xcoords, histdatax, p0=[1, mean_gaussian_x, sigma_x])
        # plt.plot(xcoords, gaussian(xcoords, *optimised_params_x), label='fit')
        # fit ro Lorentzian
        # optimised_params_lor_x, pcov = curve_fit(lorentzian, xcoords, histdatax,
        #                                          p0=[1, mean_gaussian_x, sigma_x])
        # plt.plot(xcoords, lorentzian(xcoords, *optimised_params_lor_x), 'k')
        plt.xlabel("z-score")
        plt.ylabel("frequency")
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        plt.xlim(-4, 4)
        plt.savefig('zscorepartimestep.png')
        plt.show()
        # get histogram distribution for terms perpendicular to velocity
        histdatay, bin_edges, patches = plt.hist(zscorelistinty, bins, density=True)

        # take the histdata point to be at the centre of the bin_edges:
        # Gaussian fit- we expect a good model to give mean = 0 and sigma = 1
        ## for fit to full graph. NOTE USE histdatay instead of histdatay_new
        ycoords = np.empty(len(bin_edges) - 1)
        for i in range(len(bin_edges) - 1):
            ycoords[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
        numberofpoints = len(ycoords)

        # for fit to all points except central peak. NOTE USE histdatay_new instead of histdatay
        # ycoords = np.empty(0)
        # histdatay_new = np.empty(0)
        # ycoords_small = np.empty(0)
        # histdatay_small = np.empty(0)
        # for i in range(len(bin_edges) - 1):
        #     if (abs(bin_edges[i]) > 0.5):
        #         ycoords = np.append(ycoords,[(bin_edges[i] + bin_edges[i + 1]) / 2])
        #         histdatay_new = np.append(histdatay_new,[histdatay[i]])
        #     else:
        #         ycoords_small = np.append(ycoords_small, [(bin_edges[i] + bin_edges[i + 1]) / 2])
        #         histdatay_small = np.append(histdatay_small, [histdatay[i]])
        # numberofpoints = len(ycoords)

        # mean is 1/N SUM(xy)
        mean_gaussian_y = np.sum(ycoords * histdatay) / numberofpoints
        # var = 1/N SUM(y*(x-mean) ** 2)
        sigma_y = np.sqrt(np.sum(histdatay * (ycoords - mean_gaussian_y) ** 2) / numberofpoints)
        # Fit to Gaussian
        # optimised_params_y, pcov = curve_fit(gaussian, ycoords, histdatay, p0=[1, mean_gaussian_y, sigma_y])
        # plt.plot(ycoords, gaussian(ycoords, *optimised_params_y), label='fit')
        # Fit to Lorentzian
        # optimised_params_lor_y, pcov = curve_fit(lorentzian, ycoords, histdatay,
        #                                          p0=[1, mean_gaussian_y, sigma_y])
        # plt.plot(ycoords, lorentzian(ycoords, *optimised_params_lor_y), 'k')
        plt.xlabel("z-score")
        plt.ylabel("frequency")
        plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
        # plt.title('Timestep = ' + str(j))
        plt.xlim(-4, 4)
        plt.savefig('zscoreorth.png')
        plt.show()

        # print("Gaussian Fit parallel to vel with mean: " + str(
        #     optimised_params_x[1]) + " and std: " + str(optimised_params_x[2]))
        # print(
        #     "Lorentzian Fit parallel to vel with mean: " + str(optimised_params_lor_x[1]) + " and std: " + str(
        #         optimised_params_lor_x[2]))
        # print(
        #     "Gaussian Fit perpendicular to vel with mean: " + str(optimised_params_y[1]) + " and std: " + str(
        #         optimised_params_y[2]))
        # print("Lorentzian Fit perpendicular to vel with mean: " + str(
        #     optimised_params_lor_y[1]) + " and std: " + str(
        #     optimised_params_lor_y[2]))

        # each timestep z-score uncomment for this plot.
        # for j in range(1,len(zscorelist_x[0][0,0])-1):
        #     zscorelistintx = np.empty((0))
        #     zscorelistinty = np.empty((0))
        #     for i in range(len(zscorelist_x)):
        #         zscorelistintx = np.append(zscorelistintx, zscorelist_x[i][:, :, j, :].numpy())
        #         zscorelistinty = np.append(zscorelistinty, zscorelist_y[i][:, :, j, :].numpy())
        #     bins = np.arange(-4, 4.1, 0.1)
        #     # get histogram distribution
        #     histdatax, bin_edges, patches = plt.hist(zscorelistintx, bins, density = True)

            # take the histdata point to be at the centre of the bin_edges:
            # Gaussian fit- we expect a good model to give mean = 0 and sigma = 1
            # xcoords = np.empty(len(bin_edges) - 1)
            # for i in range(len(bin_edges) - 1):
            #     xcoords[i] = (bin_edges[i] + bin_edges[i+1]) /2
            # numberofpoints = len(xcoords)
            # # mean is 1/N SUM(xy)
            # mean_gaussian_x = np.sum(xcoords * histdatax) / numberofpoints
            # # var = 1/N SUM(y*(x-mean) ** 2)
            # sigma_x = np.sqrt(np.sum(histdatax * (xcoords - mean_gaussian_x) ** 2) / numberofpoints)
            # optimised_params_x, pcov =  curve_fit(gaussian, xcoords, histdatax, p0 = [1, mean_gaussian_x, sigma_x])
            # plt.plot(xcoords, gaussian(xcoords, *optimised_params_x), label =  'fit')
            # optimised_params_lor_x, pcov = curve_fit(lorentzian, xcoords, histdatax, p0=[1, mean_gaussian_x, sigma_x])
            # plt.plot(xcoords, lorentzian(xcoords, *optimised_params_lor_x), 'k')
            # plt.xlabel("z-score")
            # plt.ylabel("frequency")
            # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
            # plt.title('Timestep = ' + str(j))
            # plt.xlim(-4, 4)
            # plt.savefig('zscorepartimestep' + str(j) + '.png')
            # plt.show()
            # # get histogram distribution
            # histdatay, bin_edges, patches = plt.hist(zscorelistinty, bins, density=True)
            #
            # # take the histdata point to be at the centre of the bin_edges:
            # # Gaussian fit- we expect a good model to give mean = 0 and sigma = 1
            # ycoords = np.empty(len(bin_edges) - 1)
            # for i in range(len(bin_edges) - 1):
            #     ycoords[i] = (bin_edges[i] + bin_edges[i + 1]) / 2
            # numberofpoints = len(ycoords)
            # # mean is 1/N SUM(xy)
            # mean_gaussian_y = np.sum(ycoords * histdatay) / numberofpoints
            # # var = 1/N SUM(y*(x-mean) ** 2)
            # sigma_y = np.sqrt(np.sum(histdatay * (ycoords - mean_gaussian_y) ** 2) / numberofpoints)
            # optimised_params_y, pcov = curve_fit(gaussian, ycoords, histdatay, p0=[1, mean_gaussian_y, sigma_y])
            # plt.plot(ycoords, gaussian(ycoords, *optimised_params_y), label='fit')
            # optimised_params_lor_y, pcov = curve_fit(lorentzian, ycoords, histdatay, p0=[1, mean_gaussian_y, sigma_y])
            # plt.plot(ycoords, lorentzian(ycoords, *optimised_params_lor_y), 'k')
            # plt.xlabel("z-score")
            # plt.ylabel("frequency")
            # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
            # plt.title('Timestep = ' + str(j))
            # plt.xlim(-4, 4)
            # plt.savefig('zscoreorthtimestep' + str(j)+ '.png')
            # plt.show()
            #
            # print('Timestep = ' + str(j) + ". Gaussian Fit parallel to vel with mean: " + str(optimised_params_x[1]) + " and std: " + str(optimised_params_x[2]))
            # # print("Lorentzian Fit parallel to vel with mean: " + str(optimised_params_lor_x[1]) + " and std: " + str(optimised_params_lor_x[2]))
            # print(
            #     "Gaussian Fit perpendicular to vel with mean: " + str(optimised_params_y[1]) + " and std: " + str(optimised_params_y[2]))
            # # print("Lorentzian Fit perpendicular to vel with mean: " + str(optimised_params_lor_y[1]) + " and std: " + str(
            # #  optimised_params_lor_y[2]))



class Trainer(Model):
    def __init__(self, args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler):
        super(Trainer, self).__init__(args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler)

        # gets the prior for the normal inverse wishart distribution
        if args.loss_type.lower() == 'norminvwishart':
            # prior for training data
            t = time.time()
            self.prior_pos_tensor_train = np.empty(0)
            self.prior_vel_tensor_train = np.empty(0)
            for batch_idx, (data, relations) in enumerate(self.train_loader):
                if args.cuda:
                    data, relations = data.cuda(), relations.cuda()
                data, relations = Variable(data), Variable(relations)

                data = data.clone()
                relations = relations.clone()
                indices_pos = torch.LongTensor([0, 1])
                indices_vel = torch.LongTensor([2, 3])
                if args.cuda:
                    indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
                data_pos = torch.index_select(data, 3, indices_pos)
                data_vel = torch.index_select(data, 3, indices_vel)
                sigma_pos = torch.index_select(self.sigma_target, 3, indices_pos)
                sigma_vel = torch.index_select(self.sigma_target, 3, indices_vel)
                prior_pos = getpriordist(data_pos, sigma_pos, 4)
                prior_vel = getpriordist(data_vel, sigma_vel, 4)
                self.prior_pos_tensor_train = np.concatenate((self.prior_pos_tensor_train, [prior_pos]))
                self.prior_vel_tensor_train = np.concatenate((self.prior_vel_tensor_train, [prior_vel]))
            print('train time: {:.1f}s'.format(time.time() - t))
            t = time.time()
            # prior for validation data
            self.prior_pos_tensor_valid = np.empty(0)
            self.prior_vel_tensor_valid = np.empty(0)
            for batch_idx, (data, relations) in enumerate(self.valid_loader):
                if args.cuda:
                    data, relations = data.cuda(), relations.cuda()
                data, relations = Variable(data), Variable(relations)
                data = data.clone()
                relations = relations.clone()

                indices_pos = torch.LongTensor([0, 1])
                indices_vel = torch.LongTensor([2, 3])
                if args.cuda:
                    indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
                data_pos = torch.index_select(data, 3, indices_pos)
                data_vel = torch.index_select(data, 3, indices_vel)
                sigma_pos = torch.index_select(self.sigma_target, 3, indices_pos)
                sigma_vel = torch.index_select(self.sigma_target, 3, indices_vel)
                prior_pos = getpriordist(data_pos, sigma_pos, 4)
                prior_vel = getpriordist(data_vel, sigma_vel, 4)
                self.prior_pos_tensor_valid = np.concatenate((self.prior_pos_tensor_valid, [prior_pos]))
                self.prior_vel_tensor_valid = np.concatenate((self.prior_vel_tensor_valid, [prior_vel]))
            print('validation time: {:.1f}s'.format(time.time() - t))

    def train(self, epoch, best_val_loss, args):
        t = time.time()
        # train set
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

        self.encoder.train()
        self.decoder.train()
        self.scheduler.step()
        if not args.plot:
            for batch_idx, (data, relations) in enumerate(self.train_loader):  # relations are the ground truth interactions graphs
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
                sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms, inversesoftplus(pow(args.var,1/2), args.temp_softplus))
                if args.cuda:
                    sigma = sigma.cuda()
                if args.loss_type.lower() == 'semi_isotropic'.lower():
                    sigma = tile(sigma, 3, 2)
                sigma = Variable(sigma)
                self.optimizer.zero_grad()

                logits = self.encoder(data_encoder, self.rel_rec, self.rel_send)

                if args.NRI:
                    # dim of logits, edges and prob are [batchsize, N^2-N, edgetypes] where N = no. of particles
                    edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
                    prob = my_softmax(logits, -1)

                    loss_kl = kl_categorical_uniform(prob, args.num_atoms, self.edge_types)
                    loss_kl_split = [loss_kl]
                    loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, self.edge_types)]

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
                        loss_kl_split = [kl_categorical(prob_split[type_idx], self.log_prior[type_idx], args.num_atoms)
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
                # fixed variance train loss calculation
                target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                if args.loss_type.lower() == 'fixed_var'.lower():
                    loss_nll, loss_nll_var, output = self.loss_fixed(data_decoder, edges, sigma, args)
                # variable variance train loss calculation
                elif args.loss_type.lower() == 'anisotropic':
                    loss_nll, loss_nll_var, output = self.loss_anisotropic(data_decoder, edges, sigma, args)
                elif args.loss_type.upper() == 'KL':
                    loss_nll, loss_nll_var, output = self.loss_KL(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'norminvwishart':
                    loss_nll, loss_nll_var, output = self.loss_normalinversewishart(data_decoder, edges, sigma, args, batch_idx, 'train')
                elif args.loss_type.lower() == 'kalmanfilter':
                    loss_nll, loss_nll_var, output = self.loss_kalmanfilter(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'isotropic':
                    loss_nll, loss_nll_var, output = self.loss_isotropic(data_decoder, edges, sigma, args, epoch)
                elif args.loss_type.lower() == 'lorentzian':
                    loss_nll, loss_nll_var, output = self.loss_lorentzian(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'semi_isotropic'.lower():
                    loss_nll, loss_nll_var, output = self.loss_semi_isotropic(data_decoder, edges, sigma, args, epoch)
                elif args.loss_type.lower() == 'ani_convex'.lower():
                    target = data_decoder[:, :, 1:, :]
                    if epoch == 0:
                        vvec = target.clone()
                        sigma_vec = sigma[:,:,1:,:].clone()
                    loss_nll, loss_nll_var, output, vvec, sigma_vec = self.loss_anisotropic_withconvex(data_decoder, edges, sigma, args, vvec, sigma_vec)


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
                self.optimizer.step()

                mse_train.append(F.mse_loss(output, target).data.item())
                nll_train.append(loss_nll.data.item())
                kl_train.append(loss_kl.data.item())
                kl_list_train.append([kl.data.item() for kl in loss_kl_split])

                nll_var_train.append(loss_nll_var.data.item())
                kl_var_list_train.append([kl_var.data.item() for kl_var in loss_kl_var_split])

        # validation set
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

        self.encoder.eval()
        self.decoder.eval()
        for batch_idx, (data, relations) in enumerate(self.valid_loader):
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
                sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms, inversesoftplus(pow(args.var,1/2), args.temp_softplus))
                if args.cuda:
                    sigma = sigma.cuda()
                if args.loss_type.lower() == 'semi_isotropic'.lower():
                    sigma = tile(sigma, 3, 2)
                sigma = Variable(sigma)
                # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
                logits = self.encoder(data_encoder, self.rel_rec, self.rel_send)

                if args.NRI:
                    # dim of logits, edges and prob are [batchsize, N^2-N, edgetypes] where N = no. of particles
                    edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)  # uses concrete distribution (for hard=False) to sample edge types
                    prob = my_softmax(logits, -1)  # my_softmax returns the softmax over the edgetype dim

                    loss_kl = kl_categorical_uniform(prob, args.num_atoms, self.edge_types)
                    loss_kl_split = [loss_kl]
                    loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, self.edge_types)]

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
                        loss_kl_split = [kl_categorical(prob_split[type_idx], self.log_prior[type_idx], args.num_atoms)
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
                target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                # validation loss calculation
                if args.loss_type.lower() == 'fixed_var'.lower():
                    loss_nll, loss_nll_var, output = self.loss_fixed(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_fixed(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'isotropic':
                    loss_nll, loss_nll_var, output = self.loss_isotropic(data_decoder, edges, sigma, args, epoch, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_isotropic(data_decoder, edges, sigma, args, epoch)
                elif args.loss_type.lower() == 'lorentzian':
                    loss_nll, loss_nll_var, output = self.loss_lorentzian(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_lorentzian(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'semi_isotropic'.lower():
                    loss_nll, loss_nll_var, output = self.loss_semi_isotropic(data_decoder, edges, sigma, args, epoch, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_semi_isotropic(data_decoder, edges, sigma, args, epoch)
                elif args.loss_type.lower() == 'anisotropic':
                    loss_nll, loss_nll_var, output = self.loss_anisotropic(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_anisotropic(data_decoder, edges, sigma, args)
                elif args.loss_type.upper() == 'KL':
                    loss_nll, loss_nll_var, output = self.loss_KL(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_KL(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'norminvwishart':
                    loss_nll, loss_nll_var, output = self.loss_normalinversewishart(data_decoder, edges, sigma,
                                                                                    args, batch_idx, 'validation', use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_normalinversewishart(data_decoder, edges, sigma,
                                                                                    args, batch_idx, 'validation')
                elif args.loss_type.lower() == 'kalmanfilter':
                    loss_nll, loss_nll_var, output = self.loss_kalmanfilter(data_decoder, edges, sigma, args,
                                                                            use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_kalmanfilter(data_decoder, edges, sigma, args)

                elif args.loss_type.lower() == 'ani_convex'.lower():
                    target = data_decoder[:, :, 1:, :]
                    if epoch == 0:
                        vvec_ver = target.clone()
                        sigma_vec_ver = sigma[:,:,1:,:].clone()
                    loss_nll, loss_nll_var, output, vvec_ver_n, sigma_vec_ver_n = self.loss_anisotropic_withconvex(data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M, vvec_ver, sigma_vec_ver = self.loss_anisotropic_withconvex(data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver)

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
              file=self.log)
        print('nll_trn: {:.5f}'.format(np.mean(nll_train)),
              'kl_trn: {:.5f}'.format(np.mean(kl_train)),
              'mse_trn: {:.10f}'.format(np.mean(mse_train)),
              'acc_trn: {:.5f}'.format(np.mean(acc_train)),
              'KLb_trn: {:.5f}'.format(np.mean(KLb_train)),
              'acc_b_trn: ' + str(np.around(np.mean(np.array(acc_blocks_train), axis=0), 4)),
              'kl_trn: ' + str(np.around(np.mean(np.array(kl_list_train), axis=0), 4)),
              file=self.log)
        print('nll_val: {:.5f}'.format(np.mean(nll_M_val)),
              'kl_val: {:.5f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.5f}'.format(np.mean(acc_val)),
              'KLb_val: {:.5f}'.format(np.mean(KLb_val)),
              'acc_b_val: ' + str(np.around(np.mean(np.array(acc_blocks_val), axis=0), 4)),
              'kl_val: ' + str(np.around(np.mean(np.array(kl_list_val), axis=0), 4)),
              file=self.log)
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
            self.csv_writer.writerow(labels)

            labels = ['trn ' + str(i) for i in range(len(perm_train[0]))]
            labels += ['val ' + str(i) for i in range(len(perm_val[0]))]
            self.perm_writer.writerow(labels)

        self.csv_writer.writerow([epoch, np.mean(nll_train), np.mean(kl_train),
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
        self.perm_writer.writerow(list(np.mean(np.array(perm_train), axis=0)) +
                             list(np.mean(np.array(perm_val), axis=0))
                             )

        self.log.flush()
        # save condn
        if args.save_folder and np.mean(nll_M_val) < best_val_loss:
            torch.save(self.encoder.state_dict(), self.encoder_file)
            torch.save(self.decoder.state_dict(), self.decoder_file)
            print('Best model so far, saving...')
        # save model in different folder even if not the best model. This is a temporary fix for BUS errors- not ideal but
        # the best we can do..
        if args.save_folder:
            torch.save(self.encoder.state_dict(), self.current_encoder_file)
            torch.save(self.decoder.state_dict(), self.current_decoder_file)
        return np.mean(acc_val), np.mean(nll_M_val), np.around(np.mean(np.array(acc_blocks_val), axis=0), 4)

    def train_plot(self, epoch, args):
        t = time.time()
        # validation set
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

        self.encoder.eval()
        self.decoder.eval()
        for batch_idx, (data, relations) in enumerate(self.valid_loader):
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
                sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms,
                                  inversesoftplus(pow(args.var, 1 / 2), args.temp_softplus))
                if args.cuda:
                    sigma = sigma.cuda()
                if args.loss_type.lower() == 'semi_isotropic'.lower():
                    sigma = tile(sigma, 3, 2)
                sigma = Variable(sigma)
                # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
                logits = self.encoder(data_encoder, self.rel_rec, self.rel_send)

                if args.NRI:
                    # dim of logits, edges and prob are [batchsize, N^2-N, edgetypes] where N = no. of particles
                    edges = gumbel_softmax(logits, tau=args.temp,
                                           hard=args.hard)  # uses concrete distribution (for hard=False) to sample edge types
                    prob = my_softmax(logits, -1)  # my_softmax returns the softmax over the edgetype dim

                    loss_kl = kl_categorical_uniform(prob, args.num_atoms, self.edge_types)
                    loss_kl_split = [loss_kl]
                    loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, self.edge_types)]

                    KLb_val.append(0)
                    KLb_blocks_val.append([0])

                    if args.no_edge_acc:
                        acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = 0, np.array([0]), np.zeros(
                            len(args.edge_types_list)), 0, np.zeros(len(args.edge_types_list))
                    else:
                        acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_NRI(logits,
                                                                                                     relations,
                                                                                                     args.edge_types_list)

                else:
                    # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
                    logits_split = torch.split(logits, args.edge_types_list, dim=-1)
                    edges_split = tuple([gumbel_softmax(logits_i, tau=args.temp, hard=args.hard)
                                         for logits_i in logits_split])
                    edges = torch.cat(edges_split, dim=-1)
                    prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split]

                    if args.prior:
                        loss_kl_split = [kl_categorical(prob_split[type_idx], self.log_prior[type_idx], args.num_atoms)
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
                        acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = 0, np.array([0]), np.zeros(
                            len(args.edge_types_list)), 0, np.zeros(len(args.edge_types_list))
                    else:
                        acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_fNRI(logits_split,
                                                                                                      relations,
                                                                                                      args.edge_types_list,
                                                                                                      args.skip_first)

                    KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                    KLb_val.append(sum(KLb_blocks).data.item())
                    KLb_blocks_val.append([KL.data.item() for KL in KLb_blocks])
                # plotting for fixed variance models
                if args.loss_type.lower() == 'fixed_var'.lower():
                    target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                    if args.plot:
                        # for plotting
                        output_plot, sigma_plot, accel_plot, vel_plot = self.decoder(data_decoder, edges, self.rel_rec,
                                                                                self.rel_send, sigma, False, False,
                                                                                args.temp_softplus, 49)
                        if args.NRI:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                             args.edge_types_list)
                        else:
                            acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split,
                                                                                              relations,
                                                                                              args.edge_types_list)
                        self.fixed_var_plot(args, acc_blocks_batch, target, output_plot)

                    loss_nll, loss_nll_var, output = self.loss_fixed(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_fixed(data_decoder, edges, sigma, args)
                # plotting for non-fixed variance models
                elif args.loss_type.lower() == 'isotropic':
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.isotropic_plot(args, data_decoder, edges, sigma, logits,
                                                            logits_split, relations, target, zscorelist_x, zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_isotropic(data_decoder, edges, sigma, args, epoch, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_isotropic(data_decoder, edges, sigma, args, epoch)
                elif args.loss_type.lower() == 'lorentzian':
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.isotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                     logits_split, relations, target, zscorelist_x,
                                                                     zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_lorentzian(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_lorentzian(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'semi_isotropic'.lower():
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.isotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                     logits_split, relations, target, zscorelist_x,
                                                                     zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_semi_isotropic(data_decoder, edges, sigma, args, epoch, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_semi_isotropic(data_decoder, edges, sigma, args, epoch)
                elif args.loss_type.lower() == 'anisotropic'.lower():
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                     logits_split, relations, target, zscorelist_x,
                                                                     zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_anisotropic(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_anisotropic(data_decoder, edges, sigma, args)
                elif args.loss_type.upper() == 'KL':
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                       logits_split, relations, target, zscorelist_x,
                                                                       zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_KL(data_decoder, edges, sigma, args, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_KL(data_decoder, edges, sigma)
                elif args.loss_type.lower() == 'norminvwishart':
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                       logits_split, relations, target, zscorelist_x,
                                                                       zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_normalinversewishart(data_decoder, edges, sigma,
                                                                                    args, batch_idx, 'validation', use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_normalinversewishart(data_decoder, edges, sigma,
                                                                                    args, batch_idx, 'validation')
                elif args.loss_type.lower() == 'kalmanfilter':
                    target = data_decoder[:, :, 1:, :]
                    zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                       logits_split, relations, target, zscorelist_x,
                                                                       zscorelist_y)
                    loss_nll, loss_nll_var, output = self.loss_kalmanfilter(data_decoder, edges, sigma, args,
                                                                            use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M = self.loss_kalmanfilter(data_decoder, edges, sigma, args)
                elif args.loss_type.lower() == 'ani_convex'.lower():
                    target = data_decoder[:, :, 1:, :]
                    if epoch == 0:
                        vvec_ver = target.clone()
                        sigma_vec_ver = sigma.clone()
                    loss_nll, loss_nll_var, output, vvec_ver_n, sigma_vec_ver_n = self.loss_anisotropic_withconvex(data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver, use_onepred=True)
                    loss_nll_M, loss_nll_M_var, output_M, vvec_ver, sigma_vec_ver = self.loss_anisotropic_withconvex(data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver)


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
        # deal with z-score here - plot zscores and fit gaussian/lorentzian to the histogram plot
        if not args.loss_type.lower() == 'fixed_var'.lower():
            self.zscore_plot(zscorelist_x, zscorelist_y)
        print('Epoch: {:03d}'.format(epoch),
              'perm_val: ' + str(np.around(np.mean(np.array(perm_val), axis=0), 4)),
              'time: {:.1f}s'.format(time.time() - t))

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
              file=self.log)
        print('nll_val: {:.5f}'.format(np.mean(nll_M_val)),
              'kl_val: {:.5f}'.format(np.mean(kl_val)),
              'mse_val: {:.10f}'.format(np.mean(mse_val)),
              'acc_val: {:.5f}'.format(np.mean(acc_val)),
              'KLb_val: {:.5f}'.format(np.mean(KLb_val)),
              'acc_b_val: ' + str(np.around(np.mean(np.array(acc_blocks_val), axis=0), 4)),
              'kl_val: ' + str(np.around(np.mean(np.array(kl_list_val), axis=0), 4)),
              file=self.log)

class Tester(Model):

    def __init__(self, args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler):
        super(Tester, self).__init__(args,  edge_types, log_prior, encoder_file, decoder_file, current_encoder_file, current_decoder_file, log, loss_data, csv_writer, perm_writer, encoder, decoder, optimizer, scheduler)

        # gets the prior for the normal inverse wishart distribution on test data
        if args.loss_type.lower() == 'norminvwishart':
            # prior for test data
            self.prior_pos_tensor_test = np.empty(0)
            self.prior_vel_tensor_test = np.empty(0)
            t = time.time()
            for batch_idx, (data, relations) in enumerate(self.test_loader):
                if args.cuda:
                    data, relations = data.cuda(), relations.cuda()
                data, relations = Variable(data), Variable(relations)
                data = data.clone()
                relations = relations.clone()

                indices_pos = torch.LongTensor([0, 1])
                indices_vel = torch.LongTensor([2, 3])
                if args.cuda:
                    indices_pos, indices_vel = indices_pos.cuda(), indices_vel.cuda()
                data_pos = torch.index_select(data, 3, indices_pos)
                data_vel = torch.index_select(data, 3, indices_vel)
                sigma_pos = torch.index_select(self.sigma_target, 3, indices_pos)
                sigma_vel = torch.index_select(self.sigma_target, 3, indices_vel)
                prior_pos = getpriordist(data_pos, sigma_pos, 4)
                prior_vel = getpriordist(data_vel, sigma_vel, 4)
                self.prior_pos_tensor_test = np.concatenate((self.prior_pos_tensor_test, [prior_pos]))
                self.prior_vel_tensor_test = np.concatenate((self.prior_vel_tensor_test, [prior_vel]))
            print('test time: {:.1f}s'.format(time.time() - t))

    def test(self, args):
        # test set
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

        # for zscore analysis
        zscorelist_x = []
        zscorelist_y = []

        self.encoder.eval()
        self.decoder.eval()
        if not args.cuda:
            self.encoder.load_state_dict(torch.load(self.encoder_file, map_location='cpu'))
            self.decoder.load_state_dict(torch.load(self.decoder_file, map_location='cpu'))
        else:
            self.encoder.load_state_dict(torch.load(self.encoder_file))
            self.decoder.load_state_dict(torch.load(self.decoder_file))

        for batch_idx, (data, relations) in enumerate(self.test_loader):
            with torch.no_grad():
                if args.cuda:
                    data, relations = data.cuda(), relations.cuda()

                assert (data.size(2) - args.timesteps) >= args.timesteps
                data_encoder = data[:, :, :args.timesteps, :].contiguous()
                data_decoder = data[:, :, -args.timesteps:, :].contiguous()

                # stores the values of the uncertainty. This will be an array of size [batchsize, no. of particles, time,no. of axes (isotropic = 1, anisotropic = 2)]
                # initialise sigma to an array of large negative numbers which become small positive numbers when passted through softplus function.
                sigma = initsigma(len(data_decoder), len(data_decoder[0][0]), args.anisotropic, args.num_atoms,
                                  inversesoftplus(pow(args.var, 1 / 2), args.temp_softplus))
                if args.cuda:
                    sigma = sigma.cuda()
                if args.loss_type.lower() == 'semi_isotropic'.lower():
                    sigma = tile(sigma, 3, 2)
                sigma = Variable(sigma)
                # dim of logits, edges and prob are [batchsize, N^2-N, sum(edge_types_list)] where N = no. of particles
                logits = self.encoder(data_encoder, self.rel_rec, self.rel_send)

                if args.NRI:
                    edges = gumbel_softmax(logits, tau=args.temp, hard=args.hard)
                    prob = my_softmax(logits, -1)

                    loss_kl = kl_categorical_uniform(prob, args.num_atoms, self.edge_types)
                    loss_kl_split = [loss_kl]
                    loss_kl_var_split = [kl_categorical_uniform_var(prob, args.num_atoms, self.edge_types)]

                    KLb_test.append(0)
                    KLb_blocks_test.append([0])

                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_NRI(logits, relations,
                                                                                                 args.edge_types_list)

                else:
                    logits_split = torch.split(logits, args.edge_types_list, dim=-1)
                    edges_split = tuple(
                        [gumbel_softmax(logits_i, tau=args.temp, hard=args.hard) for logits_i in logits_split])
                    edges = torch.cat(edges_split, dim=-1)
                    prob_split = [my_softmax(logits_i, -1) for logits_i in logits_split]

                    if args.prior:
                        loss_kl_split = [kl_categorical(prob_split[type_idx], self.log_prior[type_idx],
                                                        args.num_atoms) for type_idx in
                                         range(len(args.edge_types_list))]
                        loss_kl = sum(loss_kl_split)
                    else:
                        loss_kl_split = [kl_categorical_uniform(prob_split[type_idx], args.num_atoms,
                                                                args.edge_types_list[type_idx])
                                         for type_idx in range(len(args.edge_types_list))]
                        loss_kl = sum(loss_kl_split)

                        loss_kl_var_split = [kl_categorical_uniform_var(prob_split[type_idx], args.num_atoms,
                                                                        args.edge_types_list[type_idx])
                                             for type_idx in range(len(args.edge_types_list))]

                    acc_perm, perm, acc_blocks, acc_var, acc_var_blocks = edge_accuracy_perm_fNRI(logits_split,
                                                                                                  relations,
                                                                                                  args.edge_types_list,
                                                                                                  args.skip_first)

                    KLb_blocks = KL_between_blocks(prob_split, args.num_atoms)
                    KLb_test.append(sum(KLb_blocks).data.item())
                    KLb_blocks_test.append([KL.data.item() for KL in KLb_blocks])
                    epoch = 0
                    # plotting fixed variance models
                    if args.loss_type.lower() == 'fixed_var'.lower():
                        target = data_decoder[:, :, 1:, :]  # dimensions are [batch, particle, time, state]
                        if args.plot:
                            # for plotting
                            output_plot, sigma_plot, accel_plot, vel_plot = self.decoder(data_decoder, edges,
                                                                                         self.rel_rec,
                                                                                         self.rel_send, sigma, False,
                                                                                         False,
                                                                                         args.temp_softplus, 49)
                            if args.NRI:
                                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_NRI_batch(logits, relations,
                                                                                                 args.edge_types_list)
                            else:
                                acc_batch, perm, acc_blocks_batch = edge_accuracy_perm_fNRI_batch(logits_split,
                                                                                                  relations,
                                                                                                  args.edge_types_list)
                            self.fixed_var_plot(args, acc_blocks_batch, target, output_plot)

                        loss_nll, loss_nll_var, output = self.loss_fixed(data_decoder, edges, sigma, args,
                                                                         use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_fixed(data_decoder, edges, sigma, args, pred_steps=10,
                                                                         use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_fixed(data_decoder, edges, sigma, args, pred_steps=20,
                                                                         use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_fixed(data_decoder, edges, sigma, args)
                    # plotting varying variance models. NOTE: THE MSE for 1, 10 and 20 trajectories is also calculated.
                    elif args.loss_type.lower() == 'isotropic':
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.isotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                         logits_split, relations, target, zscorelist_x,
                                                                         zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_isotropic(data_decoder, edges, sigma, args, epoch,
                                                                             use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_isotropic(data_decoder, edges, sigma, args, epoch,
                                                                             pred_steps=10, use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_isotropic(data_decoder, edges, sigma, args, epoch,
                                                                             pred_steps=20, use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_isotropic(data_decoder, edges, sigma, args,
                                                                                   epoch)
                    elif args.loss_type.lower() == 'lorentzian':
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.isotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                         logits_split, relations, target, zscorelist_x,
                                                                         zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_lorentzian(data_decoder, edges, sigma, args,
                                                                              use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_lorentzian(data_decoder, edges, sigma, args,
                                                                                       pred_steps=10, use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_lorentzian(data_decoder, edges, sigma, args,
                                                                                     pred_steps=20, use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_lorentzian(data_decoder, edges, sigma, args)
                    elif args.loss_type.lower() == 'semi_isotropic'.lower():
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.isotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                         logits_split, relations, target, zscorelist_x,
                                                                         zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_semi_isotropic(data_decoder, edges, sigma, args,
                                                                                  epoch, use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_semi_isotropic(data_decoder, edges, sigma, args,
                                                                                      epoch, pred_steps=10, use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_semi_isotropic(data_decoder, edges, sigma, args,
                                                                                      epoch, pred_steps=20, use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_semi_isotropic(data_decoder, edges, sigma,
                                                                                        args, epoch)
                    elif args.loss_type.lower() == 'anisotropic'.lower():
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                           logits_split, relations, target,
                                                                           zscorelist_x,
                                                                           zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_anisotropic(data_decoder, edges, sigma, args,
                                                                               use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_anisotropic(data_decoder, edges, sigma,
                                                                        args, pred_steps=10, use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_anisotropic(data_decoder, edges, sigma,
                                                                         args, pred_steps=20, use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_anisotropic(data_decoder, edges, sigma, args)
                    elif args.loss_type.upper() == 'KL':
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                           logits_split, relations, target,
                                                                           zscorelist_x,
                                                                           zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_KL(data_decoder, edges, sigma, args,
                                                                      use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_KL(data_decoder, edges, sigma, args, pred_steps=10,
                                                                                        use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_KL(data_decoder, edges, sigma,args, pred_steps=20,
                                                                                        use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_KL(data_decoder, edges, sigma, args)
                    elif args.loss_type.lower() == 'norminvwishart':
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                           logits_split, relations, target,
                                                                           zscorelist_x, zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_normalinversewishart(data_decoder, edges, sigma,
                                                                                        args, batch_idx, 'test',
                                                                                        use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_normalinversewishart(data_decoder, edges, sigma, args, batch_idx, 'test', pred_steps=10,
                                                                                        use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_normalinversewishart(data_decoder, edges, sigma,args, batch_idx, 'test', pred_steps=20,
                                                                                        use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_normalinversewishart(data_decoder, edges,
                                                                                              sigma,
                                                                                              args, batch_idx, 'test')
                    elif args.loss_type.lower() == 'kalmanfilter':
                        target = data_decoder[:, :, 1:, :]
                        if args.plot:
                            zscorelist_x, zscorelist_y = self.anisotropic_plot(args, data_decoder, edges, sigma, logits,
                                                                           logits_split, relations, target,
                                                                           zscorelist_x,
                                                                           zscorelist_y)
                        loss_nll, loss_nll_var, output = self.loss_kalmanfilter(data_decoder, edges, sigma, args,
                                                                                use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10 = self.loss_kalmanfilter(data_decoder, edges,
                                                                    sigma, args, pred_steps=10 ,use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20 = self.loss_kalmanfilter(data_decoder, edges,
                                                                  sigma, args,  pred_steps=20, use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M = self.loss_kalmanfilter(data_decoder, edges, sigma, args)
                    elif args.loss_type.lower() == 'ani_convex'.lower():
                        target = data_decoder[:, :, 1:, :]
                        if epoch == 0:
                            vvec_ver = target.clone()
                            sigma_vec_ver = sigma.clone()
                        loss_nll, loss_nll_var, output, vvec_ver_n, sigma_vec_ver_n = self.loss_anisotropic_withconvex(
                            data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver, use_onepred=True)
                        loss_nll_10, loss_nll_var_10, output_10, vvec_ver_n, sigma_vec_ver_n = self.loss_anisotropic_withconvex(
                            data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver, pred_steps=10, use_onepred=True)
                        loss_nll_20, loss_nll_var_20, output_20, vvec_ver_n, sigma_vec_ver_n = self.loss_anisotropic_withconvex(
                            data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver, pred_steps=20, use_onepred=True)
                        loss_nll_M, loss_nll_M_var, output_M, vvec_ver, sigma_vec_ver = self.loss_anisotropic_withconvex(
                            data_decoder, edges, sigma, args, vvec_ver, sigma_vec_ver)

                    perm_test.append(perm)
                    acc_test.append(acc_perm)
                    acc_blocks_test.append(acc_blocks)
                    acc_var_test.append(acc_var)
                    acc_var_blocks_test.append(acc_var_blocks)

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
        # deal with z-score here - plot zscores and fit gaussian/lorentzian to the histogram plot
        if not args.loss_type.lower() == 'fixed_var'.lower():
            if args.plot:
                self.zscore_plot(zscorelist_x, zscorelist_y)

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
            print('--------------------------------', file=self.log)
            print('------------Testing-------------', file=self.log)
            print('--------------------------------', file=self.log)
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
                  file=self.log)
            print('acc_b_test: ' + str(np.around(np.mean(np.array(acc_blocks_test), axis=0), 4)),
                  'acc_var_b_test: ' + str(np.around(np.mean(np.array(acc_var_blocks_test), axis=0), 4)),
                  'kl_test: ' + str(np.around(np.mean(np.array(kl_list_test), axis=0), 4)),
                  file=self.log)
            self.log.flush()
