'''
This code is based on https://github.com/ekwebb/fNRI which in turn is based on https://github.com/ethanfetaya/NRI
(MIT licence)
'''
from synthetic_sim_physerrors import *
import time
import numpy as np
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors

parser = argparse.ArgumentParser()
parser.add_argument('--num-train', type=int, default=10,
                    help='Number of training simulations to generate.')
parser.add_argument('--num-valid', type=int, default=10000,
                    help='Number of validation simulations to generate.')
parser.add_argument('--num-test', type=int, default=10000,
                    help='Number of test simulations to generate.')
parser.add_argument('--length', type=int, default=10000,
                    help='Length of trajectory.')
parser.add_argument('--length-test', type=int, default=10000,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100,
                    help='How often to sample the trajectory.')
parser.add_argument('--n-balls', type=int, default=5,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--savefolder', type=str, default='springcharge_physerrors__1',
                    help='name of folder to save everything in')
parser.add_argument('--sim-type', type=str, default='springcharge',
                    help='Type of simulation system')
parser.add_argument('--sigmatest', type=str, default=True,
                    help='Test the effect of perturbing state w/ a sample from Gaussian distribution w/ different sigma')
parser.add_argument('--animationofchaos', type=str, default=False,
                    help='Animation visualisation of chaos theory')

args = parser.parse_args()
os.makedirs(args.savefolder)
par_file = open(os.path.join(args.savefolder,'sim_args.txt'),'w')
print(args, file=par_file)
par_file.flush()
par_file.close()

# no noise-useless for our investigation
if args.sim_type == 'springcharge':
    sim = SpringChargeSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

elif args.sim_type == 'springchargequad':
    sim = SpringChargeQuadSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

elif args.sim_type == 'springquad':
    sim = SpringQuadSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

elif args.sim_type == 'springchargefspring':
    sim = SpringChargeFspringSim(noise_var=0.0, n_balls=args.n_balls, box_size=5.0)

np.random.seed(args.seed)

def generate_mse_pertime(num_sims, length, sample_freq):
    loc_all = []
    vel_all = []
    for i in range(num_sims):
        t = time.time()
        plot = False
        if i % 10 == 0:
            plot = True
        loc_mse, vel_mse, sigma= sim.sample_trajectory(args.savefolder , plotcurrent = plot, T=length, sample_freq=sample_freq)
        # dim 0 is the batch dimension, dim 1 is different sigma, dim 2 is time, dim 3 is (x,y)
        loc_all.append(loc_mse)
        vel_all.append(vel_mse)
    mse_loc = (np.asarray(loc_all).mean(axis=0)) / num_sims
    mse_vel = (np.asarray(vel_all).mean(axis=0)) / num_sims
    mse_loc_var = (np.asarray(loc_all).std(axis=0)) / num_sims
    mse_vel_var = (np.asarray(vel_all).std(axis=0)) / num_sims
    return mse_loc, mse_vel, sigma, mse_loc_var, mse_vel_var

if args.sigmatest:
    print("Calculating MSE over time due to computational errors")
    mse_loc, mse_vel, sigma, mse_loc_var, mse_vel_var= generate_mse_pertime(args.num_train, args.length, args.sample_freq)
    # dim 0 on mse is different sigma, dim 1 is different times along the motion, then (x,y)
    mse_model = mse_loc.mean(axis = 2)
    mse_vel_model = mse_vel.mean(axis = 2)
    np.save(os.path.join(args.savefolder, 'mse_model_pos.npy'), mse_model)
    np.save(os.path.join(args.savefolder, 'mse_model_vel.npy'), mse_model)
    np.save(os.path.join(args.savefolder, 'sigma.npy'), sigma)
    mse_loc_mean = (np.asarray(mse_loc).mean(axis=1).mean(axis=1))
    fig = plt.figure()
    plt.plot(sigma, mse_loc_mean)
    plt.xlabel('Sigma of Gaussian sampled/(arbitrary units)')
    plt.ylabel('Averaged Mean Square Error/(arbitrary units)')
    plt.show()

    y = sigma
    x = np.arange(0,int(args.length / args.sample_freq - 1)*0.001, 0.001)
    Z = (np.asarray(mse_loc).mean(axis=2))
    Z_plussigma = (np.asarray(mse_loc + mse_loc_var)).mean(axis=2)
    Z_minussigma = (np.asarray(mse_loc - mse_loc_var)).mean(axis=2)
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    cmaps = [matplotlib.cm.get_cmap('plasma')(i) for i in np.linspace(0,1,100)]
    for i in range(len(cmaps)):
        r, g, b, a = cmaps[i]
        a = 0.25
        cmaps[i] = (r, g, b, a)
    cmaps = mcolors.ListedColormap(cmaps)
    ax = plt.axes(projection = '3d')
    pl = ax.plot_surface(X,Y,Z, cmap = 'plasma', edgecolor= 'none')
    # ax.plot_surface(X,Y,Z_minussigma, cmap = cmaps, edgecolor= 'none')
    # ax.plot_surface(X,Y,Z_plussigma, cmap = cmaps, edgecolor= 'none')
    # fig.colorbar(pl, shrink=0.5, aspect=5)
    ax.set_ylabel('Sigma of Gaussian sampled')
    ax.set_xlabel('Time along trajectory')
    ax.set_zlabel('Averaged Mean Square Error')
    ax.view_init(45, -35)
    fig.savefig(os.path.join(args.savefolder, 'sigmamse.png'))
    ax.view_init(75,-35)
    fig.savefig(os.path.join(args.savefolder, 'sigmamse_birdseye.png'))
    ax.view_init(45,225)
    fig.savefig(os.path.join(args.savefolder, 'sigmamse_rotated.png'))
    plt.show()
    # for i in range(len(mse_loc[0])):
    #         fig = plt.figure()
    #         mse_loc_mean = (np.asarray(mse_loc).mean(axis=2))
    #         timeslot = i*0.001
    #         plt.plot(delta_T, mse_loc_mean[:,i])
    #         plt.show()
if args.animationofchaos:
    if args.sim_type == 'springcharge':
        sim.sample_trajectory_animation(args.savefolder, T=args.length, sample_freq=args.sample_freq)
    else:
        print('Animation only implemented for springcharge Model')