"""
This code is based on https://github.com/ethanfetaya/NRI
(MIT licence)
"""
from synthetic_sim_comperrors import *
import time
import numpy as np
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--num-train', type=int, default=100,
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
parser.add_argument('--savefolder', type=str, default='springcharge_comperrors',
                    help='name of folder to save everything in')
parser.add_argument('--sim-type', type=str, default='springcharge',
                    help='Type of simulation system')
parser.add_argument('--timesteptest', type=str, default=True,
                    help='Generate many datasets with different timesteps')
parser.add_argument('--animation', type=str, default=False,
                    help='Generate animation of many datasets with different timesteps')

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
    _delta_T_ = []
    for i in range(num_sims):
        t = time.time()
        loc_mse, vel_mse, delta_T = sim.sample_trajectory(T=length, sample_freq=sample_freq)
        # dim 0 is the batch dimension, dim 1 is time: already averaged MSE over particles
        loc_all.append(loc_mse)
        vel_all.append(vel_mse)
        _delta_T_ = delta_T
    mse_loc = (np.asarray(loc_all).mean(axis=0)) / num_sims
    mse_vel = (np.asarray(vel_all).mean(axis=0)) / num_sims

    return mse_loc, mse_vel, delta_T

if (args.timesteptest):
    print("Calculating MSE over time due to computational errors")
    mse_loc, mse_vel, delta_T = generate_mse_pertime(args.num_train, args.length, args.sample_freq)
    # dim 0 on mse is different delta_T, dim 1 is different times along the motion, then (x,y)
    # [0,:,:] is the delta_T that the model gets
    mse_model = mse_loc[0].mean(axis=1)
    mse_model_vel = mse_vel[0].mean(axis=1)
    np.save(os.path.join(args.savefolder, 'mse_model_pos.npy'), mse_model)
    np.save(os.path.join(args.savefolder, 'mse_model_vel.npy'), mse_model_vel)
    mse_loc_mean = (np.asarray(mse_loc).mean(axis=1).mean(axis=1))
    fig = plt.figure()
    plt.plot(delta_T, mse_loc_mean)
    plt.xlabel('Time-step of Integration/(arbitrary units)')
    plt.ylabel('Averaged Mean Square Error/(arbitrary units)')
    plt.show()
    """
    https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
    """
    y = delta_T
    x = np.arange(0,int(args.length / args.sample_freq - 1)*0.001, 0.001)
    Z = (np.asarray(mse_loc).mean(axis=2))
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot_surface(X,Y,Z, cmap = 'viridis', edgecolor= 'none')
    ax.set_ylabel('Time-step of Integration')
    ax.set_xlabel('Time along trajectory')
    ax.set_zlabel('Averaged Mean Square Error')
    ax.view_init(45, -35)
    fig.savefig(os.path.join(args.savefolder, 'timestepmse.png'))
    ax.view_init(75,-35)
    fig.savefig(os.path.join(args.savefolder, 'timestepmse_birdseye.png'))
    ax.view_init(45,225)
    fig.savefig(os.path.join(args.savefolder, 'timestepmse_rotated.png'))
    plt.show()
    # for i in range(len(mse_loc[0])):
    #         fig = plt.figure()
    #         mse_loc_mean = (np.asarray(mse_loc).mean(axis=2))
    #         timeslot = i*0.001
    #         plt.plot(delta_T, mse_loc_mean[:,i])
    #         plt.show()
if (args.animation):
    if args.sim_type == 'springcharge':
        sim.sample_trajectory_animation(T=args.length, sample_freq=args.sample_freq)
    else:
        print('Animation only implemented for springcharge Model')