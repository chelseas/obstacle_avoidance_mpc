"""Test the obstacle avoidance MPC for a quadrotor"""

import sys
sys.path.append('/home/smkatz/Documents/obstacle_avoidance_mpc/')

import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math

from NNet.converters.onnx2nnet import onnx2nnet

from mpc.costs import (
    lqr_running_cost,
    distance_travelled_terminal_cost,
)
from mpc.dynamics_constraints import quad6d_dynamics
from mpc.mpc import construct_MPC_problem, solve_MPC_problem
from mpc.obstacle_constraints import hypersphere_sdf
from mpc.simulator import simulate_nn, simulate_mpc
from mpc.network_utils import pytorch_to_nnet

from mpc.nn import PolicyCloningModel

# parameters for the obstacle
<<<<<<< HEAD
radius = np.sqrt(2)
margin = 0.25
=======
radius = 1.0
margin = 0.5
>>>>>>> 5ba0cc330faa9f4602ebac9f2a93260b040541ce
center = [0.0, 0.0, 2.5] # should y be 1e-5 ?
n_states = 6
horizon = 20
n_controls = 3
dt = 1.0
dynamics_fn = quad6d_dynamics

state_space = [
    (-5.5, 3.),  # px
    (-5.5, 5.5),  # py
    (2.0, 3.0),  # pz
    (-1.0, 1.0),  # vx
    (-1.0, 1.0),  # vy
    (-1.0, 1.0),  # vz
]

# define state clipping limits 
clip = [False, False, False, True, True, True] # clip velocity
clip_lims = ([0.,0.,0.,-1.0, -1.0, -1.0],
                [0.,0.,0.,1.0, 1.0, 1.0])

def define_quad_mpc_expert():
    # -------------------------------------------
    # Define the MPC problem
    # -------------------------------------------

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1, 2], center), margin)]

    # Define costs to make the quad go to the right
    x_goal = np.array([100.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([0.0, 0.0, 0.0, 0.1, 0.1, 0.1]), 1 * np.eye(3)
    )
    terminal_cost_fn = lambda x: distance_travelled_terminal_cost(x, goal_direction)
    # terminal_cost_fn = lambda x: squared_error_terminal_cost(x, x_goal)

    # Define control bounds
    control_bounds = [np.pi / 10, np.pi / 10, 2.0]

    # Define MPC problem
    opti, x0_variables, u0_variables, x_variables, u_variables = construct_MPC_problem(
        n_states,
        n_controls,
        horizon,
        dt,
        dynamics_fn,
        obstacle_fns,
        running_cost_fn,
        terminal_cost_fn,
        control_bounds,
        clip=clip,
        clip_lims=clip_lims
    )

    # Wrap the MPC problem to accept a tensor and return a tensor
    def mpc_expert(current_state: torch.Tensor) -> torch.Tensor:
        _, control_output, _, _ = solve_MPC_problem(
            opti.copy(),
            x0_variables,
            u0_variables,
            current_state.detach().numpy(),
        )

        return torch.from_numpy(control_output)

    return mpc_expert


def clone_quad_mpc(save_path, hidden_layers=2, hidden_layer_width=32, lambd=1, train=True, data_path=None, load_from_file=None, epochs=50, n_pts=int(1e5), learning_rate=1e-3):
    # -------------------------------------------
    # Clone the MPC policy
    # -------------------------------------------
    mpc_expert = define_quad_mpc_expert()
    # hidden_layers = 2
    # hidden_layer_width = 32
    cloned_policy = PolicyCloningModel(
        hidden_layers,
        hidden_layer_width,
        n_states,
        n_controls,
        state_space,
        load_from_file=load_from_file,
    )

    # n_pts = int(1e5)
    n_epochs = epochs
    if train:
        print("Training...")
        print("Using ", lambd, " weight on regularization loss")
        cloned_policy.clone(
            mpc_expert,
            n_pts,
            n_epochs,
            learning_rate,
            save_path=save_path,
            data_path=data_path,
            lambd=lambd
        )

    return cloned_policy

def generate_quad_data(npts, save_file):
    mpc_expert = define_quad_mpc_expert()
    hidden_layers = 2
    hidden_layer_width = 32
    cloned_policy = PolicyCloningModel(
        hidden_layers,
        hidden_layer_width,
        n_states,
        n_controls,
        state_space  # ,
        #load_from_file="mpc/tests/data/cloned_quad_policy.pth",
    )

    cloned_policy.gen_training_data(mpc_expert, npts, save_file)

def simulate_and_plot(policy, savename="nn_policy.png", n_steps=20):
    # -------------------------------------------
    # Plot a rollout of the cloned
    # -------------------------------------------
    ys = np.linspace(-0.25, 0.25, 3)
    xs = np.linspace(-5.25, -4.75, 3)
    vxs = np.linspace(0.96, 0.98, 3)
    vys = np.linspace(-0.5, 0.5, 3)
    x0s = []
    for y in ys:
        for x in xs:
            for vx in vxs:
                for vy in vys:
                    x0s.append(np.array([x, y, 2.5, vx, vy, 0.0]))

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)

    # n_steps = 20
    x_max = np.ones(policy.n_state_dims)*(-math.inf)
    x_min = np.ones(policy.n_state_dims)*(math.inf)
    for x0 in x0s:
        _, x, u, x_max_i, x_min_i = simulate_nn(
            policy,
            x0,
            dt,
            dynamics_fn,
            n_steps,
            substeps=10,
            clip=clip,
            clip_lims=clip_lims
        )
        x_max = np.maximum(x_max_i, x_max)
        x_min = np.minimum(x_min_i, x_min)

        # Plot it (in x-y plane)
        ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-", linewidth=1)
        # and in (x-z plane)
        # ax_xz.plot(x0[0], x0[2], "ro")
        ax_xz.plot(x[:, 0], x[:, 2], "r-", linewidth=1)
    
    print("max state values: ", x_max)
    print("min state values: ", x_min)

    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    obs_z = radius * np.sin(theta) + center[2]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    margin_z = (radius + margin) * np.sin(theta) + center[2]
    ax_xy.plot(obs_x, obs_y, "k-")
    ax_xy.plot(margin_x, margin_y, "k:")
    ax_xy.plot([-1., 1., 1., -1., -1.], [-1, -1, 1, 1, -1], "r-", label="the box")
    ax_xz.plot(obs_x, obs_z, "k-", label="Obstacle")
    ax_xz.plot(margin_x, margin_z, "k:", label="Safety margin")

    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy.set_xlim([-6., 10.])
    ax_xy.set_ylim([-5.0, 5.0])
    ax_xz.set_xlim([-6., 10.])
    ax_xz.set_ylim([-0.0, 5.0])

    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")

    ax_xz.legend()

    plt.savefig(savename)

def save_to_onnx(policy, save_path):
    """Save to an onnx file"""
    pytorch_to_nnet(policy, policy.n_state_dims, policy.n_control_dims, save_path)

    input_mins = [state_range[0] for state_range in state_space]
    input_maxes = [state_range[1] for state_range in state_space]
    means = [0.5 * (state_range[0] + state_range[1]) for state_range in state_space]
    means += [0.0]
    ranges = [state_range[1] - state_range[0] for state_range in state_space]
    ranges += [1.0]
    onnx2nnet(save_path, input_mins, input_maxes, means, ranges)


if __name__ == "__main__":
<<<<<<< HEAD
    generate_quad_data(int(1e5), 'mpc/tests/data/quad_data_3')
    model_save_path = "mpc/tests/data/quad_policy_3"
    policy = clone_quad_mpc(model_save_path+'.pth', hidden_layer_width=16, hidden_layers=4, lambd=1e-12, train=True, epochs=50, data_path='mpc/tests/data/quad_data_2',) #, load_from_file=model_load_path) # data_path='mpc/tests/data/quad_mpc_data',
    save_to_onnx(policy, model_save_path+".onnx")
    simulate_and_plot(policy)
=======
    suffix = "5e5"
    data_path = 'mpc/tests/data/quad_data_'+suffix
    generate_quad_data(int(5e5), data_path)
    model_save_path = "mpc/tests/data/quad_policy_"+suffix
    # model_load_path = model_save_path+".pth"
    policy = clone_quad_mpc(model_save_path+'.pth', 
                            hidden_layer_width=16, 
                            hidden_layers=2, 
                            lambd=0.0, 
                            train=True, 
                            epochs=50,
                            data_path=data_path) # load_from_file=model_load_path) # data_path='mpc/tests/data/quad_mpc_data', n_pts=1e5
    save_to_onnx(policy, model_save_path+".onnx")
    simulate_and_plot(policy, savename="nn_policy_"+suffix+".png", n_steps=20)
>>>>>>> 5ba0cc330faa9f4602ebac9f2a93260b040541ce
