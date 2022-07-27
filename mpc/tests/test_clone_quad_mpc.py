"""Test the obstacle avoidance MPC for a quadrotor"""

import sys
sys.path.append('/home/smkatz/Documents/obstacle_avoidance_mpc/')

import numpy as np
import torch
import matplotlib.pyplot as plt
import time

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


radius = 0.2
margin = 0.1
center = [0.0, 1e-5, 0.0]
n_states = 6
horizon = 20
n_controls = 3
dt = 0.1
dynamics_fn = quad6d_dynamics

state_space = [
    (-1.5, 1.5),  # px
    (-1.0, 1.0),  # py
    (-1.0, 1.0),  # pz
    (-1.0, 1.0),  # vx
    (-1.0, 1.0),  # vy
    (-1.0, 1.0),  # vz
]


def define_quad_mpc_expert():
    # -------------------------------------------
    # Define the MPC problem
    # -------------------------------------------

    # Define obstacle as a hypercylinder (a sphere in xyz and independent of velocity)
    obstacle_fns = [(lambda x: hypersphere_sdf(x, radius, [0, 1, 2], center), margin)]

    # Define costs to make the quad go to the right
    x_goal = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    goal_direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    running_cost_fn = lambda x, u: lqr_running_cost(
        x, u, x_goal, dt * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 1 * np.eye(3)
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


def clone_quad_mpc(save_path, hidden_layers=2, hidden_layer_width=32, lambd=1, train=True, data_path=None, load_from_file=None, epochs=50, n_pts=int(1e5)):
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
    learning_rate = 1e-3
    if train:
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

def simulate_and_plot(policy):
    # -------------------------------------------
    # Plot a rollout of the cloned
    # -------------------------------------------
    ys = np.linspace(-0.5, 0.5, 3)
    xs = np.linspace(-1.0, -0.3, 3)
    vxs = np.linspace(-0.5, 0.5, 3)
    vys = np.linspace(-0.5, 0.5, 3)
    x0s = []
    for y in ys:
        for x in xs:
            for vx in vxs:
                for vy in vys:
                    x0s.append(np.array([x, y, 0.0, vx, vy, 0.0]))

    fig = plt.figure(figsize=plt.figaspect(1.0))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)

    n_steps = 20
    for x0 in x0s:
        _, x, u = simulate_nn(
            policy,
            x0,
            dt,
            dynamics_fn,
            n_steps,
            substeps=10,
        )

        # Plot it (in x-y plane)
        ax_xy.plot(x0[0], x0[1], "ro")
        ax_xy.plot(x[:, 0], x[:, 1], "r-", linewidth=1)
        # and in (x-z plane)
        # ax_xz.plot(x0[0], x0[2], "ro")
        ax_xz.plot(x[:, 0], x[:, 2], "r-", linewidth=1)

    # Plot obstacle
    theta = np.linspace(0, 2 * np.pi, 100)
    obs_x = radius * np.cos(theta) + center[0]
    obs_y = radius * np.sin(theta) + center[1]
    margin_x = (radius + margin) * np.cos(theta) + center[0]
    margin_y = (radius + margin) * np.sin(theta) + center[1]
    ax_xy.plot(obs_x, obs_y, "k-")
    ax_xy.plot(margin_x, margin_y, "k:")
    ax_xz.plot(obs_x, obs_y, "k-", label="Obstacle")
    ax_xz.plot(margin_x, margin_y, "k:", label="Safety margin")

    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    ax_xy.set_xlim([-1.5, 1.5])
    ax_xy.set_ylim([-1.0, 1.0])
    ax_xz.set_xlim([-1.5, 1.5])
    ax_xz.set_ylim([-1.0, 1.0])

    ax_xy.set_aspect("equal")
    ax_xz.set_aspect("equal")

    ax_xz.legend()

    plt.savefig('nn_policy.png')

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
    # generate_quad_data(100, 'mpc/tests/data/quad_small_TEST')
    model_save_path = "mpc/tests/data/TIMETEST"
    policy = clone_quad_mpc(model_save_path+'.pth', hidden_layer_width=12, hidden_layers=1, lambd=1e-7, train=True, epochs=10, n_pts=100) #, load_from_file=model_load_path) # data_path='mpc/tests/data/quad_mpc_data',
    # save_to_onnx(policy, model_save_path+".onnx")
    simulate_and_plot(policy)
