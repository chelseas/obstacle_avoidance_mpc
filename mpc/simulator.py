"""Define functions for simulating the performance of an MPC controller"""
from typing import Optional

import casadi
import numpy as np
from tqdm import tqdm
import time
import math

from mpc.dynamics_constraints import DynamicsFunction
from mpc.mpc import solve_MPC_problem
from mpc.nn import PolicyCloningModel

def core_simulation_steps(x_t, u_t, substeps, dynamics_fn, n_states, dt, clip=[], clip_lims=[]):
    # Update the state using the dynamics. Integrate at a higher frequency using
    # zero-order hold controls
    x_tp1 = np.zeros(len(x_t))
    for _ in range(substeps):
        dx_dt = dynamics_fn(x_t, u_t)
        for i in range(n_states):
            if clip and clip[i]:
                L,U = clip_lims
                x_tp1_i = x_t[i] + dt / substeps * np.array(dx_dt[i])
                x_tp1[i] = np.clip(x_tp1_i, L[i], U[i]) 
            else:
                x_tp1[i] = x_t[i] + dt / substeps * np.array(dx_dt[i])
        x_t = x_tp1
    
    return x_tp1

def simulate_mpc(
    opti: casadi.Opti,
    x0_variables: casadi.MX,
    u0_variables: casadi.MX,
    x0: np.ndarray,
    dt: float,
    dynamics_fn: DynamicsFunction,
    n_steps: int,
    verbose: bool = False,
    x_variables: Optional[casadi.MX] = None,
    u_variables: Optional[casadi.MX] = None,
    substeps: int = 1,
    clip=[],
    clip_lims=[]
):
    """
    Simulate a rollout of the MPC controller specified by the given optimization problem.

    args:
        opti: an optimization problem
        x0_variables: the variables representing the start state of the MPC problem
        u0_variables: the variables representing the control input in the MPC problem at
            the first timestep in the MPC problem
        x0: the starting state of the system
        dt: the timestep used for integration
        dynamics_fn: the dynamics of the system
        n_steps: how many total steps to simulate
        verbose: if True, print the results of the optimization. Defaults to False
        x_variables, u_variables, x_guess, and u_guess allow you to provide an initial
            guess for x and u (often from the previous solution). If not provided, use
            the default casadi initial guess (zeros).
        substeps: how many smaller substeps to use for the integration
    returns:
        - an np.ndarray of timesteps
        - an np.ndarray of states
        - an np.ndarray of control inputs
    """
    n_states = x0_variables.shape[1]
    n_controls = u0_variables.shape[1]
    # Create some arrays to store the results
    t = dt * np.linspace(0, dt * (n_steps - 1), n_steps)
    assert t.shape[0] == n_steps
    x = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps - 1, n_controls))

    # Set the initial conditions
    x[0] = x0

    # Track how often the MPC problem is infeasible
    n_infeasible = 0

    # Initialize empty guesses for the MPC problem
    x_guess: Optional[np.ndarray] = None
    u_guess: Optional[np.ndarray] = None

    # Simulate
    t_range = tqdm(range(n_steps - 1))
    t_range.set_description("Simulating")  # type: ignore
    for tstep in t_range:
        # Solve the MPC problem to get the next state
        success, u_current, x_guess, u_guess = solve_MPC_problem(
            opti.copy(),
            x0_variables,
            u0_variables,
            x[tstep],
            verbose,
            x_variables,
            u_variables,
            x_guess,
            u_guess,
        )

        if success:
            u[tstep] = u_current
        else:
            n_infeasible += 1

        # step
        x_t = np.array(x[tstep])
        u_t = u[tstep]
        x_tp1 = core_simulation_steps(x_t, u_t, substeps, dynamics_fn, n_states, dt, clip=clip, clip_lims=clip_lims)
        x[tstep + 1] = x_tp1

    print(f"{n_infeasible} infeasible steps")

    return t, x, u


def simulate_nn(
    policy: PolicyCloningModel,
    x0: np.ndarray,
    dt: float,
    dynamics_fn: DynamicsFunction,
    n_steps: int,
    substeps: int = 1,
    clip=[],
    clip_lims=[]
):
    """
    Simulate a rollout of a neural controller

    args:
        policy: a neural control policy
        x0: the starting state of the system
        dt: the timestep used for integration
        dynamics_fn: the dynamics of the system
        n_steps: how many total steps to simulate
        substeps: how many smaller substeps to use for the integration
    returns:
        - an np.ndarray of timesteps
        - an np.ndarray of states
        - an np.ndarray of control inputs
    """
    n_states = policy.n_state_dims
    n_controls = policy.n_control_dims
    # Create some arrays to store the results
    t = dt * np.linspace(0, dt * (n_steps - 1), n_steps)
    assert t.shape[0] == n_steps
    x = np.zeros((n_steps, n_states))
    u = np.zeros((n_steps - 1, n_controls))

    # collect some statistics 
    x_max = np.ones(policy.n_state_dims)*(-math.inf)
    x_min = np.ones(policy.n_state_dims)*(math.inf)

    # Set the initial conditions
    x[0] = x0

    # Simulate
    t_range = tqdm(range(n_steps - 1))
    t_range.set_description("Simulating")  # type: ignore
    avg_nn_etime = 0
    for tstep in t_range:
        # Solve the MPC problem to get the next state
        t1 = time.time()
        u_current = policy.eval_np(x[tstep])
        avg_nn_etime += time.time() - t1 
        u[tstep] = u_current

        # step
        x_t = np.array(x[tstep])
        u_t = u[tstep]
        x_tp1 = core_simulation_steps(x_t, u_t, substeps, dynamics_fn, n_states, dt, clip=clip, clip_lims=clip_lims)
        x[tstep + 1] = x_tp1

        # keep track of stuff
        x_max = np.maximum(x_max, x_tp1)
        x_min = np.minimum(x_max, x_tp1)

    avg_nn_etime /= len(t_range)
    print("Avg NN execution time: (miliseconds) ", avg_nn_etime*1000)

    return t, x, u, x_max, x_min
