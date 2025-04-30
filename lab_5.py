#@title Import packages
import time
import numpy as np
import math
# Graphics and plotting.
import mediapy as media
# Mujoco, MJX, and Brax
import jax
from jax import numpy as jp
import argparse
import create_env

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

N_STEPS = 200
RENDER_EVERY = 2
fix_base = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_base', action='store_true', help='Fix the base position and orientation')
    parser.set_defaults(fix_base=False)
    return parser.parse_args()

def get_target_joint_position(t):
    frequency = 20
    amplitude = 0.3
    offset_front_legs = 0
    offset_rear_legs = 0.2
    hip = amplitude * math.sin(frequency * t)
    knee = amplitude * math.cos(frequency * t)
    return [hip, 0, knee] + [hip, 0, knee] + [-hip - offset_rear_legs, 0, -knee] + [-hip + offset_rear_legs, 0, -knee]

def main():
    args = parse_args()
    env = create_env.create_env()
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    # initialize the state
    rng = jax.random.PRNGKey(1)

    # grab a trajectory
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    for i in range(N_STEPS):
        prev_t = time.time()
        t = env.dt * i
        target_joint_position = get_target_joint_position(t)
        ctrl = jp.array(target_joint_position)
        state = jit_step(state, ctrl)
        
        if args.fix_base:
            qpos = state.pipeline_state.qpos.at[:7].set([0,0,0.5, 1.0, 0,0,0])
            qvel = state.pipeline_state.qvel.at[:6].set([0,0,0, 0,0,0])
            state = state.tree_replace({"pipeline_state.qpos":qpos,"pipeline_state.qvel":qvel})

        rollout.append(state.pipeline_state)

    media.write_video('v.mp4',
        env.render(rollout[::RENDER_EVERY], camera='tracking_cam'),
        fps=1.0 / env.dt / RENDER_EVERY)

if __name__ == "__main__":
    main()

