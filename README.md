# Operational Space Control Barrier Functions

Code for *"Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions"* -- Daniel Morton and Marco Pavone

Submitted to IROS 2025, Hangzhou

## What is OSCBF?

This is a safe, high-performance, and easy-to-use controller / safety filter for robotic manipulators.

With OSCBF, you can...
- Operate at kilohertz speed even with over 400 active safety constraints
- Design safety constraints (barrier functions) easily via CBFpy
- Enforce safety on both torque-controlled and velocity-controlled robots
- Either apply a safety filter on top of your existing controller, or use our provided controller

In general, this will be especialy useful for enforcing safety during **teleoperation** or while executing **learned policies**

For more details and videos, check out the [project webpage](https://stanfordasl.github.io/oscbf/) as well as the [CBFpy documentation](https://danielpmorton.github.io/cbfpy/).


## How do I use this?

Check out the `examples/` folder for interactive demos in Pybullet! This is the best place to start

If you're applying this to a different robot, you'll need to provide a URDF -- we can parse the kinematics and dynamics from that. Some notes:
- I haven't written an MJCF parser yet, but it should be feasible
- All joints that shouldn't be controlled as part of the kinematic chain should be set to `fixed` (gripper joints, for instance). Since you probably want to still be able to use the gripper joints in simulation, the best way to handle this is to make a copy of the URDF: load the non-fixed one in sim, and parse the fixed one with OSCBF
- I manually defined the collision model for the Franka. There are probably better ways to parse or generate this data from meshes, but I haven't done it yet. For now, I'd recommend doing the same for your robot.

## FAQ

- "I already have a well-tuned operational space controller! I don't want to replace that"
  - You can use OSCBF as a safety filter on top of your existing controller!
- "I'm working with a mobile manipulator. Does this still work?"
  - Yes! We support prismatic and revolute joints, so adding a mobile base just adds 3DOF (PPR) to the beginning of the kinematic chain.
- "I'm controlling joint-space motions -- does this still apply?"
  - Depending on the task, you will likely want to modify the objective function. But, the robot dynamics and CBF formulation will still be useful!


## Installation

A virtual environment is optional, but highly recommended. For `pyenv` installation instructions, see [here](https://danielpmorton.github.io/cbfpy/pyenv).

```
git clone https://github.com/stanfordasl/oscbf
cd oscbf
pip install -e .
```

Note: This code will work with most versions of `jax`, but there seems to have been a CPU slowdown introduced in version `0.4.32`. To avoid this, I use version `0.4.30`, which is the version indicated in this repo's `pyproject.toml` as well. However, feel free to use any version you like.


## Documentation

See the CBFpy documentation, available at [this link](https://danielpmorton.github.io/cbfpy)


## Citation
```
@article{morton2025oscbf,
      author = {Morton, Daniel and Pavone, Marco},
      title = {Safe, Task-Consistent Manipulation with Operational Space Control Barrier Functions},
      year = {2025},
      journal = {arXiv preprint arXiv:2503.06736},
      note = {Submitted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Hangzhou, 2025},
      }
```

