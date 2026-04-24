# grasp2sim

A MuJoCo-based simulation framework for evaluating **6-DoF grasp poses** from the [GraspNet-1Billion](https://graspnet.net) dataset using a Franka Emika Panda gripper.

## Overview

`grasp2sim` bridges the GraspNet dataset and physics simulation. Given a set of grasp poses $G = [R \mid t \mid w]$ — where $R \in \mathbb{R}^{3 \times 3}$ is the gripper orientation, $t \in \mathbb{R}^{3}$ the grasp center, and $w$ the gripper width — it simulates each grasp and evaluates success based on a **lift height** threshold $\Delta z \geq 0.08\,\text{m}$.

## How It Works

1. **Scene generation** (`grasp2scene.py`): reads object 6D poses from GraspNet annotations and generates a MuJoCo XML scene with dynamic objects and a kinematic Panda hand.

2. **Grasp publishing** (`publisher.py`): loads grasp labels, filters by friction coefficient ($\mu \leq \text{thresh}$) and collision, then builds a `GraspGroup`.

3. **Simulation** (`grasp_sim.py`): for each grasp pose, transforms it from camera frame to world frame via:

$$T_{\text{world}} = T_{\text{cam} \to \text{table}} \cdot T_{\text{cam}}$$

   then executes an approach → close → lift sequence. Success is determined by checking $\Delta z$ of target objects post-lift.

## Grasp Evaluation

A grasp is considered **successful** if at least one object is lifted by $\Delta z \geq 0.08\,\text{m}$ after the full approach-close-lift sequence.

## Notes

- Rendering uses EGL offscreen (`MUJOCO_GL=egl`) for headless environments.