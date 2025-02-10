# Introduction
This is a project that I worked on with Ruinian Chang for the course "Intelligent Systems and Robotics." We utilized the [Unitree RL Gym](https://github.com/unitreerobotics/unitree_rl_gym) and its H1_2 robot model, selecting [Rapid Motor Adaptation (RMA)](https://arxiv.org/abs/2107.04034) as our method. Through continuous parameter tuning and multiple training iterations, we successfully enabled a legged robot to run stably at a speed of 4 m/s on randomly generated terrains with heights ranging from [-0.07, 0.07 m] in the Isaac Gym simulation environment.

https://github.com/user-attachments/assets/ff227833-eb70-4336-984e-eb3266653924

# Installation
The installation process is identical to that of the unitree_rl_gym repository.

# View Our Result
Run the script run2.sh to see our results.

# Introduction of RMA

![rma](https://github.com/user-attachments/assets/7a686caa-67dc-4e17-97a3-0b8e7e36ff02)

RMA first trains two policies using the PPO algorithm. The teacher policy receives both the robot's own parameters and environmental parameters to generate the base policy, while the student policy only receives the robot's parameters to generate the adaptation policy. After training both policies separately, a mean square loss is applied to their actions. This allows the student policy to walk on complex terrains without relying on environmental perception—instead, it adapts by sensing changes in its own posture.
