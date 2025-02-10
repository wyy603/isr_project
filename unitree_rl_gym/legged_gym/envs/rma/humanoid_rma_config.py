from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO


class H12RMACfg(H1_2RoughCfg):
    """
    Configuration class for the H12 humanoid robot.
    """
    class env(H1_2RoughCfg.env):
        num_height_points = 187
        frame_stack = 15
        num_observations = 47
        num_privileged_obs = 50 + num_height_points
        num_rma_obs = 50 + num_height_points

    class terrain(H1_2RoughCfg.terrain):
        # mesh_type = 'trimesh'
        mesh_type = 'trimesh' # 只是用于调试

    class commands(H1_2RoughCfg.commands):
        class ranges:
            lin_vel_x = [2.0, 6.0] # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]

class H12RMACfgPPO(H1_2RoughCfgPPO):
    runner_class_name = "RMARunner"
    class runner( H1_2RoughCfgPPO.runner ):
        policy_class_name = 'ActorCriticRecurrentRMA'
        algorithm_class_name = 'RMA'
        run_name = 'rma'
        experiment_name = 'H12_rma'

        # load and resume
        resume = True
        load_run = "2024-12-26_17-16-14_wyy-rma-teacher"
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class policy:
        init_noise_std = 0.8
        actor_hidden_dims = [128, 32]
        critic_hidden_dims = [128, 32]
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        rnn_type = 'lstm'
        rnn_hidden_size = 256
        rnn_num_layers = 1
    
    class rma:
        num_latents = 128

class H12RMAAdaptationCfgPPO(H12RMACfgPPO):
    runner_class_name = "RMAAdaptationRunner"
    class runner( H12RMACfgPPO.runner ):
        load_teacher = True
        policy_class_name = 'ActorCriticRecurrentAdaption'
        algorithm_class_name = 'RMAAdaptation'
        run_name = 'adaptation'
        experiment_name = 'H12_adaptation'

        resume = False
        load_run = -1
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt

    class rma( H12RMACfgPPO.rma ):
        teacher_load_run = "2024-12-26_18-26-21_wyy-rma-teacher"
        teacher_checkpoint = -1
        teacher_experiment_name = "H12_rma"
        teacher_run_name = 'rma'

#2024-12-23_15-20-44_wyy-rma-teacher v=4