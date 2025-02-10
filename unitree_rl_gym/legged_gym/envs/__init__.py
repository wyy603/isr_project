from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())


from legged_gym.envs.rma.humanoid_rma import H12RMAEnv
from legged_gym.envs.rma.humanoid_rma_config import H12RMACfg, H12RMACfgPPO, H12RMAAdaptationCfgPPO
task_registry.register( "h12_rma", H12RMAEnv, H12RMACfg(), H12RMACfgPPO())
task_registry.register( "h12_adaption", H12RMAEnv, H12RMACfg(), H12RMAAdaptationCfgPPO())