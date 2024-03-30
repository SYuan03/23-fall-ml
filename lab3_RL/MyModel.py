import math
from typing import Optional

from gymnasium.envs.classic_control import CartPoleEnv


# 创建一个新的 CartPole 环境类，继承自原始的 CartPoleEnv
class CustomCartPoleEnv(CartPoleEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        super(CustomCartPoleEnv, self).__init__(render_mode=render_mode)

        # 修改 theta_threshold_radians 参数为 ±15度
        self.theta_threshold_radians = 15 * 2 * math.pi / 360

# # 注册新的 CartPole 环境
# gym.envs.register(
#     id='CustomCartPole-v1',
#     entry_point='xxxxx:CustomCartPoleEnv',
#     max_episode_steps=500,  # 可以根据需要修改
#     reward_threshold=475.0,  # 可以根据需要修改
# )
#
# # 创建新的环境实例
# custom_env = gym.make('CustomCartPole-v1')
