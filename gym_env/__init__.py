# coding: utf-8
# Copyright (c) 2024 Jungheil <jungheilai@gmail.com>
# iLQR_cartpole is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#     http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR
# PURPOSE.
# See the Mulan PSL v2 for more details.

from gym.envs.registration import register

register(
    id="CartPoleContinuous-v0",
    entry_point="gym_env.cartpole_continuous_env:CartPoleContinuousEnv",
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id="CartPoleContinuous-v1",
    entry_point="gym_env.cartpole_continuous_env:CartPoleContinuousEnv",
    max_episode_steps=500,
    reward_threshold=475.0,
)
