from typing import (
    Tuple,
)

import torch
import numpy as np

from utils_types import (
    BatchAction,
    BatchDone,
    BatchNext,
    BatchReward,
    BatchState,
    TensorStack5,
    TorchDevice,
)

from utils_sumtree import SumTree


class ReplayMemory(object):

    def __init__(
            self,
            channels: int,
            capacity: int,
            device: TorchDevice,
    ) -> None:
        self.__device = device
        self.__capacity = capacity
        self.tree = SumTree(capacity)
        self.belta = 0.4
        self.belta_increment_per_sampling = 0.001
        self.__size = 0
        # self.__pos = 0
        #
        # self.__m_states = torch.zeros(
        #     (capacity, channels, 84, 84), dtype=torch.uint8)
        # self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        # self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        # self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            p: float,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        self.tree.add(p, (folded_state, action, reward, done))
        # self.__m_states[self.__pos] = folded_state
        # self.__m_actions[self.__pos, 0] = action
        # self.__m_rewards[self.__pos, 0] = reward
        # self.__m_dones[self.__pos, 0] = done
        #
        # self.__pos = (self.__pos + 1) % self.__capacity
        self.__size = max(self.__size, self.tree.data_pointer)

    def sample(self, batch_size: int) -> Tuple[
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
    ]:
        pri_step = self.tree.total_p / batch_size
        self.belta = np.min(1, self.belta + self.belta_increment_per_sampling)
        min_p = self.tree.min_p / self.tree.total_p
        min_p = 0.0001 if min_p == 0 else min_p
        b_state = []
        b_next = []
        b_action = []
        b_reward = []
        b_done = []
        ISweight = []
        for i in range(batch_size):
            st, ed = pri_step * i, pri_step * (i + 1)
            v = torch.rand(st, ed, size=(1,))
            indec, p, (m_states, m_actions, m_rewards, m_dones) = self.tree.get_leaf(v)

            b_state.append(m_states[:4])
            b_next.append(m_states[1:])
            b_action.append(m_actions)
            b_reward.append(m_rewards)
            b_done.append(m_dones)
            ISweight.append(np.power(p / min_p, self.belta))
        b_state = torch.Tensor(b_state).to(self.__device)
        b_next = torch.Tensor(b_next).to(self.__device)
        b_action = torch.Tensor(b_action).to(self.__device)
        b_reward = torch.Tensor(b_reward).to(self.__device)
        b_done = torch.Tensor(b_done).to(self.__device)
        return b_state, b_action, b_reward, b_next, b_done

    def __len__(self) -> int:
        return self.__size
