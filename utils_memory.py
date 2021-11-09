from typing import (
    Tuple,
)

import torch
import numpy as np

from utils_types import (
    BatchIndex,
    BatchAction,
    BatchDone,
    BatchNext,
    BatchPriority,
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
        self.epsilon = 0.01
        self.alpha = 0.6
        self.abs_err_upper = 1.0
        self.__pos = 0
        self.__m_states = torch.zeros(
            (capacity, channels, 84, 84), dtype=torch.uint8)
        self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

    def push(
            self,
            folded_state: TensorStack5,
            action: int,
            reward: int,
            done: bool,
    ) -> None:
        max_p = np.max(self.tree.max_p)
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p)
        self.__m_states[self.__pos] = folded_state
        self.__m_actions[self.__pos, 0] = action
        self.__m_rewards[self.__pos, 0] = reward
        self.__m_dones[self.__pos, 0] = done
       
        self.__pos = (self.__pos + 1) 
        self.__size = max(self.__size, self.__pos)
        self.__pos = (self.__pos + 1) % self.__capacity

    def sample(self, batch_size: int) -> Tuple[
            BatchIndex,
            BatchState,
            BatchAction,
            BatchReward,
            BatchNext,
            BatchDone,
            BatchPriority
    ]:
        pri_step = self.tree.total_p / batch_size
        self.belta = np.min([1, self.belta + self.belta_increment_per_sampling])
        min_p = self.tree.min_p / self.tree.total_p
        min_p = 0.0001 if min_p <= 0.0001 else min_p
        v = [np.random.uniform(i * pri_step, (i + 1) * pri_step) for i in range(batch_size)]
        indices, ISweight = self.tree.get_leaf(v)
        ISweight = [[np.power(i / min_p, self.belta)] for i in ISweight]
        b_state = self.__m_states[indices, :4].to(self.__device).float()
        b_next = self.__m_states[indices, 1:].to(self.__device).float()
        b_action = self.__m_actions[indices].to(self.__device)
        b_reward = self.__m_rewards[indices].to(self.__device).float()
        b_done = self.__m_dones[indices].to(self.__device).float()
        ISweight = torch.Tensor(np.array(ISweight)).to(self.__device).float()
        return indices, b_state, b_action, b_reward, b_next, b_done, ISweight

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti + self.__capacity - 1, p)

    def __len__(self) -> int:
        return self.__size
