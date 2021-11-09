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
        self.epsilon = 0.01
        self.alpha = 0.6
        self.abs_err_upper = 1.0
        # self.__pos = 0
        #
        # self.__m_states = torch.zeros(
        #     (capacity, channels, 84, 84), dtype=torch.uint8)
        # self.__m_actions = torch.zeros((capacity, 1), dtype=torch.long)
        # self.__m_rewards = torch.zeros((capacity, 1), dtype=torch.int8)
        # self.__m_dones = torch.zeros((capacity, 1), dtype=torch.bool)

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
        self.tree.add(max_p, (folded_state, action, reward, done))
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
        self.belta = np.min([1, self.belta + self.belta_increment_per_sampling])
        min_p = self.tree.min_p / self.tree.total_p
        min_p = 0.0001 if min_p <= 0.0001 else min_p
        inde = []
        b_state = []
        b_next = []
        b_action = []
        b_reward = []
        b_done = []
        ISweight = []
        for i in range(batch_size):
            st, ed = pri_step * i, pri_step * (i + 1)
            v = np.random.uniform(st, ed)
            try:
                indec, p, (m_states, m_actions, m_rewards, m_dones) = self.tree.get_leaf(v)
            except:
                print(f"{st} {ed} {v} {self.tree.get_leaf(v)} {self.tree.data_pointer}")

            inde.append(indec)
            b_state.append(m_states[0][:4])
            b_next.append(m_states[0][1:])
            b_action.append([m_actions])
            b_reward.append([m_rewards])
            b_done.append([m_dones])
            ISweight.append([np.power(p / min_p, self.belta)])
        b_state = torch.Tensor(np.array([item.detach().numpy() for item in b_state])).to(self.__device)
        b_next = torch.Tensor(np.array([item.detach().numpy()for item in b_next])).to(self.__device)
        b_action = torch.Tensor(np.array(b_action)).to(self.__device).long()
        b_reward = torch.Tensor(np.array(b_reward)).to(self.__device).float()
        b_done = torch.Tensor(np.array(b_done)).to(self.__device).float()
        ISweight = torch.Tensor(np.array(ISweight)).to(self.__device).float()
        return inde, b_state, b_action, b_reward, b_next, b_done, ISweight

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def __len__(self) -> int:
        return self.__size
