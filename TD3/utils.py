from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter


class ReplayBuffer(object):
    def __init__(
        self, state_dim, action_dim, max_size=int(1e6), writer: Optional[SummaryWriter] = None
    ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = writer
        self.writer_iter = 0

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        if self.writer is not None and bool(done):
            last_done = np.argmax(self.not_done[: self.ptr - 1 : -1])
            episode_return = np.mean(self.reward[last_done + 1 : self.ptr - 1])
            self.writer.add_scalar("return", episode_return, self.writer_iter)
            self.writer_iter += 1

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
        )

