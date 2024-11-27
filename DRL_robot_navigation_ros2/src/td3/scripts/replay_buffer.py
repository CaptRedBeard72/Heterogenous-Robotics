"""
Data structure for implementing experience replay
Author: Patrick Emami
"""
import random
from collections import deque

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, buffer_size, alpha=0.6, random_seed=123):
        """
        Introduce priority sampling based on TD-error.
        """
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=buffer_size)  # Limit deque size directly
        self.priorities = deque(maxlen=buffer_size)  # Limit priority size directly
        self.alpha = alpha  # Controls how much prioritization is used
        random.seed(random_seed)
        self.count = 0

    def add(self, s, a, r, t, s2, td_error=None):
        """
        Add experiences to the buffer and assign normalized priority.
        """
        experience = (
            s.detach().cpu() if isinstance(s, torch.Tensor) else s,
            a.detach().cpu() if isinstance(a, torch.Tensor) else a,
            r,
            t,
            s2.detach().cpu() if isinstance(s2, torch.Tensor) else s2
        )

        # Normalize the TD error relative to existing priorities
        if td_error is None:
            td_error = 1.0  # Default priority for new experiences
        else:
            # Normalize TD error
            max_priority = max(self.priorities) if self.priorities else 1.0
            td_error = td_error / max_priority  # Normalize to [0, 1] relative to max priority

        # Ensure priority is always a float
        td_error = float(td_error)

        if self.count < self.buffer_size:
            # Add experience and priority if buffer is not full
            self.buffer.append(experience)
            self.priorities.append(td_error)
            self.count += 1
        else:
            # Replace oldest experience and priority if buffer is full
            self.buffer.popleft()
            self.priorities.popleft()
            self.buffer.append(experience)
            self.priorities.append(td_error)

    def size(self):
        """
        Get the current size of the replay buffer.
        """
        return len(self.buffer)

    def sample_batch(self, batch_size):
        """
        Sample a batch of experiences from the buffer.
        """
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        def to_numpy(data):
            if isinstance(data, torch.Tensor):
                return data.detach().cpu().numpy()
            return np.array(data)

        s_batch = np.array([to_numpy(exp[0]) for exp in batch], dtype=object)
        a_batch = np.array([to_numpy(exp[1]) for exp in batch], dtype=object)
        r_batch = np.array([exp[2] for exp in batch])
        d_batch = np.array([exp[3] for exp in batch])
        s2_batch = np.array([to_numpy(exp[4]) for exp in batch], dtype=object)

        indices = np.arange(len(batch))

        return s_batch, a_batch, r_batch, d_batch, s2_batch, indices

    def update_priorities(self, indices, td_errors):
        epsilon = 1e-6  # Small constant to avoid zero priorities
        for idx, td_error in zip(indices, td_errors):
            if idx >= len(self.priorities):  # Ensure index is within bounds
                continue
            if isinstance(td_error, (torch.Tensor, np.ndarray)):  # Handle tensor/array cases
                td_error = float(td_error.item()) if td_error.size == 1 else float(td_error[0])
            priority = max(epsilon, td_error)
            self.priorities[idx] = priority ** self.alpha

    def clear(self):
        """
        Clear the replay buffer and priorities.
        """
        self.buffer.clear()
        self.priorities.clear()


    # def sample_batch(self, batch_size):
    #     """
    #     Sample a batch using priority sampling.
    #     """
    #     if not self.priorities:
    #         raise ValueError("Priorities are empty. Cannot sample from an empty buffer.")

    #     # Convert priorities to a NumPy array with explicit dtype
    #     scaled_priorities = np.array(self.priorities, dtype=np.float64) ** self.alpha
    #     sampling_probs = scaled_priorities / scaled_priorities.sum()

    #     # Sample indices based on priority probabilities
    #     indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs)

    #     batch = [self.buffer[i] for i in indices]
    #     s_batch = np.array([_[0] for _ in batch])
    #     a_batch = np.array([_[1] for _ in batch])
    #     r_batch = np.array([_[2] for _ in batch]).reshape(-1, 1)
    #     t_batch = np.array([_[3] for _ in batch]).reshape(-1, 1)
    #     s2_batch = np.array([_[4] for _ in batch])

    #     return s_batch, a_batch, r_batch, t_batch, s2_batch, indices