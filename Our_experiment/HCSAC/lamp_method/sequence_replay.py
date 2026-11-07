"""Episode-owned raw snapshots, sampled as contiguous sequences."""
import random
from typing import Any
import torch


class SequenceReplay:
    def __init__(self) -> None:
        self.episodes: list[list[dict[str, Any]]] = []

    def add_episode(self, transitions: list[dict[str, Any]]) -> None:
        if not transitions or any(t.get('terminal', False) for t in transitions[:-1]):
            raise ValueError('Episode must be nonempty and cannot cross a terminal')
        if not transitions[-1].get('terminal', False):
            raise ValueError('Episode must end in a terminal')
        self.episodes.append(transitions)

    def sample(self, unroll_length: int, batch_size: int) -> list[tuple[list, list, torch.Tensor]]:
        if not self.episodes or unroll_length < 1 or batch_size < 1:
            raise ValueError('Replay and lengths must be nonempty')
        result = []
        for _ in range(batch_size):
            episode = random.choice(self.episodes)
            start = random.randrange(len(episode))
            segment=episode[start:start+unroll_length]
            # Ragged, unpadded sequences still carry an explicit validity mask.
            sequence_mask=torch.ones(len(segment),dtype=torch.bool)
            result.append((episode[:start], segment, sequence_mask))
        return result

    def state_dict(self) -> dict:
        return {'episodes': self.episodes}

    def load_state_dict(self, state: dict) -> None:
        self.episodes = state['episodes']
