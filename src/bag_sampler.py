from abc import ABC, abstractmethod
import numpy as np
from typing import List


class BagSampler(ABC):
    """Abstract base class for bag sampling strategies."""
    
    @abstractmethod
    def sample(self, indices: np.ndarray, *, rng: np.random.RandomState) -> List[np.ndarray]:
        """
        Sample indices for runtime sampling.
        
        Args:
            indices: Array of indices to sample from
            rng: Random number generator
            
        Returns:
            List of sampled index arrays
        """
        pass


class BootstrapBagSampler(BagSampler):
    """Bootstrap sampling strategy for runtime sampling."""
    
    def __init__(self, ratio: float = 0.5, n_bags: int = 10):
        """
        Initialize bootstrap sampler.
        
        Args:
            ratio: Ratio of cells to sample in each bag (0.0-1.0)
            n_bags: Number of bags to create
        """
        self.ratio = ratio
        self.n_bags = n_bags

    def sample(self, indices: np.ndarray, *, rng: np.random.RandomState) -> List[np.ndarray]:
        """
        Sample indices using bootstrap sampling (with replacement).
        
        Args:
            indices: Array of indices to sample from
            rng: Random number generator
            
        Returns:
            List of sampled index arrays
        """
        N = len(indices)
        n_sample = max(1, int(N * self.ratio))
        return [rng.choice(indices, size=n_sample, replace=True) for _ in range(self.n_bags)]