"""Fixed random streams and deterministic numerical kernels."""

import os
import random
import numpy as np
import torch
import cv2


def configure(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    cv2.setRNGSeed(int(seed) % (2**31 - 1))
    cv2.setNumThreads(1)


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(int(seed) % (2**31 - 1))
    cv2.setNumThreads(1)
