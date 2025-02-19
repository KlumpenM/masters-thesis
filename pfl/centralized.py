import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import sys
import torch

# Both Jupyter and `pfl` use async. `nest_asyncio` allows `pfl` to run inside the notebook 
import nest_asyncio
nest_asyncio.apply()

torch.random.manual_seed(1)
np.random.seed(1)

# Always import the `pfl` model first before initializing any `pfl` components to let `pfl` know which Deep Learning framework you will use.
from pfl.model.pytorch import PyTorchModel