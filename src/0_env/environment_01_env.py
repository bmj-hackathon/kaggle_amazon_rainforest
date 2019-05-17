#%% ===========================================================================
# Logging
# =============================================================================
import sys
import logging

logger = logging.getLogger()
logger.handlers = []

# Set level
logger.setLevel(logging.INFO)

# Create formatter
FORMAT = "%(levelno)-2s %(asctime)s : %(message)s"
DATE_FMT = "%Y-%m-%d %H:%M:%S"
formatter = logging.Formatter(FORMAT, DATE_FMT)

# Create handler and assign
handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(formatter)
logger.handlers = [handler]
logging.info("Logging started")

# %% Paths
from pathlib import Path
PATH_DATA_ROOT = Path.cwd() / "data"
assert PATH_DATA_ROOT.exists()

# %%
import warnings
import gc
warnings.simplefilter("ignore", category=DeprecationWarning)

from pathlib import Path

# %% Standard imports
import os
from pathlib import Path
import functools
import xml.etree.ElementTree as etree
import xmltodict
# %%
# Scientific stack
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import sklearn as sk

import h5py

logging.info("{:>10}=={} as {}".format('numpy', np.__version__, 'np'))
logging.info("{:>10}=={} as {}".format('pandas', pd.__version__, 'pd'))
logging.info("{:>10}=={} as {}".format('sklearn', sk.__version__, 'sk'))
logging.info("{:>10}=={} as {}".format('matplotlib', mpl.__version__, 'mpl'))

# %%
assert "LD_LIBRARY_PATH" in os.environ
assert "/usr/local/cuda-9.0/bin" in [p for p in os.environ['PATH'].split(':')]
# Deep learning stack
import tensorflow as tf
import tensorflow.keras as ks
logging.info("{:>10}=={} as {}".format('tensorflow', tf.__version__, 'tf'))
logging.info("{:>10}=={} as {}".format('keras', ks.__version__, 'ks'))

# %%

# import sys
from pathlib import Path
# PATH_UTILS_MODULE = Path.cwd() / 'src/utils'
# PATH_UTILS_MODULE = Path.cwd() / 'src'
# PATH_UTILS_MODULE.exists()
# sys.path.append(PATH_UTILS_MODULE)
# # sys.path.append('../tests')
# for p in sys.path:
#     print(p)
# %% [markdown]

# ## Import required modules

# %%

import os
# import gc
# import bcolz
# import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.python.keras.optimizers import Adam
# from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History
# import vgg16
# from utils import vgg16
# from . import utils
# from .. import utils
# import .utils
# from  .utils import vgg16
# from utils import vgg16
# import vgg16
# import data_helper

from src.utils import vgg16
from src.utils import data_helper
from src.utils.data_helper import AmazonPreprocessor
# from kaggle_data.downloader import KaggleDataDownloader


# %%
def mm2inch(value):
    return value/25.4
PAPER = {
    "A3_LANDSCAPE" : (mm2inch(420),mm2inch(297)),
    "A4_LANDSCAPE" : (mm2inch(297),mm2inch(210)),
    "A5_LANDSCAPE" : (mm2inch(210),mm2inch(148)),
}
