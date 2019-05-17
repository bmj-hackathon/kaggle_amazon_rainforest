# %% [markdown]
# ## Section 0: Prepare environment and libraries

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

# Move to the project directory if inside a subdirectory (JupyterLab)
import os
cwd = Path.cwd().parts
if cwd[-1] == 'kernel_submission':
    cwd = cwd[0:-1]
    cwd = Path(*cwd)
    os.chdir(cwd)
    logging.info("Changed working directory to {}".format(cwd))

# %%
import warnings
import gc
warnings.simplefilter("ignore", category=DeprecationWarning)

# %%
# Scientific stack
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import sklearn as sk
import seaborn as sns
import h5py

logging.info("{:>10}=={} as {}".format('numpy', np.__version__, 'np'))
logging.info("{:>10}=={} as {}".format('pandas', pd.__version__, 'pd'))
logging.info("{:>10}=={} as {}".format('sklearn', sk.__version__, 'sk'))
logging.info("{:>10}=={} as {}".format('seaborn', sns.__version__, 'sns'))


# %%
# Deep Learning
assert "LD_LIBRARY_PATH" in os.environ
# Deep learning stack
import tensorflow as tf
import tensorflow.keras as ks
logging.info("{:>10}=={} as {}".format('tensorflow', tf.__version__, 'tf'))
logging.info("{:>10}=={} as {}".format('keras', ks.__version__, 'ks'))

# %%

import os
# import gc
# import bcolz
# import numpy as np
import pandas as pd

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

# import data_helper
from src.utils import vgg16
try:
    from src.utils import vgg16
    from src.utils import data_helper
    from src.utils.data_helper import AmazonPreprocessor
except:
    path_utils = Path.cwd() / '../src'
    assert path_utils.exists()
    sys.path.insert(0, str(path_utils.resolve()))
    from utils import vgg16
    from utils import data_helper
    from utils.data_helper import AmazonPreprocessor
# from src.utils import vgg16
# from src.utils import data_helper
# from src.utils.data_helper import AmazonPreprocessor
# from kaggle_data.downloader import KaggleDataDownloader


# %%
def mm2inch(value):
    return value/25.4
PAPER = {
    "A3_LANDSCAPE" : (mm2inch(420),mm2inch(297)),
    "A4_LANDSCAPE" : (mm2inch(297),mm2inch(210)),
    "A5_LANDSCAPE" : (mm2inch(210),mm2inch(148)),
}
