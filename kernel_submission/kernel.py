# %% [markdown]

# <p><img src="https://oceanprotocol.com/static/media/banner-ocean-03@2x.b7272597.png" alt="drawing" width="800" align="center"/>

# %% [markdown]
# <h1><center>Ocean Protocol - Manta Ray project</center></h1>
# <h3><center>Decentralized Data Science and Engineering, powered by Ocean Protocol</center></h3>
# <p>Version 0.5.1 - beta</p>
# <p>Package compatibility: squid-py v0.5.11, keeper-contracts 0.9.0, utilities 0.2.1,
# <p>Component compatibility: Brizo v0.3.2, Aquarius v0.2.1, Nile testnet smart contracts 0.8.6</p>
# <p><a href="https://github.com/oceanprotocol/mantaray">mantaray on Github</a></p>
# <p>


# %% [markdown]
# # Decentralized Data Science use case - Understanding the Amazon deforestation from Space challenge

# This notebook demonstrates accessing a dataset from Ocean Protocol and training a deep learning classifier.

# %% [markdown]
# <p><img src="https://www.borgenmagazine.com/wp-content/uploads/2013/11/Deforestation.jpg" alt="drawing" width="1000" align="center"/>

# %% [markdown]

# Attribution: Source for this notebook was prepared by Tuatini Godard for the Kaggle Competition "Planet: Understanding the Amazon from Space"
#
# See the [source kernel](https://www.kaggle.com/ekami66/0-92837-on-private-lb-solution-with-keras)
#
# And the [source GitHub repository](https://github.com/EKami/planet-amazon-deforestation)
#
# Modifications and refactoring made by M.Jones 17 May 2019
# - Data source paths updated
# - New plotting function
# - Various formatting and comments
#
# Below, the kernel;

# %% [markdown]
#
# Special thanks to the kernel contributors of this challenge (especially @anokas and @Kaggoo) who helped me find a starting point for this notebook.
#
# The whole code including the `data_helper.py` and `keras_helper.py` files are available on github [here](https://github.com/EKami/planet-amazon-deforestation) and the notebook can be found on the same github [here](https://github.com/EKami/planet-amazon-deforestation/blob/master/notebooks/amazon_forest_notebook.ipynb)
#
# **If you found this notebook useful some upvotes would be greatly appreciated! :) **


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
# %% [markdown]
# ## Section 1: Access and download the data from Ocean Protocol
# Data asset Decentralized Identifier (did): `did:op:3fdcc402b9994d88828e82f9be16e40eaf8eed10036c48ae9a826415e3ca46ce`
#
# Commons market link: [Amazon rainforest satellite imagery](https://commons.oceanprotocol.com/asset/did:op:3fdcc402b9994d88828e82f9be16e40eaf8eed10036c48ae9a826415e3ca46ce)
#
# Optionally, download the data directly in your notebook environment:

# %% [markdown]
# ```python
# import logging
# import os
# from squid_py import Metadata, Ocean
# import squid_py
# import mantaray_utilities as manta_utils
#
# # Setup logging
# from mantaray_utilities.user import get_account_from_config
# from mantaray_utilities.blockchain import subscribe_event
# manta_utils.logging.logger.setLevel('INFO')
# import mantaray_utilities as manta_utils
# from squid_py import Config
# from squid_py.keeper import Keeper
# from pathlib import Path
# import datetime
# import web3
#
# ```


# %%
data_root_folder = Path.cwd() / 'data'
assert data_root_folder.exists()
train_jpeg_dir = data_root_folder / 'train-jpg'
assert train_jpeg_dir.exists()
test_jpeg_dir = data_root_folder / 'test-jpg'
assert test_jpeg_dir.exists()
test_jpeg_additional_dir = data_root_folder / 'test-jpg-additional'
assert test_jpeg_additional_dir.exists()
train_csv_file = data_root_folder / 'train_v2.csv'
assert train_csv_file.exists()

# %% [markdown]
# ## Section 2: Basic data exploration and visualization
# %% [markdown]

# ## Inspect image labels
# Visualize what the training set looks like

# %%

# train_jpeg_dir, test_jpeg_dir, test_jpeg_additional, train_csv_file = data_helper.get_jpeg_data_files_paths()
labels_df = pd.read_csv(train_csv_file)
labels_df.head()

# %% [markdown]

# Each image can be tagged with multiple tags, lets list all uniques tags

# %%

# Print all unique tags
from itertools import chain
labels_list = list(chain.from_iterable([tags.split(" ") for tags in labels_df['tags'].values]))
labels_set = set(labels_list)
print("{} labels: {}".format(len(labels_set), labels_set))

# %% [markdown]
# ### Repartition of each labels

# %%

# Histogram of label instances
labels_s = pd.Series(labels_list).value_counts() # To sort them by count
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=labels_s, y=labels_s.index, orient='h')
plt.show()
# %% [markdown]

# ## Images
# Visualize some chip images to know what we are dealing with.
# Lets vizualise 1 chip for the 17 images to get a sense of their differences.

# %%

images_title = [labels_df[labels_df['tags'].str.contains(label)].iloc[i]['image_name'] + '.jpg'
                for i, label in enumerate(labels_set)]

plt.rc('axes', grid=False)
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir / image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))
plt.show()