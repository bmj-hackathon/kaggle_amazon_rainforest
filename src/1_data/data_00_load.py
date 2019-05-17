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
# The data path is relative to current path
from pathlib import Path
PATH_DATA_ROOT = Path.cwd() / "data"
if not PATH_DATA_ROOT.exists():
    PATH_DATA_ROOT = Path.cwd() / "../data"
assert PATH_DATA_ROOT.exists()

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

