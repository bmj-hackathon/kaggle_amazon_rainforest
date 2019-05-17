# %% [markdown]
# ## Section 1: Access and download the data from Ocean Protocol
# Data asset Decentralized Identifier (did): `did:op:3fdcc402b9994d88828e82f9be16e40eaf8eed10036c48ae9a826415e3ca46ce`
#
# Commons market link: [Amazon rainforest satellite imagery](https://commons.oceanprotocol.com/asset/did:op:3fdcc402b9994d88828e82f9be16e40eaf8eed10036c48ae9a826415e3ca46ce)
#
# Optionally, download the data directly in your notebook environment:

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

