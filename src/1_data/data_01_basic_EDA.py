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
#cols = 3
#rows = 3
_, axs = plt.subplots(5, 4, sharex='col', sharey='row', figsize=(15, 20))
axs = axs.ravel()

for i, (image_name, label) in enumerate(zip(images_title, labels_set)):
    img = mpimg.imread(train_jpeg_dir / image_name)
    axs[i].imshow(img)
    axs[i].set_title('{} - {}'.format(image_name, label))
plt.show()