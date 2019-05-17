# %% [markdown]

# ## Load Best Weights

# %%

model.load_weights("weights/weights.best.hdf5")
print("Weights loaded")

# %% [markdown]

# ## Check Fbeta Score

# %%

fbeta_score = vgg16.fbeta(model, X_val, y_val)

fbeta_score

# %% [markdown]

# ## Make predictions

# %%

predictions, x_test_filename = vgg16.predict(model, preprocessor, batch_size=128)
print("Predictions shape: {}\nFiles name shape: {}\n1st predictions ({}) entry:\n{}".format(predictions.shape,
                                                                              x_test_filename.shape,
                                                                              x_test_filename[0], predictions[0]))

# %% [markdown]

# Before mapping our predictions to their appropriate labels we need to figure out what threshold to take for each class

# %%

thresholds = [0.2] * len(labels_set)

# %% [markdown]

# Now lets map our predictions to their tags by using the thresholds

# %%

predicted_labels = vgg16.map_predictions(preprocessor, predictions, thresholds)

# %% [markdown]

# Finally lets assemble and visualize our predictions for the test dataset

# %%

tags_list = [None] * len(predicted_labels)
for i, tags in enumerate(predicted_labels):
    tags_list[i] = ' '.join(map(str, tags))

final_data = [[filename.split(".")[0], tags] for filename, tags in zip(x_test_filename, tags_list)]

# %%

final_df = pd.DataFrame(final_data, columns=['image_name', 'tags'])
print("Predictions rows:", final_df.size)
final_df.head()

# %%

tags_s = pd.Series(list(chain.from_iterable(predicted_labels))).value_counts()
fig, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x=tags_s, y=tags_s.index, orient='h');

# %% [markdown]

# If there is a lot of `primary` and `clear` tags, this final dataset may be legit...
