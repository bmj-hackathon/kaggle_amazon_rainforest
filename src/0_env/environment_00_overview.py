# %% [markdown]
# # Decentralized Data Science use case - Understanding the Amazon deforestation from Space challenge

# This notebook demonstrates accessing a dataset from Ocean Protocol and training a deep learning classifier.

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


