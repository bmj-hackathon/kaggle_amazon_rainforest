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


