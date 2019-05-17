# %% [markdown]
# ## Section 4: Define model
# %%

history = model.fit_generator(train_generator, steps, epochs=5, verbose=1,
                    validation_data=(X_val, y_val), callbacks=callbacks)

# %%
# ## Visualize Loss Curve

# %%

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()