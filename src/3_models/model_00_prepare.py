# %% [markdown]
# ## Section 4: Define model
# %%

model = vgg16.create_model(img_dim=(128, 128, 3))
model.summary()

# %% [markdown]

# ## Fine-tune conv layers
# We will now finetune all layers in the VGG16 model.

# %%

history = History()
callbacks = [history,
             EarlyStopping(monitor='val_loss', patience=3, verbose=1, min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, cooldown=0, min_lr=1e-7, verbose=1),
             ModelCheckpoint(filepath='weights/weights.best.hdf5', verbose=1, save_best_only=True,
                             save_weights_only=True, mode='auto')]

X_train, y_train = preprocessor.X_train, preprocessor.y_train
X_val, y_val = preprocessor.X_val, preprocessor.y_val

batch_size = 64
train_generator = preprocessor.get_train_generator(batch_size)
steps = len(X_train) / batch_size
epochs = 2

model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics = ['accuracy'])