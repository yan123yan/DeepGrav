import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ============== 1. load data and create label ==============
background = np.load('background.npz')['data']
stds = np.std(background, axis=-1)[:, :, np.newaxis]
background = background / stds
background = np.swapaxes(background, 1, 2)
backgroundlabel = np.zeros(100000)

bbh = np.load('bbh_for_challenge.npy')
stds = np.std(bbh, axis=-1)[:, :, np.newaxis]
bbh = bbh / stds
bbh = np.swapaxes(bbh, 1, 2)
bbhlabel = np.ones(100000)

sglf = np.load('sglf_for_challenge.npy')
stds = np.std(sglf, axis=-1)[:, :, np.newaxis]
sglf = sglf / stds
sglf = np.swapaxes(sglf, 1, 2)
sglflabel = np.ones(100000)

# ============== 2. merge label and data ==============
X = np.concatenate([background, bbh, sglf], axis=0)  # shape: (30000, 200, 2)
y = np.concatenate([backgroundlabel, bbhlabel, sglflabel], axis=0)  # shape: (30000,)

# ============== 3. split training set and validation set  ==============
train, val, trainlabel, vallabel = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Shape of training set: ", train.shape)
print("Shape of training set's label: ", trainlabel.shape)
print("Shape of validation set: ", val.shape)
print("Shape of validation set's label: ", vallabel.shape)


# ============== 4. Build model ==============

# Build ResNet-like network
def build_model(input_shape, n_classes):
    input_layer = layers.Input(shape=input_shape)

    # first conv
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # first residual block
    residual = x
    x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(64, 3, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #make sure residual and x have the same shape, use 1x1 conv to adjust the channel number
    residual = layers.Conv1D(64, 1, activation=None, padding='same')(residual)
    x = layers.add([x, residual])  # add residual connection
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # second residual block
    residual = x
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(128, 3, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)

    residual = x
    x = layers.Conv1D(256, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(256, 3, activation=None, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #same as above, adjust channel number to make sure residual and x have the same shape
    residual = layers.Conv1D(256, 1, activation=None, padding='same')(residual)
    x = layers.add([x, residual])  # add residual connection
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # pooling and flatten
    x = layers.GlobalAveragePooling1D()(x)

    # fully connected layer
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    output_layer = layers.Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    return model


# ================== 5. Configure optimizer and callbacks ==================
optimizer = tf.keras.optimizers.Nadam(
    lr=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    schedule_decay=0.005
)

save_dir = 'modelyan2/'
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
checkpointer = ModelCheckpoint(
    os.path.join(save_dir, 'fitmodel_{epoch:03d}.hdf5'),
    verbose=1,
    save_weights_only=False,
    period=1
)

# ============== 6. Compile model ==============
model = build_model(input_shape=(200, 2), n_classes=2)
model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',  #Sparse Categorical Crossentropy
    metrics=['accuracy']
)

# ============== 7. Train model ==============
history = model.fit(
    train,
    trainlabel,
    epochs=100,
    batch_size=512,
    validation_data=(val, vallabel),
    callbacks=[reduce_lr, early_stopping, checkpointer],
    shuffle=True
)

# ============== 8. visualize training process ==============
# visualize training process
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Train and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''
# ============== 9. Model Evaluation ==============
# Evaluate the model on the validation set
loss, accuracy = model.evaluate(val, vallabel)
print(f'Loss of Validation Set: {loss}')
print(f'Acc of Validation Set: {accuracy}')

# ============== 10. Model Prediction ==============
# predict on validation set and calculate ROC curve
y_pred = model.predict(val)
fpr, tpr, thresholds = roc_curve(vallabel, y_pred)
roc_auc = auc(fpr, tpr)

# Visualize ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
'''
