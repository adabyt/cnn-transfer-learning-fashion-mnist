import tensorflow as tf
from tensorflow import keras
from keras import layers 
from keras import models 
from keras.applications import MobileNetV2 
import numpy as np

print("\n----- 1. Loading and Initial Data Preparation -----")

# Load the Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"Original x_train shape: {x_train.shape}")   # (60000, 28, 28)
print(f"Original y_train shape: {y_train.shape}")   # (60000,)
print(f"Original x_test shape: {x_test.shape}")     # (10000, 28, 28)
print(f"Original y_test shape: {y_test.shape}")     # (10000,)

# Define class names for better understanding
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("-"*100)

print("\n----- 2. Data Preprocessing for Transfer Learning -----")

# Pre-trained models (like MobileNetV2, VGG16, ResNet50) expect:
# - 3 color channels (RGB)
# - Larger image sizes (e.g., 96x96, 128x128, 224x224)

# For Fashion MNIST (28x28 grayscale), we need to:
# a) Resize the images to a larger dimension (e.g., 96x96)
# b) Convert them from 1 channel (grayscale) to 3 channels (RGB)

# Example preprocessing steps 
def preprocess_for_transfer_learning(images):
    # Ensure images are float32 and normalized (0-1)
    images = tf.cast(images, tf.float32) / 255.0 # Shape: (None, 28, 28)

    # EXPLICITLY ADD CHANNEL DIMENSION (from (None, 28, 28) to (None, 28, 28, 1))
    # This is crucial because tf.image.resize expects 4 dimensions.
    images = tf.expand_dims(images, axis=-1) # Shape: (None, 28, 28, 1)

    # Resize images (now correctly operates on (None, 28, 28, 1) to (None, 96, 96, 1))
    images = tf.image.resize(images, (96, 96), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) # Shape: (None, 96, 96, 1)

    # Convert grayscale (1 channel) to RGB (3 channels) by duplicating the channel
    # This now operates on the correct channel dimension (axis=-1 refers to the 1)
    images = tf.repeat(images, 3, axis=-1) # Shape: (None, 96, 96, 3)

    return images

# Apply this to your Fashion MNIST data
x_train_processed = preprocess_for_transfer_learning(x_train)
x_test_processed = preprocess_for_transfer_learning(x_test)


# --- 2. Load the Pre-trained Base Model ---
# We'll use MobileNetV2 as it's relatively lightweight and good for demonstrations.
# 'weights="imagenet"' ensures we load the weights trained on the ImageNet dataset.
# 'include_top=False' removes the original classification layers of MobileNetV2,
# as we'll replace them with our own for 10 Fashion MNIST classes.
# 'input_shape' must match the processed image size (96, 96, 3).

print("-"*100)

print("\n----- 3. Base Model Definition -----")

base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(96, 96, 3))

print("-"*100)

print("\n----- 4. Freeze Base Model -----")
# This makes all layers in the base_model non-trainable, so their weights
# won't be updated during the initial training phase. This is the "Feature Extraction" step.
base_model.trainable = False

print("-"*100)

print("\n----- 5. Build New Classification Head -----")

# We'll add our own layers on top of the base model's output to classify Fashion MNIST items.

# Create an input layer matching the processed image shape
inputs = keras.Input(shape=(96, 96, 3))

# Pass the inputs through the base model.
# IMPORTANT: set `training=False` for the base model when it's frozen.
# This ensures its BatchNormalization layers run in inference mode (using global stats),
# even if the overall model is in training mode.
# This means the weights won't be adjusted for the frozen, already-trained layers.
x = base_model(inputs, training=False)

# Add a GlobalAveragePooling2D layer to flatten the feature maps.
# This averages the spatial dimensions (height, width) for each channel,
# resulting in an output shape of (None, num_channels_from_last_conv_layer).
x = layers.GlobalAveragePooling2D()(x)

# Add classification Dense layers, similar to the previous CNNs
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x) 
outputs = layers.Dense(10, activation='softmax')(x) # 10 classes for Fashion MNIST

# Create the full model
model_transfer = keras.Model(inputs, outputs)

print("-"*100)

print("\n----- 5. Compile the Model -----")

# Use a suitable optimiser and loss function.
model_transfer.compile(optimizer='adam',
                       loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                       metrics=['accuracy'])

model_transfer.summary() 

# Model: "functional"
# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
# ┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
# ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
# │ input_layer_1 (InputLayer)           │ (None, 96, 96, 3)           │               0 │  # An InputLayer defines the shape of the data entering the model. It doesn't perform any computation or have any learnable parameters (weights or biases), so its parameter count is always zero.
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ mobilenetv2_1.00_96 (Functional)     │ (None, 3, 3, 1280)          │       2,257,984 │  # The sum of all learnable parameters (weights and biases) within all the convolutional layers, depthwise separable convolutional layers, Batch Normalization layers, and other components that make up the MobileNetV2 architecture up to the point where it was cut off.
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ global_average_pooling2d             │ (None, 1280)                │               0 │  # This layer calculates the average value for each feature map (channel) across its spatial dimensions (height and width). It's a purely mathematical operation (an average) and does not have any learnable weights or biases. Therefore, its parameter count is zero.
# │ (GlobalAveragePooling2D)             │                             │                 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 128)                 │         163,968 │  # Total Parameters = (Input Features * Number of Neurons) + Number of Neurons (biases); Total Parameters = (1280 * 128) + 128 = 163,968.
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dropout (Dropout)                    │ (None, 128)                 │               0 │  # A Dropout layer randomly sets a fraction of its input units to zero during training to prevent overfitting. It's a regularisation technique that doesn't involve any learnable parameters, so its parameter count is zero.
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │  # Total Parameters = (Input Features * Number of Neurons) + Number of Neurons (biases); Total Parameters = (128 * 10) + 10 = 1290
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 2,423,242 (9.24 MB)
#  Trainable params: 165,258 (645.54 KB)
#  Non-trainable params: 2,257,984 (8.61 MB)


print("-"*100)

print("\n----- 6. Model Training -----")

history_fine_tune = model_transfer.fit(
    x_train_processed, 
    y_train, 
    epochs=10,
    batch_size=32,
    validation_split=0.1
    )

print("-"*100)

# ----- 6. Model Training -----
# Epoch 1/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 59s 34ms/step - accuracy: 0.7815 - loss: 0.6457 - val_accuracy: 0.8725 - val_loss: 0.3318
# Epoch 2/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 59s 35ms/step - accuracy: 0.8595 - loss: 0.3803 - val_accuracy: 0.8847 - val_loss: 0.3096
# Epoch 3/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 60s 36ms/step - accuracy: 0.8734 - loss: 0.3462 - val_accuracy: 0.8895 - val_loss: 0.2888
# Epoch 4/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 60s 36ms/step - accuracy: 0.8773 - loss: 0.3303 - val_accuracy: 0.8858 - val_loss: 0.2968
# Epoch 5/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 62s 37ms/step - accuracy: 0.8837 - loss: 0.3194 - val_accuracy: 0.8995 - val_loss: 0.2797
# Epoch 6/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 60s 36ms/step - accuracy: 0.8897 - loss: 0.2943 - val_accuracy: 0.9000 - val_loss: 0.2771
# Epoch 7/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 61s 36ms/step - accuracy: 0.8936 - loss: 0.2911 - val_accuracy: 0.8975 - val_loss: 0.2675
# Epoch 8/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 62s 36ms/step - accuracy: 0.8957 - loss: 0.2805 - val_accuracy: 0.8955 - val_loss: 0.2757
# Epoch 9/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 63s 38ms/step - accuracy: 0.8998 - loss: 0.2734 - val_accuracy: 0.8992 - val_loss: 0.2756
# Epoch 10/10
# 1688/1688 ━━━━━━━━━━━━━━━━━━━━ 61s 36ms/step - accuracy: 0.8996 - loss: 0.2673 - val_accuracy: 0.8968 - val_loss: 0.2782

"""
Observations:
- While accuracy increases and loss decreases for the training data per epoch (up to epoch 9), there's a marked decrease in the model's efficiency after epoch 6
- This suggests that the model is being trained too well on the training set (i.e. overfitting), and an appropriate number of epochs would be 6
"""

print("\n----- 7. Model Evaluation -----")

# Evaluate the CNN model on the test data
test_loss_cnn, test_accuracy_cnn = model_transfer.evaluate(x_test_processed, y_test, verbose=1) # Must use x_test_processed which has the desired input shape (96, 96, 3)

print(f"\nCNN Test Loss: {test_loss_cnn:.4f}")          # 0.2904    
print(f"CNN Test Accuracy: {test_accuracy_cnn:.4f}")    # 0.8930

print("-"*100)

print("\n----- 8. Making Predictions -----")

# Make predictions on a few test samples using the CNN model
predictions_transfer = model_transfer.predict(x_test_processed[:5])
predicted_classes_transfer = np.argmax(predictions_transfer, axis=1)

print(f"\nCNN Predictions for the first 5 test samples (classes): {predicted_classes_transfer}")    # [9 2 1 1 6]
print(f"Actual classes for the first 5 samples: {y_test[:5]}")                                      # [9 2 1 1 6]

print("\nCNN Comparison of predicted vs actual:")
for i in range(5):
    print(f"Sample {i+1}: Predicted: {class_names[predicted_classes_transfer[i]]} (Index: {predicted_classes_transfer[i]}), Actual: {class_names[y_test[i]]} (Index: {y_test[i]})")

# CNN Comparison of predicted vs actual:
# Sample 1: Predicted: Ankle boot (Index: 9), Actual: Ankle boot (Index: 9)
# Sample 2: Predicted: Pullover (Index: 2), Actual: Pullover (Index: 2)
# Sample 3: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 4: Predicted: Trouser (Index: 1), Actual: Trouser (Index: 1)
# Sample 5: Predicted: Shirt (Index: 6), Actual: Shirt (Index: 6)

"""
Observations:

Model (best performing)     | Test Accuracy     | Test Loss
----------------------------|-------------------|-----------     
Transfer Learning CNN       | 0.8930            | 0.2904
Standard CNN                | 0.9101            | 0.2874
Regularised CNN             | 0.9102            | 0.2941
Augmented CNN               | 0.8976            | 0.2889

Overall, the Transfer Learning CNN performed worse than the other CNN models
- This may reflect issues with resizing the Fashion MNIST images (from (28, 28, 1) to (96, 96, 3)) that may introduce unnecessary distortions to the images that prevent the model from accurately identifying them
- This may also be due to the use of nearest neighbours for resizing, as nearest neighbour finds the single closest pixel in the original 28x28 grid and simply copies its value
    - Although this is computationally faster, it may have lead to a "blocky" or "pixelated" appearance, which the model cannot clearly identify
    - An alternative approach, bilinear, which looks at the four closest pixels in the original image and then calculates a new pixel value by taking a weighted average, may be better (although more computationally intensive)

"""

"""
Note: To unfreeze layers to potentially improve the model

1. Unfreeze the entire base_model:
```
base_model.trainable = True
```

2. Selectively freeze earlier layers
```
# Iterate through the base model's layers
for layer in base_model.layers:
    # For example, freeze layers up to a certain point (e.g., based on layer name or index)
    if not isinstance(layer, layers.BatchNormalization):    # Keep BN layers trainable by default in Keras 3.x if base_model.trainable=True
        layer.trainable = True                              # Ensure all are trainable first

    # Example: freeze layers below a certain depth (e.g., for MobileNetV2)
    # MobileNetV2 has many layers - look at base_model.summary() and identify blocks. Let's say we want to unfreeze layers from 'block_13_expand' onwards.
    if layer.name.startswith('block_') and int(layer.name.split('_')[1]) < 13:
         layer.trainable = False
    else:
         layer.trainable = True # Unfreeze the later blocks
```

3. IMPORTANT: Re-compile the entire mode
```
model_transfer.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),                # Use a VERY small learning rate (as pre-trained weights are already very good)
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

4. Continue the training
```
history_fine_tune_full = model_transfer.fit(
    x_train_processed,
    y_train,
    epochs=6                # Based on the original model before unfreezing
    batch_size=32,
    validation_split=0.1
)
```
"""
