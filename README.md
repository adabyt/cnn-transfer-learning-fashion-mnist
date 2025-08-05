# CNN Transfer Learning on Fashion MNIST

This project applies **transfer learning** using a pre-trained [MobileNetV2](https://keras.io/api/applications/mobilenet/) model on the Fashion MNIST dataset. The goal is to compare its performance against custom CNN architectures trained from scratch.

---

## What is Transfer Learning?

**Transfer learning** leverages pre-trained deep learning models (often trained on large-scale datasets like ImageNet) to solve new but related tasks. Instead of training a model from scratch, we:

1. Use the feature extraction layers of a pre-trained model.
2. Freeze those layers (initially).
3. Add a custom classifier head for the new task (Fashion MNIST in this case).
4. Optionally unfreeze and fine-tune some layers later.

This is especially powerful when working with small or specialised datasets.

---

## Related Projects (Custom CNN Models)

| Model                    | GitHub Link                                                                 |
|--------------------------|------------------------------------------------------------------------------|
| Standard CNN             | [tensorflow-cnn-fashion-mnist](https://github.com/adabyt/tensorflow-cnn-fashion-mnist) |
| Regularised CNN          | [cnn-regularisation-fashion-mnist](https://github.com/adabyt/cnn-regularisation-fashion-mnist) |
| Augmented CNN            | [cnn-augmentation-fashion-mnist](https://github.com/adabyt/cnn-augmentation-fashion-mnist) |

---

## Model Performance Comparison

| Model (best performing)     | Test Accuracy | Test Loss |
|----------------------------|---------------|-----------|
| **Transfer Learning CNN**   | 0.8930        | 0.2904    |
| Standard CNN                | 0.9101        | 0.2874    |
| Regularised CNN             | 0.9102        | 0.2941    |
| Augmented CNN               | 0.8976        | 0.2889    |

**Observation**: The Transfer Learning CNN performed *worse* than the custom-built CNNs trained from scratch.

---

## Training Observations

- While accuracy improves and loss decreases during training (up to epoch 9), model efficiency on validation data decreases after epoch 6.
- This suggests **overfitting**: the model starts to memorise the training data rather than generalise.
- An optimal number of epochs appears to be **6**.

---

## Why Did Transfer Learning Underperform?

Several factors might have contributed:

- **Image Resizing Artifacts**:
  - Fashion MNIST images were resized from `(28, 28, 1)` to `(96, 96, 3)` to meet MobileNetV2's input requirements.
  - This resizing may introduce distortions that confuse the pre-trained model.
  
- **Resizing Method**:
  - Used `nearest neighbor` interpolation (fast, but can lead to "blocky" pixelation).
  - Switching to **bilinear** interpolation may preserve visual features better, improving performance.

---

## Future Directions

To potentially improve results:

1. **Unfreeze and Fine-Tune Deeper Layers**:

```python
base_model.trainable = True

for layer in base_model.layers:
    if layer.name.startswith('block_') and int(layer.name.split('_')[1]) < 13:
        layer.trainable = False
    else:
        layer.trainable = True
```

2.	Re-compile with a lower learning rate:

```
model_transfer.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)
```

3. Continue training for a few additional epochs.

---

## Dataset

[Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)

Contains 70,000 grayscale images across 10 fashion categories.

---

## License

MIT License
