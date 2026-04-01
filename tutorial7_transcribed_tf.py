# Tutorial 7: Transfer Learning (TensorFlow / Keras)

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.utils import to_categorical

# Step 1: Load CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values
train_images = train_images / 255.0
test_images = test_images / 255.0


def train_val_split(images, labels, validation_split=0.1, seed=42):
    """Create a validation split from training data while preserving a true test holdout."""
    num_samples = images.shape[0]
    split_idx = int(num_samples * (1 - validation_split))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    train_x = images[train_idx]
    train_y = labels[train_idx]
    val_x = images[val_idx]
    val_y = labels[val_idx]
    return train_x, train_y, val_x, val_y


# Create validation data only from the training split.
# Keep test split untouched for final evaluation only.
train_images, train_labels, val_images, val_labels = train_val_split(
    train_images,
    train_labels,
    validation_split=0.1,
    seed=42,
)

# Convert labels to categorical
train_labels = to_categorical(train_labels, 10)
val_labels = to_categorical(val_labels, 10)
test_labels = to_categorical(test_labels, 10)


# Step 2: Feature Extraction using VGG16

# Load VGG16 pretrained model (without top layer)
vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze convolutional base
vgg_base.trainable = False

# Build model
vgg_model = models.Sequential([
    vgg_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
vgg_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

# Train
vgg_model.fit(train_images, train_labels,
              epochs=5,
              validation_data=(val_images, val_labels))

# Evaluate
vgg_loss, vgg_acc = vgg_model.evaluate(test_images, test_labels)
print("VGG16 Feature Extraction Accuracy:", vgg_acc)


# Step 3: Fine-Tuning using ResNet50

# Load ResNet50 pretrained model (without top layer)
resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze all layers initially
for layer in resnet_base.layers:
    layer.trainable = False

# Build model
resnet_model = models.Sequential([
    resnet_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
resnet_model.compile(optimizer='adam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Train initial classifier
resnet_model.fit(train_images, train_labels,
                 epochs=5,
                 validation_data=(val_images, val_labels))


# Step 4: Fine-tuning (unfreeze some layers)

# Unfreeze last few layers
for layer in resnet_base.layers[-10:]:
    layer.trainable = True

# Recompile with lower learning rate
resnet_model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])

# Continue training
resnet_model.fit(train_images, train_labels,
                 epochs=5,
                 validation_data=(val_images, val_labels))

# Evaluate
resnet_loss, resnet_acc = resnet_model.evaluate(test_images, test_labels)
print("ResNet50 Fine-Tuning Accuracy:", resnet_acc)


# Step 5: Comparison

print("\nFinal Comparison:")
print(f"VGG16 Feature Extraction Accuracy: {vgg_acc:.4f}")
print(f"ResNet50 Fine-Tuning Accuracy: {resnet_acc:.4f}")
