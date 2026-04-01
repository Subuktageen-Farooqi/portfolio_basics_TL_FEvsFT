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

# Create a validation split from the training set.
# Keep test split untouched for final evaluation only.
validation_split = 0.1
num_train = train_images.shape[0]
split_idx = int(num_train * (1 - validation_split))
rng = np.random.default_rng(42)
indices = rng.permutation(num_train)

train_idx = indices[:split_idx]
val_idx = indices[split_idx:]

val_images = train_images[val_idx]
val_labels = train_labels[val_idx]
train_images = train_images[train_idx]
train_labels = train_labels[train_idx]

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
