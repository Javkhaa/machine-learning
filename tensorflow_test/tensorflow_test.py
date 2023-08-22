"""
You might get warning saying 

2023-08-02 22:43:00.517247: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:303] 
Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2023-08-02 22:43:00.517278: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:269] 
Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice 
(device: 0, name: METAL, pci bus id: <undefined>)

"""

import tensorflow as tf

cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.fit(x_train, y_train, epochs=5, batch_size=64)

