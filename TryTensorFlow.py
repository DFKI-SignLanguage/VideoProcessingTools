import tensorflow as tf

print("Loading CIFAR 100 data ...")
cifar = tf.keras.datasets.cifar100
(x_train, y_train), (x_test, y_test) = cifar.load_data()

print("Building model ...")
model = tf.keras.applications.ResNet50(
    include_top=True,
    weights=None,
    input_shape=(32, 32, 3),
    classes=100,)

print("Compiling ...")
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

print("Fitting ...")
model.fit(x_train, y_train, epochs=5, batch_size=64)

print("All done.")
