import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf
from powerdl.highlevel import profile_tf

print("TF version:", tf.__version__)
print("TF GPUs:", tf.config.list_physical_devices("GPU"))

# -------------------------
# Data
# -------------------------
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") / 255.0)[..., None]   # (N, 28, 28, 1)

BS = 512
x_inf = x_train[:60000]

ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
ds = ds.shuffle(10000).batch(BS).prefetch(tf.data.AUTOTUNE)

# -------------------------
# Model
# -------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Input((28, 28, 1)),
    tf.keras.layers.Conv2D(16, 3, activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -------------------------
# PowerDL (yalın kullanım)
# -------------------------
with profile_tf(out_dir=None, device_index=0, interval_s=0.02, verbose=1) as prof:
    prof.fit(model, ds, epochs=10, verbose=2)
    prof.infer_keras(model, x_inf, batch_size=BS)

rep = prof.report()

# Produce a rich set of figures (in-memory; no CSV needed).
rep.plot(all=True, out_dir="results/tf_mnist_figs", show=False, smooth=5, shade_phases=True)

# Optional: persist raw artifacts for reproducibility
# rep.export("runs/tf_mnist", include_csv=True, include_summary=True, include_figures=True)

print("Done. Figures -> results/tf_mnist_figs")
print("Available figure keys:", rep.list_figures())
