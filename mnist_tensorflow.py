"""A small CNN learning to read handwriting (MNIST), in TensorFlow/Keras.

This is the *framework* counterpart to the from-scratch spiral net next door:
there, every line of forward-pass and backprop is hand-written in NumPy so you
can see the machinery. Here Keras handles all of that — so instead of watching
the math, you watch the *result*: a 5x5 panel of real test digits, each labeled
with the model's current guess. They start out mostly wrong (pink) and, epoch by
epoch, flip to correct (cyan) as test accuracy climbs into the high-90s. Same
"starts as mush, then snaps into place" feel — on real handwriting this time.

Run:
    python learning/mnist_tensorflow.py
    python learning/mnist_tensorflow.py --save mnist.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.viz import (PALETTE, card_figure, caption, save_animation,
                        strip_axes, use_headless_if_saving)

# ---- tunables -------------------------------------------------------------
GRID = 5              # GRID x GRID = number of test digits shown on the card
EPOCHS = 12           # one animation frame per epoch
BATCH = 128
HOLD_FRAMES = 8       # linger on the trained result so the loop reads clearly
SEED = 7              # fixed so the panel + training are reproducible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", metavar="OUT.mp4")
    args = ap.parse_args()
    use_headless_if_saving(args.save)

    # Keep TensorFlow quiet and deterministic-ish; CPU is plenty for this net.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    # ---- data -------------------------------------------------------------
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train / 255.0).astype("float32")[..., None]
    x_test = (x_test / 255.0).astype("float32")[..., None]

    # A fixed panel of test digits we'll keep predicting as the model learns.
    n = GRID * GRID
    idx = np.random.choice(len(x_test), n, replace=False)
    panel_x, panel_y = x_test[idx], y_test[idx]

    # ---- model: Conv -> Pool -> Conv -> Dense -> softmax(10) ---------------
    model = tf.keras.Sequential([
        tf.keras.layers.Input((28, 28, 1)),
        tf.keras.layers.Conv2D(16, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    # ---- figure: a GRID x GRID panel of digit tiles -----------------------
    fig, _ = card_figure()
    # card_figure() hands back one axis; we replace it with a tight tile grid.
    fig.clf()
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = fig.subplots(GRID, GRID)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06,
                        wspace=0.12, hspace=0.12)

    labels = []  # the predicted-digit text artist sitting on each tile
    for ax, img in zip(axes.ravel(), panel_x):
        ax.set_facecolor(PALETTE["bg"])
        strip_axes(ax)
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
        t = ax.text(0.5, 0.06, "", transform=ax.transAxes, ha="center",
                    va="bottom", fontsize=11, fontweight="bold",
                    family="monospace")
        labels.append(t)

    cap = fig.text(0.035, 0.965, "epoch  0   acc  --", color=PALETTE["ink"],
                   fontsize=10, alpha=0.9, ha="left", va="top",
                   family="monospace")

    state = {"epoch": 0}

    def refresh_panel(acc):
        preds = model.predict(panel_x, verbose=0).argmax(1)
        for t, pred, truth in zip(labels, preds, panel_y):
            ok = pred == truth
            t.set_text(str(int(pred)))
            t.set_color(PALETTE["cyan"] if ok else PALETTE["pink"])
        cap.set_text(f"epoch {state['epoch']:>2}   acc {acc*100:4.1f}%")
        return [*labels, cap]

    def render(frame):
        # Frames past EPOCHS just hold the final, trained result.
        if frame < EPOCHS:
            state["epoch"] = frame + 1
            model.fit(x_train, y_train, batch_size=BATCH, epochs=1, verbose=0)
            _, acc = model.evaluate(x_test, y_test, verbose=0)
            render.last_acc = acc
        return refresh_panel(render.last_acc)

    render.last_acc = 0.0

    anim = FuncAnimation(fig, render, frames=EPOCHS + HOLD_FRAMES,
                         interval=400, blit=False)

    if args.save:
        save_animation(anim, args.save, fps=3)
    else:
        plt.show()


if __name__ == "__main__":
    main()
