# Handwriting Recognition: TensorFlow CNN

A convolutional neural network in TensorFlow learns to recognise handwritten digits from the MNIST dataset.

Part of my portfolio of small, from-scratch visualisations of computer-science ideas. Built on numpy and matplotlib, so every moving part is visible.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python mnist_tensorflow.py                  # live animated window
python mnist_tensorflow.py --save out.gif   # export a looping GIF
python mnist_tensorflow.py --save out.mp4   # smaller file, best for the web (needs ffmpeg)
```
