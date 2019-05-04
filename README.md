# Deep Diving into GANs: from theory to production

With our accrued experience with GANs, we would like to guide you through the required steps to go from theory to production with this revolutionary technology.

Starting from the very basic of what a GAN is, passing trough Tensorflow implementation, using the most cutting edge APIs available in the framework, and finally, production-ready serving at scale using Google Cloud ML Engine.

This is the [ZURU Tech](https://zuru.tech/) way of making GANs: enjoy it.

## Workshop's Table of contents

- Introduction to GANs: Theory and Applications
    - Unconditional GAN
    - Conditional GAN

- GANs in Tensorflow:
    - Writing an GAN from scratch: a complete example
    - Define generator with `tf.estimator` API
    - Input pipeline with `tf.data` API
    - How to use `tf.estimator` to train both generator and discriminator?

- TFGAN:
    - API overview
    - Generator and discriminator definition
    - Input pipeline definition
    - Loss function: a bond between generator and discriminator
    - Train end Prediction
    - Export the trained model

- Production:
    - Google Cloud ML
    - Serving at scale

---

## Requirements

This tutorial requires the following packages:

- `python` >= 3.6
- `tensorflow` >=2.0: https://www.tensorflow.org/install/install_linux
- `jupyter`
- `numpy`
- `requests` to run the CelebA Dataset downloader

### Optional, but recommended:

- `virtualenv` to manage a virtual environment
- NVIDIA CUDA®: Compute Unified Device Architecture
- cuDNN: The NVIDIA CUDA® Deep Neural Network library
- **NOTE:** If you have an NVIDIA GPU with Compute Capability 3.0 or higher, you can install `tensorflow-gpu` instead of `tensorflow`.
- Google Cloud account with access to the CloudML APIs (only needed for the serving in production section).
- [Google Cloud SDK](https://cloud.google.com/sdk/) (only needed for the serving in production section).
- `jsonlines` to easily generate .ndjson files for CloudML Engine

## Setting up the environment (Linux, MacOS)

### Clone the repository

```bash
git clone https://github.com/zurutech/gans-from-theory-to-production
cd gans-from-theory-to-production
```

### Prepare a virtual environment

- `virtualenv`: `virtualenv venv && source venv/bin/activate`

### Installing the required packages

```bash
pip install -r no-gpu-requirements.txt
# or pip install -r gpu-requirements if a GPU with Compute Capability >= 3.0 is present
```

### Start your Jupyter server

`jupyter notebook .` or the newer `jupyter lab .`.

---

If you're here, you're ready to go.

Happy workshop!

---

## We're hiring!

Do you just love machine learning and you're also interested in Computer Vision? Join us at [ZURU Tech](https://zuru.tech/)!

## Authors

- Michele "Ubik" De Simoni - https://essays.ubik.tech/ - michele.d[at]zuru.tech
- Paolo Galeone - https://pgaleone.eu/ - paolo[at]zuru.tech
- Federico Di Mattia - federico.d[at]zuru.tech
- Emanuele Ghelfi - emanuele[at]zuru.tech
