# Deep Diving into GANs: from theory to production

With our accrued experience with GANs, we would like to guide you through the required steps to go from theory to production with this revolutionary technology.

Starting from the very basic of what a GAN is, passing trough Tensorflow implementation, using the most cutting edge APIs available in the framework, and finally, production-ready serving at scale using Google Cloud ML Engine.

## Table of contents

- Introduction to GANs: theory and applications
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
    - Train end evaluation
- Production:
    - Export the trained model
    - Google Cloud ML
    - Serving at scale

## Requirements

This tutorial requires the following packages:

- python >= 3.6
- `tensorflow` >=1.9: https://www.tensorflow.org/install/install_linux

Optional, but recommended:

- NVIDIA CUDA®: Compute Unified Device Architecture
- cuDNN: The NVIDIA CUDA® Deep Neural Network library

If you have an NVIDIA GPU with Compute Capability 3.0 or higher, you can install `tensorflow-gpu` instead of `tensorflow`.

- Google Cloud ML account: only needed for the production section.

### Setting up the environment

**Clone the repository**

```bash
git clone https://github.com/zurutech/gans-from-theory-to-production
```

**Installing Tensorflow**

```bash
pip install tensorflow-gpu
# of pip install tensorflow if no GPU with Compute Capability >= 3.0 is present
```

If you're here, you're ready to go.
