# Deep Diving into GANs: from theory to production

With our accrued experience with GANs, we would like to guide you through the required steps to go from theory to production with this revolutionary technology.

Starting from the very basic of what a GAN is, passing trough TensorFlow implementation, using the most cutting edge APIs available in the framework, and finally, production-ready serving at scale using Google Cloud ML Engine.

This is the [ZURU Tech](https://zuru.tech/) way of making GANs: enjoy it.

## Workshop's Table of contents

- Introduction to GANs: Theory and Applications
    -  Generator
    -  Discriminator
    -  Intuitive explaination
    -  Non saturating value function
    -  Models definition
    -  Training phase
    -  Types of GANs
    -  Conditional GANs
    -  Applications
        -  Unconditional GAN
        -  Conditional GAN

- GANs in TensorFlow 2.0:
	-  What does a GAN learn?
	-  Input data
	-  Generator and discriminator networks: Keras functional API
	-  Define input and instantiate networks
	-  The loss function and the training procedure
	-  Discriminator loss function
	-  Generator loss function
	-  Gradient ascent
	-  Visualize training
	-  Advantages and disadvantages
	-  Bonus exercise: converting it to a Conditional GAN

- Writing a GAN using AshPy and TensorFlow Datasets
	-  [AshPy Essentials](https://github.com/zurutech/ashpy)
	-  tfds and AshPy input format
	-  Getting the data ready to use
	-  DCGAN Theory and Practice
		- Generator: from noise to insight
		- Deconvolution
		- Batch Normalization
		- Discriminator
		- Loss function: a bridge between two networks
	-  Training
	-  Tensorboard
	-  Towards Serving

- Production:
    - Google Cloud ML
    - Serving at scale

---

## Requirements

<!--
TODO: remove the comment when colab will support Python 3.7
**NOTE**: every notebook has a "try in a colab notebook" button you can use, to directly load the notebook in a colab instance and run it, without the need to set up the environment by yourself.
-->

This tutorial requires the following packages:

- `python` >= 3.7
- `tensorflow` >=2.0: https://www.tensorflow.org/install/install_linux
- `jupyter`
- `numpy`

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
- Emanuele Ghelfi - https://emanueleghelfi.github.io - emanuele[at]zuru.tech
