"""
Serving a TF 2.0 Model via Cloud Function.

More info: http://bit.ly/310wRZl
"""
from __future__ import annotations

import base64
import json
import os
import tempfile
from io import BytesIO
from typing import TYPE_CHECKING, List

import numpy as np
from PIL import Image
import tensorflow as tf  # pylint: disable=import-error
from google.cloud import storage  # pylint: disable=import-error,no-name-in-module

if TYPE_CHECKING:
    from flask import Request  # pylint: disable=no-name-in-module

DOWNLOAD_CONFIG = {
    "bucket_name": "euroscipy-2019-workshop",
    "model_id": "dcgan-weights-128",
    "destination_folder": tempfile.gettempdir(),
}

headers = {
'Access-Control-Allow-Headers': 'Content-Type',
'Access-Control-Allow-Origin': '*'
}

LATENT_DIMENSION = 100


# We keep model as global variable so we don't have to reload it in case of warm invocations
MODEL: tf.keras.Model = None


def get_model():
    G = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                1024 * 4 * 4, use_bias=False, input_shape=(LATENT_DIMENSION,)
            ),
            tf.keras.layers.Reshape((4, 4, 1024)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                256, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                128, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),
            tf.keras.layers.Conv2DTranspose(
                64, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False, kernel_initializer=weight_init),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(
                3, (5, 5), strides=(2, 2), padding="same", use_bias=False
            ),
            tf.keras.layers.Activation(tf.math.tanh),
        ]
    )
    return G


def inspect_bucket(
    model_id: str, bucket_name: str = "zuru-ml-models", pretty_log=True
) -> List[storage.Blob]:
    """
    Inspect the content of a bucket retrieving the blob of the desired models.

    Args:
        bucket_name (str): Name of the bucket holding the model - Default to "zuru-ml-models".
        model_id (str): Name of the model: es. "semantikaiser".
        pretty_log (bool): If True pretty print logs

    Returns:
        :py:obj:`list` of [:py:class:`google.cloud.storage.Blob`]: Blobs belonging to the target model.

    """
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=model_id)
    if pretty_log:
        from pprint import pprint

        pprint([b.name for b in blobs])
        blobs = storage_client.list_blobs(bucket_name, prefix=model_id)
    return blobs


def download_blobs(model_id: str, destination_folder: str, bucket_name: str) -> str:
    """
    Download all the models related blobs from the bucket.

    Args:
        model_id (str): ID of the model we want to download AKA the name of the folder on
            GCS. This is used as a prefix filtering out all unneeded blobs.
        destination_folder (str): Path of the destination folder where blobs will be downloaded.
        bucket_name (str): Name of the bucket to use as target storage.

    Returns:
        :py:obj:`list` of [:py:obj:`str`]: Returns a list containing the path of the stored SavedModel(s).
    """
    weights_uri: str = ""

    prefix = model_id

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)

    # List blobs iterate in folder
    blobs = bucket.list_blobs(prefix=prefix)  # Excluding folder inside bucket
    for blob in blobs:
        print("Downloading ", blob.name)
        blob_name = blob.name.split(os.path.sep)[-1]
        if blob_name != "":
            destination_uri = os.path.join(destination_folder, blob_name)
            # We are downloading the blob of a file
            try:
                blob.download_to_filename(destination_uri)
                if blob.name.endswith(".index"):
                    weights_uri = destination_uri
            except IsADirectoryError:
                # We cannot download the blob of a folder
                continue

    if weights_uri == "":
        raise ValueError(f"No index file found in {bucket_name}/{model_id}")

    return weights_uri


def postprocess(output: np.ndarray) -> str:
    """
    Post process the model output
    Args:
        output: output image to process

    Returns:
        The image encoded as str
    """
    pil_img = Image.fromarray(output)
    buff = BytesIO()
    image_format = "PNG"
    pil_img.save(buff, format=image_format)
    encoded_img = base64.b64encode(buff.getvalue()).decode("utf-8")
    return encoded_img


def handler(request: Request = None):
    """
    Entry point of the Serveless call.

    Args:
        request (:py:class:`flask.Request`): Flask Request holding our payload.

    """
    # Model load which only happens during cold starts
    global MODEL
    if not MODEL:
        print("Cold start: Loading model")
        print("Downloading saved models")
        weights = download_blobs(**DOWNLOAD_CONFIG)
        MODEL = get_model()
        MODEL.load_weights(weights.replace(".index", ""))
        print("weights loaded")
    else:
        print("Warm start: Using cached model")

    print("Received", request)
    request = request.get_json(silent=True)

    try:
        print("Noise vector", request["noise_vector"])
        # use the feeded noise
        noise = tf.constant(np.array(request["noise_vector"]))
        if noise.shape != (1, LATENT_DIMENSION):
            return ({}, 422, headers)
        print("Using feeded noise")
    except Exception:
        print("No noise vector provided, generating noise")
        noise = tf.random.normal((1, LATENT_DIMENSION))

    # call the model
    output = MODEL.call(noise).numpy().squeeze()

    # back in the rage [0, 255]
    output = ((output + 1) * 127.5).astype(np.uint8)

    # postprocess stage (encode)
    encoded_image = postprocess(output)

    # compose the output response
    return (
        {"base64_image": encoded_image,
         "format": "png",
         "noise_vector": noise.numpy().tolist()
        },
        200,
        headers,
    )
