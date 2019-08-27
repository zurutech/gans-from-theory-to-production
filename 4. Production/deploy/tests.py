import base64
from unittest.mock import Mock
import numpy as np
import cv2

from main import handler


def test_main():
    """
    Test the handler function.
    Assert the return value is 200
    and the response contains the noise vector and the image
    """

    # mock the request
    req = Mock()
    req.get_json = lambda silent: {}

    # call the handler
    out = handler(req)

    # assert the response is correctly formed
    assert out[1] == 200
    assert out[0]["noise_vector"]
    assert out[0]["base64_image"]

    # try decoding the image
    # decode image bytes
    image_bytes = base64.b64decode(out[0]["base64_image"])

    # decode bytes as rgb image
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # assert the image has the correct shape
    assert decoded.shape == (64, 64, 3)


def test_with_noise():
    """
    Test the handler function.
    Assert the return value is 200
    and the response contains the noise vector and the image
    """

    # generate random noise vector
    noise = np.random.normal(size=(1, 100))

    # mock the request with the noise vector
    req = Mock()
    req.get_json = lambda silent: {"noise_vector": noise.tolist()}

    # call the handler passing the request
    out = handler(req)

    # assert the response is correctly formed
    assert out[1] == 200

    # assert the latent vector is the same
    np.testing.assert_almost_equal(np.array(out[0]["noise_vector"]), noise)
    assert out[0]["base64_image"]

    # try decoding the image
    # decode image bytes
    image_bytes = base64.b64decode(out[0]["base64_image"])

    # decode bytes as rgb image
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # assert the image has the correct shape
    assert decoded.shape == (64, 64, 3)


def test_noise_error():
    # the latent space is 200 instead of 100
    noise = np.random.normal(size=(1, 200))

    # mock the request with the wrong noise vector
    req = Mock()
    req.get_json = lambda silent: {"noise_vector": noise.tolist()}

    # call the handler passing the request
    out = handler(req)

    # the response should be a user error
    assert out[1] == 400
