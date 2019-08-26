from unittest.mock import Mock
import numpy as np

from main import handler


def test_main():
    req = Mock()
    req.get_json = lambda silent: {}
    out = handler(req)
    assert out[1] == 200

def test_with_noise():
    noise = np.random.normal(size=(1, 100))
    req = Mock()
    req.get_json = lambda silent: {"noise_vector": noise.tolist()}
    out = handler(req)
    assert out[1] == 200
