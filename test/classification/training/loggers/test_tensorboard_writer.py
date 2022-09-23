import json
import os
import shutil
import tempfile
import unittest

from snorkel.classification.training.loggers import TensorBoardWriter
from snorkel.types import Config


class TempConfig(Config):
    a: int = 42
    b: str = "foo"


class TestTensorBoardWriter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)
