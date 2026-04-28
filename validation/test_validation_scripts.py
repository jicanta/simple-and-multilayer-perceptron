import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


VALIDATION_DIR = Path(__file__).resolve().parent
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))


def load_module(filename, module_name):
    path = VALIDATION_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ValidationScriptsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.step_module = load_module("single-layer-step-perceptron.py", "step_validation")
        cls.linear_module = load_module("single-layer-linear-perceptron.py", "linear_validation")
        cls.non_linear_module = load_module("single-layer-non-linear-perceptron.py", "non_linear_validation")
        cls.multi_layer_module = load_module("multi-layer-perceptron.py", "multi_layer_validation")

    def test_step_perceptron_solves_and(self):
        x, y = self.step_module.build_dataset()
        w, b, errors_by_epoch = self.step_module.train_model(x, y)
        result = self.step_module.evaluate_model(x, y, w, b)

        self.assertTrue(np.array_equal(result["predictions"], np.array([-1, -1, -1, 1])))
        self.assertEqual(errors_by_epoch[-1], 0)
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_linear_perceptron_fits_identity_function(self):
        x, y = self.linear_module.build_dataset("identity")
        train_x, train_y, test_x, test_y = self.linear_module.split_dataset(x, y)
        w, b, losses = self.linear_module.train_model(train_x, train_y)
        result = self.linear_module.evaluate_model(test_x, test_y, w, b)

        self.assertLess(result["mse"], 1e-4)
        self.assertLess(result["max_abs_error"], 2e-2)
        self.assertLess(losses[-1], losses[0])
        self.assertAlmostEqual(w, 1.0, places=2)
        self.assertAlmostEqual(b, 0.0, places=2)

    def test_linear_perceptron_supports_multiple_target_functions(self):
        expected_parameters = {
            "identity": (1.0, 0.0),
            "scaled": (2.0, 0.0),
            "shifted": (1.0, 1.0),
            "descending": (-1.5, 0.5),
        }

        for target_name, (expected_weight, expected_bias) in expected_parameters.items():
            x, y = self.linear_module.build_dataset(target_name)
            train_x, train_y, test_x, test_y = self.linear_module.split_dataset(x, y)
            w, b, _ = self.linear_module.train_model(train_x, train_y)
            result = self.linear_module.evaluate_model(test_x, test_y, w, b)

            self.assertAlmostEqual(w, expected_weight, places=2)
            self.assertAlmostEqual(b, expected_bias, places=2)
            self.assertLess(result["mse"], 1e-4)

    def test_linear_perceptron_rejects_unknown_target_function(self):
        with self.assertRaises(ValueError):
            self.linear_module.build_dataset("unknown")

    def test_regression_validations_use_disjoint_train_and_test_splits(self):
        linear_x, linear_y = self.linear_module.build_dataset("identity")
        linear_train_x, _, linear_test_x, _ = self.linear_module.split_dataset(linear_x, linear_y)

        non_linear_x, non_linear_y = self.non_linear_module.build_dataset()
        non_linear_train_x, _, non_linear_test_x, _ = self.non_linear_module.split_dataset(non_linear_x, non_linear_y)

        self.assertLess(np.max(linear_train_x), np.min(linear_test_x))
        self.assertLess(np.max(non_linear_train_x), np.min(non_linear_test_x))

    def test_non_linear_perceptron_fits_tanh(self):
        x, y = self.non_linear_module.build_dataset()
        train_x, train_y, test_x, test_y = self.non_linear_module.split_dataset(x, y)
        w, b, losses = self.non_linear_module.train_model(train_x, train_y)
        result = self.non_linear_module.evaluate_model(test_x, test_y, w, b)

        self.assertLess(result["mse"], 2e-2)
        self.assertLess(result["max_abs_error"], 3e-1)
        self.assertLess(losses[-1], losses[0])

    def test_multi_layer_perceptron_solves_xor_with_2_2_1(self):
        result = self.multi_layer_module.evaluate_model([2, 2, 1])

        self.assertTrue(np.array_equal(result["predictions"], np.array([1, 1, -1, -1])))
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_multi_layer_perceptron_solves_xor_with_2_3_2_1(self):
        result = self.multi_layer_module.evaluate_model([2, 3, 2, 1])

        self.assertTrue(np.array_equal(result["predictions"], np.array([1, 1, -1, -1])))
        self.assertAlmostEqual(result["accuracy"], 1.0)

    def test_multi_layer_perceptron_rejects_unknown_architecture(self):
        with self.assertRaises(ValueError):
            self.multi_layer_module.build_model([2, 1])


if __name__ == "__main__":
    unittest.main()
