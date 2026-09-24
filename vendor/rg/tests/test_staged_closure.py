from __future__ import annotations

import copy
import unittest

import numpy as np
import torch

from row_rgmap.staged_closure import (
    candidate_signatures,
    canonical_transition_state_sha256,
    optimizer_state_sha256,
)


def is_gauge(name: str) -> bool:
    return name.startswith("layer")


def layer(name: str) -> int:
    return int(name.removeprefix("layer").split(".", 1)[0])


def checkpoint() -> dict:
    model = {
        "layer0.weight": torch.tensor([-2.0, 1.0]),
        "layer1.weight": torch.tensor([3.0, -4.0]),
        "embedding.weight": torch.tensor([[0.5, 0.25], [0.75, -0.5]]),
    }
    states = {
        index: {
            "step": torch.tensor(7.0),
            "exp_avg": torch.tensor([0.1, -0.2])
            if index < 2
            else torch.tensor([[0.3, 0.4], [-0.1, 0.2]]),
            "exp_avg_sq": torch.tensor([0.4, 0.5])
            if index < 2
            else torch.tensor([[0.2, 0.3], [0.4, 0.5]]),
        }
        for index in range(3)
    }
    return {
        "step": 7,
        "trajectory_id": "T",
        "seed": 11,
        "model": model,
        "optimizer": {
            "state": states,
            "param_groups": [
                {
                    "params": [0, 1, 2],
                    "lr": 0.01,
                    "betas": (0.9, 0.95),
                    "eps": 1e-5,
                }
            ],
        },
        "python_random_state": (1, 2),
        "numpy_random_state": (
            "MT19937",
            np.array([1, 2], dtype=np.uint32),
            2,
            0,
            0.0,
        ),
        "torch_random_state": torch.tensor([1, 2], dtype=torch.uint8),
        "cuda_random_state": [torch.tensor([3, 4], dtype=torch.uint8)],
    }


def apply_gauge(value: dict, *, parameters: bool, moments: bool) -> None:
    names = list(value["model"])
    ids = [
        item
        for group in value["optimizer"]["param_groups"]
        for item in group["params"]
    ]
    for name, state_id in zip(names, ids):
        if not is_gauge(name):
            continue
        if parameters:
            value["model"][name].neg_()
        if moments:
            value["optimizer"]["state"][state_id]["exp_avg"].neg_()


class StagedClosureFingerprintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = checkpoint()
        self.phase = {"step": 7, "stream": "fixed"}

    def optimizer_digest(self, value, canonical):
        return optimizer_state_sha256(
            value,
            canonicalize_first_moment=canonical,
            is_gauge_parameter=is_gauge,
            gauge_layer=layer,
        )

    def transition_digest(self, value):
        return canonical_transition_state_sha256(
            value,
            phase=self.phase,
            is_gauge_parameter=is_gauge,
            gauge_layer=layer,
        )

    def test_consistent_gauge_has_same_canonical_state(self) -> None:
        transformed = copy.deepcopy(self.base)
        apply_gauge(transformed, parameters=True, moments=True)
        self.assertNotEqual(
            self.optimizer_digest(self.base, False),
            self.optimizer_digest(transformed, False),
        )
        self.assertEqual(
            self.optimizer_digest(self.base, True),
            self.optimizer_digest(transformed, True),
        )
        self.assertEqual(
            self.transition_digest(self.base), self.transition_digest(transformed)
        )

    def test_parameter_only_gauge_preserves_raw_but_not_canonical_moments(self) -> None:
        transformed = copy.deepcopy(self.base)
        apply_gauge(transformed, parameters=True, moments=False)
        self.assertEqual(
            self.optimizer_digest(self.base, False),
            self.optimizer_digest(transformed, False),
        )
        self.assertNotEqual(
            self.optimizer_digest(self.base, True),
            self.optimizer_digest(transformed, True),
        )

    def test_hidden_model_change_preserves_every_reduced_signature(self) -> None:
        transformed = copy.deepcopy(self.base)
        transformed["model"]["embedding.weight"][1, 0].add_(0.125)
        energy = np.array([0.0, 1.0], dtype=np.float64)
        before = candidate_signatures(
            self.base,
            energy,
            "a" * 64,
            phase=self.phase,
            is_gauge_parameter=is_gauge,
            gauge_layer=layer,
        )
        after = candidate_signatures(
            transformed,
            energy,
            "a" * 64,
            phase=self.phase,
            is_gauge_parameter=is_gauge,
            gauge_layer=layer,
        )
        self.assertEqual(before, after)
        self.assertNotEqual(
            self.transition_digest(self.base), self.transition_digest(transformed)
        )


if __name__ == "__main__":
    unittest.main()
