import unittest

import torch
from torch import nn

from diffusion import GaussianDiffusion
from models.dit import SimpleDiT
from sample import parse_class_labels


def _small_dit(class_dropout_prob: float = 0.0) -> SimpleDiT:
    return SimpleDiT(
        image_size=16,
        patch_size=4,
        hidden_dim=32,
        depth=2,
        num_heads=4,
        num_classes=3,
        class_dropout_prob=class_dropout_prob,
    )


def _make_output_condition_sensitive(model: SimpleDiT) -> None:
    """打破最终层的零初始化，以直接观察不同条件产生的不同输出。"""
    with torch.no_grad():
        nn.init.normal_(model.final_layer.ada_ln[-1].weight, std=0.1)
        nn.init.normal_(model.final_layer.projection.weight, std=0.1)


class ClassConditionalDiTTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.images = torch.randn(2, 3, 16, 16)
        self.timesteps = torch.tensor([1, 2], dtype=torch.long)

    def test_adaln_zero_starts_with_zero_prediction(self) -> None:
        model = _small_dit()
        output = model(
            self.images,
            self.timesteps,
            torch.tensor([0, 1], dtype=torch.long),
        )

        self.assertEqual(output.shape, self.images.shape)
        torch.testing.assert_close(output, torch.zeros_like(output))

    def test_class_label_changes_prediction(self) -> None:
        model = _small_dit()
        _make_output_condition_sensitive(model)
        model.eval()

        output_0 = model(
            self.images,
            self.timesteps,
            torch.tensor([0, 0], dtype=torch.long),
        )
        output_1 = model(
            self.images,
            self.timesteps,
            torch.tensor([1, 1], dtype=torch.long),
        )

        self.assertFalse(torch.allclose(output_0, output_1))

    def test_full_label_dropout_matches_null_condition(self) -> None:
        model = _small_dit(class_dropout_prob=1.0)
        _make_output_condition_sensitive(model)

        model.train()
        dropped_output = model(
            self.images,
            self.timesteps,
            torch.tensor([0, 1], dtype=torch.long),
        )
        model.eval()
        null_output = model(self.images, self.timesteps, None)

        torch.testing.assert_close(dropped_output, null_output)


class _LabelValueModel(nn.Module):
    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del timesteps
        if class_labels is None:
            return torch.zeros_like(noisy_images)
        values = class_labels.to(noisy_images.dtype).reshape(-1, 1, 1, 1)
        return values.expand_as(noisy_images)


class ClassifierFreeGuidanceTest(unittest.TestCase):
    def test_cfg_combines_conditional_and_unconditional_predictions(self) -> None:
        diffusion = GaussianDiffusion(timesteps=2)
        images = torch.zeros(2, 3, 4, 4)
        timesteps = torch.tensor([1, 1], dtype=torch.long)
        labels = torch.tensor([1, 2], dtype=torch.long)

        prediction = diffusion._model_prediction(
            _LabelValueModel(),
            images,
            timesteps,
            class_labels=labels,
            cfg_scale=3.0,
        )

        expected_values = torch.tensor([3.0, 6.0]).reshape(2, 1, 1, 1)
        torch.testing.assert_close(prediction, expected_values.expand_as(images))


class ClassLabelParsingTest(unittest.TestCase):
    def test_single_name_repeats_for_all_images(self) -> None:
        labels = parse_class_labels("dog", 3, ["cat", "dog"])
        self.assertEqual(labels, [1, 1, 1])

    def test_default_labels_cycle_over_classes(self) -> None:
        labels = parse_class_labels(None, 5, ["cat", "dog"])
        self.assertEqual(labels, [0, 1, 0, 1, 0])


if __name__ == "__main__":
    unittest.main()
