"""Inputs/Outputs mapping for the Collab (xArm) pick-and-place dataset.

Your dataset has:
  - state: 8D  (xyz position + quaternion + gripper_actual_position)
  - action: 8D (xyz position + quaternion + gripper_commanded_position)
  - images: mount camera (third-person) + gripper camera (wrist) + optional side camera

The model expects three image slots (base, left_wrist, right_wrist).
We map mount → base, gripper → left_wrist, side → right_wrist.
The side camera is optional (off by default): when absent, right_wrist is
zero-filled and masked off. Toggle it with CollabInputs(use_side_camera=True).
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_collab_example(use_side_camera: bool = False) -> dict:
    """Creates a random input example for testing the Collab policy."""
    example = {
        "observation/state": np.random.rand(8).astype(np.float32),
        "observation/mount_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/gripper_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "pick up the object",
    }
    if use_side_camera:
        example["observation/side_image"] = np.random.randint(256, size=(224, 224, 3), dtype=np.uint8)
    return example


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    # LeRobot stores as (C, H, W), model expects (H, W, C)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class CollabInputs(transforms.DataTransformFn):
    """Maps Collab dataset observations → model input format.

    Used during both training (on LeRobot data) and inference (from policy server).
    The key names here must match what the repack_transform produces (training) or
    what your inference client sends (deployment).
    """

    model_type: _model.ModelType
    # Whether the dataset/deployment provides a side camera (mapped to right_wrist_0_rgb).
    # When False (default), that slot is zero-filled and masked off — except for pi0-FAST,
    # which cannot mask images and instead sees a black image.
    use_side_camera: bool = False

    def __call__(self, data: dict) -> dict:
        # Mount camera = third-person view → base_0_rgb
        mount_image = _parse_image(data["observation/mount_image"])
        # Gripper camera = wrist view → left_wrist_0_rgb
        gripper_image = _parse_image(data["observation/gripper_image"])

        # Side camera → right_wrist_0_rgb. Optional: when absent, pad with zeros and mask off.
        if self.use_side_camera:
            side_image = _parse_image(data["observation/side_image"])
            side_mask = np.True_
        else:
            side_image = np.zeros_like(mount_image)
            side_mask = np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_

        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": mount_image,
                "left_wrist_0_rgb": gripper_image,
                "right_wrist_0_rgb": side_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": side_mask,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class CollabOutputs(transforms.DataTransformFn):
    """Maps model output actions back to dataset format.

    Action dim = 8: (xyz + quaternion + gripper).
    The model pads actions to its internal dim, so we slice back to 8.
    """

    # 7D pose + 1D gripper = 8D
    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
