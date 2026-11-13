"""Tests for DataConfig.frame_mask: MaskedDataset, load_frame_mask, and the masked sim-tray dataset end to end.

The end-to-end tests need the local/sim_tray LeRobot dataset with meta/leak_mask_h16.json and the action-leakage-sim
episode.h5 files it was converted from; they are skipped elsewhere.
"""

import json
import os
import pathlib

import numpy as np
import pytest

from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

LEROBOT_HOME = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot"))
SIM_TRAY = LEROBOT_HOME / "local" / "sim_tray"
H5_DIR = pathlib.Path(__file__).resolve().parents[3].parent / "data" / "tray_demos"


# ------------------------------------------------------------------ MaskedDataset
def test_masked_dataset_maps_indices():
    ds = _data_loader.MaskedDataset(list("abcdefgh"), np.array([0, 2, 3, 7]))
    assert len(ds) == 4
    assert [ds[k] for k in range(4)] == ["a", "c", "d", "h"]
    assert ds[np.int64(1)] == "c"


# ------------------------------------------------------------------ load_frame_mask
def _write_mask(tmp_path: pathlib.Path, **overrides) -> pathlib.Path:
    mask = {
        "repo_id": "local/toy",
        "horizon": 4,
        "fps": 10,
        "total_frames": 10,
        "episode_lengths": [6, 4],
        "excluded": {"start": [1, 2], "flip": [7]},
    }
    mask.update(overrides)
    (tmp_path / "meta").mkdir(exist_ok=True)
    (tmp_path / "meta" / "mask.json").write_text(json.dumps(mask))
    return tmp_path


def _load(root, **overrides):
    kwargs = {"repo_id": "local/toy", "action_horizon": 4, "fps": 10, "total_frames": 10, "episode_lengths": [6, 4]}
    kwargs.update(overrides)
    return _data_loader.load_frame_mask(root, "mask.json", **kwargs)


def test_keeps_everything_not_excluded(tmp_path):
    np.testing.assert_array_equal(_load(_write_mask(tmp_path)), [0, 3, 4, 5, 6, 8, 9])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"action_horizon": 8}, "horizon=4, but this run has horizon=8"),
        ({"fps": 20}, "fps=10"),
        ({"total_frames": 11}, "total_frames=10"),
        ({"repo_id": "local/other"}, "repo_id=local/toy"),
        ({"episode_lengths": [5, 5]}, "episode 0: 6 frames in the mask, 5 in the dataset"),
        ({"episode_lengths": [6, 2, 2]}, "2 episodes in the mask, 3 in the dataset"),
    ],
)
def test_refuses_a_mask_made_for_something_else(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        _load(_write_mask(tmp_path), **overrides)


def test_refuses_out_of_range_indices(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        _load(_write_mask(tmp_path, excluded={"start": [10]}))


def test_missing_mask_says_how_to_make_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_leak_mask.py"):
        _load(tmp_path)


# ------------------------------------------------------------------ the real masked dataset
needs_sim_tray = pytest.mark.skipif(
    not (SIM_TRAY / "meta" / "leak_mask_h16.json").exists() or not H5_DIR.exists(),
    reason="needs local/sim_tray with meta/leak_mask_h16.json and the tray_demos episode.h5 files",
)


def _h5_label(episode: int, frame: int, horizon: int) -> np.ndarray:
    """The training label of (episode, frame) straight from the h5 file: actions frame .. frame+H-1 at 10 Hz."""
    import h5py

    with h5py.File(H5_DIR / f"ep{episode:06d}" / "episode.h5", "r") as f:
        act = np.column_stack([f["robot_desired_pose"][:], f["gripper_commanded_position"][:]])[::2]
    return act[np.minimum(np.arange(frame, frame + horizon), len(act) - 1)]


@needs_sim_tray
def test_noleak_config_trains_on_exactly_the_kept_samples():
    config = _config.get_config("pi05_sim_tray_noleak")
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.frame_mask == "leak_mask_h16.json"
    mask = json.loads((SIM_TRAY / "meta" / "leak_mask_h16.json").read_text())
    ds = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)

    excluded = set(mask["excluded"]["start"]) | set(mask["excluded"]["flip"])
    kept = [i for i in range(mask["total_frames"]) if i not in excluded]
    assert len(ds) == len(kept) == mask["counts"]["kept"]

    # boundary frames of the first 20 episodes, plus random samples: right frame, right label, masked ones absent
    starts = np.cumsum([0, *mask["episode_lengths"][:-1]])
    position = {g: k for k, g in enumerate(kept)}
    checks = []
    for ep in mask["episodes"][:20]:
        e, lamp, flip = ep["episode"], ep["lamp_frame"], ep["flip_frame"]
        frames = [lamp - 16, lamp - 15, lamp - 1, lamp] + ([flip - 16, flip - 15, flip - 1, flip] if flip else [])
        checks += [(e, j) for j in frames if 0 <= j < ep["frames"]]
    rng = np.random.default_rng(0)
    for g in rng.choice(kept, 40, replace=False):
        e = int(np.searchsorted(starts, g, side="right") - 1)
        checks.append((e, int(g - starts[e])))
    n_kept_checked = 0
    for e, j in checks:
        g = int(starts[e] + j)
        if g in excluded:
            assert g not in position
            continue
        item = ds[position[g]]
        assert int(item["episode_index"]) == e
        assert int(item["frame_index"]) == j
        np.testing.assert_allclose(np.asarray(item["actions"]), _h5_label(e, j, 16), atol=1e-5)
        n_kept_checked += 1
    assert n_kept_checked > 60


@needs_sim_tray
def test_unmasked_config_is_unchanged():
    config = _config.get_config("pi05_sim_tray")
    data_config = config.data.create(config.assets_dirs, config.model)
    assert data_config.frame_mask is None
    ds = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    assert len(ds) == json.loads((SIM_TRAY / "meta" / "info.json").read_text())["total_frames"]


@needs_sim_tray
def test_full_input_pipeline_runs_on_the_masked_dataset():
    config = _config.get_config("pi05_sim_tray_noleak")
    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, shuffle=True, num_batches=2)
    batches = list(loader)
    assert len(batches) == 2
    observation, actions = batches[0]
    assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
