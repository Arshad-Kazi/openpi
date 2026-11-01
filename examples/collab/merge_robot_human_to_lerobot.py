"""Merge a robot-demo folder and a human-demo folder into one LeRobot dataset.

Both folders are expected to hold Collab (xArm) HDF5 episodes in the usual
layout, i.e. <dir>/ep<NNNNNN>/episode.h5. This script only decides *which*
episodes get written and in what order; the actual HDF5 -> LeRobot conversion
is delegated to `convert_collab_data_to_lerobot.write_episode`, so both scripts
stay byte-for-byte consistent in how they build states, actions and images.

Each written episode is tagged with its source: the episode name recorded in
meta/episodes.jsonl becomes e.g. `ep000003_robot` / `ep000012_human`, and a
sidecar meta/episode_sources.jsonl maps every episode_index back to its source
folder, source file and repeat number.

Class imbalance is handled with `--counts robot:human`, which gives the exact
number of episodes to write from each folder. A count above what the folder
holds upsamples (episodes are repeated); a count below it downsamples (a random
subset is taken); a count equal to it uses every episode exactly once.

Usage:
  uv run examples/collab/merge_robot_human_to_lerobot.py \
      --robot_dir /home/rt2/datasets/assistive_suturing_robot \
      --human_dir /home/rt2/datasets/assistive_suturing_human \
      --save_dir  /home/rt2/datasets/lerobot/collab_merged_100_50 \
      --counts 100:50 \
      --use_side_camera

  # every robot episode twice (50 -> 100), and only 25 of the human episodes
  uv run examples/collab/merge_robot_human_to_lerobot.py ... --counts 100:25
"""

import dataclasses
import glob
import json
import os
from pathlib import Path
import random
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tyro

from convert_collab_data_to_lerobot import build_features
from convert_collab_data_to_lerobot import episode_length
from convert_collab_data_to_lerobot import write_episode

EPISODES_META = "meta/episodes.jsonl"
SOURCES_META = "meta/episode_sources.jsonl"


@dataclasses.dataclass
class Episode:
    """One planned write: a source episode, possibly one of several repeats."""

    path: str
    source: str  # "robot" or "human"
    name: str  # e.g. "ep000003_robot" or "ep000003_robot_rep1"
    repeat: int  # 0 for the first copy, 1.. for upsampled copies
    n_frames: int


def find_episodes(db_dir: str, source: str) -> list[str]:
    paths = sorted(glob.glob(os.path.join(db_dir, "ep*/episode.h5")))
    if not paths:
        raise FileNotFoundError(f"No ep*/episode.h5 found under {source} dir: {db_dir}")
    return paths


def parse_counts(counts: str) -> tuple[int, int]:
    """Parse "robot:human" into the exact episode count wanted from each source."""
    parts = counts.replace("/", ":").split(":")
    if len(parts) != 2:
        raise ValueError(f"--counts must look like 'robot:human', e.g. '100:50'. Got: {counts!r}")
    try:
        robot_n, human_n = (int(part) for part in parts)
    except ValueError:
        raise ValueError(
            f"--counts parts must be whole numbers of episodes. Got: {counts!r}"
        ) from None
    if robot_n < 0 or human_n < 0:
        raise ValueError(f"--counts parts must both be >= 0. Got: {counts!r}")
    return robot_n, human_n


def plan_source(
    paths: list[str],
    source: str,
    target_count: int,
    downsample_factor: int,
    rng: random.Random,
) -> list[Episode]:
    """Select exactly `target_count` episodes from one source.

    Above the number of episodes available this upsamples: every episode is
    written `target_count // n` times, and the remainder is spread over that
    many randomly chosen distinct episodes so no single one is favoured. Below
    it this downsamples to a random subset. Equal uses each episode once.
    """
    lengths = {p: episode_length(p, downsample_factor) for p in paths}
    n = len(paths)

    if target_count <= n:
        copies = dict.fromkeys(rng.sample(paths, target_count), 1)
    else:
        full_copies, remainder = divmod(target_count, n)
        extra = set(rng.sample(paths, remainder)) if remainder else set()
        copies = {p: full_copies + (1 if p in extra else 0) for p in paths}

    planned: list[Episode] = []
    for path in paths:
        base = os.path.basename(os.path.dirname(path))
        for repeat in range(copies.get(path, 0)):
            name = f"{base}_{source}" if repeat == 0 else f"{base}_{source}_rep{repeat}"
            planned.append(
                Episode(path=path, source=source, name=name, repeat=repeat, n_frames=lengths[path])
            )
    return planned


def annotate_episode_names(root: Path, planned: list[Episode]) -> None:
    """Attach source/name info to the dataset metadata.

    LeRobot's meta/episodes.jsonl has no name field, so we add one (extra keys
    are passed through untouched by its loader) and additionally write a
    standalone meta/episode_sources.jsonl for downstream filtering.
    """
    episodes_path = root / EPISODES_META
    with episodes_path.open() as f:
        records = [json.loads(line) for line in f if line.strip()]

    if len(records) != len(planned):
        raise RuntimeError(
            f"Wrote {len(planned)} episodes but {episodes_path} has {len(records)} entries; "
            "refusing to annotate mismatched metadata."
        )

    with episodes_path.open("w") as f:
        for record, ep in zip(records, planned, strict=True):
            record["name"] = ep.name
            record["source"] = ep.source
            f.write(json.dumps(record) + "\n")

    with (root / SOURCES_META).open("w") as f:
        for record, ep in zip(records, planned, strict=True):
            f.write(
                json.dumps(
                    {
                        "episode_index": record["episode_index"],
                        "name": ep.name,
                        "source": ep.source,
                        "repeat": ep.repeat,
                        "source_path": ep.path,
                    }
                )
                + "\n"
            )


def main(
    robot_dir: str,
    human_dir: str,
    counts: str,
    save_dir: str | None = None,
    repo_name: str = "local/collab_merged",
    fps: int = 10,
    downsample_factor: int = 2,
    image_size: tuple[int, int] = (224, 224),
    default_task: str = "<control_mode> end effector <control_mode> perform the collaborative task",
    use_side_camera: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    dry_run: bool = False,
    push_to_hub: bool = False,
):
    """
    Args:
        robot_dir: Directory of robot demos containing ep*/episode.h5.
        human_dir: Directory of human demos containing ep*/episode.h5.
        save_dir: Output directory for the merged dataset. Defaults to
            HF_LEROBOT_HOME/<repo_name> when omitted.
        repo_name: LeRobot repo id written into the dataset metadata.
        counts: Exact number of episodes to take from each source, written
            "robot:human", e.g. "100:50". A count above the number of episodes
            in that folder upsamples by repeating episodes; a count below it
            downsamples by taking a random subset; equal uses each once.
        fps: Target frames per second (after downsampling).
        downsample_factor: Take every Nth frame from the raw data.
        image_size: (width, height) to resize images to.
        default_task: Language instruction attached to every episode.
        use_side_camera: Whether episodes include side_image_rgb_compressed.
        shuffle: Shuffle the merged episode order (seeded). Off by default, so
            the output is robot episodes followed by human episodes.
        seed: RNG seed for subset selection, remainder sampling and shuffling.
        dry_run: Print the merge plan and exit without writing anything.
        push_to_hub: Whether to push the dataset to HuggingFace Hub.
    """
    rng = random.Random(seed)

    robot_paths = find_episodes(robot_dir, "robot")
    human_paths = find_episodes(human_dir, "human")
    robot_n, human_n = parse_counts(counts)

    planned = plan_source(robot_paths, "robot", robot_n, downsample_factor, rng)
    planned += plan_source(human_paths, "human", human_n, downsample_factor, rng)
    if shuffle:
        rng.shuffle(planned)

    robot_eps = [e for e in planned if e.source == "robot"]
    human_eps = [e for e in planned if e.source == "human"]
    robot_frames = sum(e.n_frames for e in robot_eps)
    human_frames = sum(e.n_frames for e in human_eps)

    def report(label: str, available: int, requested: int, eps: list[Episode], frames: int) -> None:
        if requested > available:
            how = f"UPSAMPLED x{requested / available:.2f}"
        elif requested < available:
            how = f"DOWNSAMPLED, {available - requested} episodes dropped"
        else:
            how = "all episodes, as-is"
        print(f"  {label}: {available} available -> {requested} requested  [{how}]")
        print(
            f"      {len({e.path for e in eps})} unique episodes, "
            f"{len(eps)} written, {frames} frames"
        )

    print(f"robot dir: {robot_dir}")
    print(f"human dir: {human_dir}")
    print(f"requested counts (robot:human) = {counts}")
    report("robot", len(robot_paths), robot_n, robot_eps, robot_frames)
    report("human", len(human_paths), human_n, human_eps, human_frames)
    if human_eps and human_frames:
        print(
            f"  balance: {len(robot_eps) / len(human_eps):.3f}:1 by episodes, "
            f"{robot_frames / human_frames:.3f}:1 by frames"
        )
    print(f"  total -> {len(planned)} episodes / {robot_frames + human_frames} frames")

    if dry_run:
        print("\n--dry_run set, nothing written.")
        return

    output_path = Path(save_dir).expanduser().resolve() if save_dir else HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        root=output_path,
        robot_type="xarm6",
        fps=fps,
        features=build_features(image_size, use_side_camera),
        image_writer_threads=10,
        image_writer_processes=5,
    )

    for i, ep in enumerate(planned):
        print(f"[{i + 1}/{len(planned)}] {ep.name}  <-  {ep.path}")
        write_episode(
            dataset,
            ep.path,
            task=default_task,
            downsample_factor=downsample_factor,
            image_size=image_size,
            use_side_camera=use_side_camera,
        )

    annotate_episode_names(output_path, planned)

    print(f"\nDataset saved to {output_path}")
    print(f"Episode names and sources: {output_path / SOURCES_META}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["collab", "xarm6", "robot-human-merged"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
