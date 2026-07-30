"""Interactive viewer and deterministic headless frame capture."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import imageio.v2 as imageio
import mujoco
import mujoco.viewer

from .scene_builder import BuiltScene


def launch_passive(scene: BuiltScene) -> mujoco.viewer.Handle:
    return mujoco.viewer.launch_passive(scene.model, scene.data)


class VideoRecorder:
    def __init__(self, scene: BuiltScene, path: Path, fps: int):
        self.scene = scene
        self.path = path
        self.fps = fps
        self.renderer = mujoco.Renderer(
            scene.model,
            height=scene.model.vis.global_.offheight,
            width=scene.model.vis.global_.offwidth,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(path, fps=fps)

    def capture(self) -> None:
        self.renderer.update_scene(self.scene.data, camera="overview")
        self.writer.append_data(self.renderer.render())

    def close(self) -> None:
        self.writer.close()
        self.renderer.close()
