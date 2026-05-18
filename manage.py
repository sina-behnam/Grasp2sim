from dataclasses import dataclass, field
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

@dataclass
class PathsConfig:
    # Primary — set these in .env
    GRASPNET_DATASET: str = field(default_factory=lambda: os.getenv("GRASPNET_DATASET", ""))
    SCENE_ID:         str = field(default_factory=lambda: os.getenv("SCENE_ID", "scene_0000"))
    MODEL_DIR:        str = field(default_factory=lambda: os.getenv("MODEL_DIR", ""))
    HAND_ASSETS:      str = field(default_factory=lambda: os.getenv("HAND_ASSETS", ""))
    GRASP2SIM_OUTPUT: str = field(default_factory=lambda: os.getenv("GRASP2SIM_OUTPUT", ""))
    CAMERA:           str = field(default_factory=lambda: os.getenv("CAMERA", "kinect"))
    # Derived — auto-computed from primaries; override any via .env
    SCENE_DIR:   str = ""
    GRASPS_NPY:  str = ""
    CAMERA_EXTR: str = ""
    CAMERA_POSE: str = ""
    OUTPUT_DIR:  str = ""
    SCENE_XML:   str = ""
    OUTPUT_XML:  str = ""

    def __post_init__(self):
        sd  = os.getenv("SCENE_DIR") or os.path.join(self.GRASPNET_DATASET, self.SCENE_ID)
        cam = self.CAMERA
        out = self.GRASP2SIM_OUTPUT
        sid = self.SCENE_ID
        self.SCENE_DIR   = self.SCENE_DIR   or sd
        self.GRASPS_NPY  = self.GRASPS_NPY  or os.getenv("GRASPS_NPY")  or os.path.join(sd, "grasp_group_mine.npy")
        self.CAMERA_EXTR = self.CAMERA_EXTR or os.getenv("CAMERA_EXTR") or os.path.join(sd, cam, "cam0_wrt_table.npy")
        self.CAMERA_POSE = self.CAMERA_POSE or os.getenv("CAMERA_POSE") or os.path.join(sd, cam, "camera_poses.npy")
        self.OUTPUT_DIR  = self.OUTPUT_DIR  or os.getenv("OUTPUT_DIR")  or os.path.join(out, "experiments")
        self.SCENE_XML   = self.SCENE_XML   or os.getenv("SCENE_XML")   or os.path.join(out, "scenes", f"{sid}_mocap.xml")
        self.OUTPUT_XML  = self.OUTPUT_XML  or os.getenv("OUTPUT_XML")  or os.path.join(out, "scenes", f"{sid}_mocap.xml")


@dataclass
class SimConfig:
    RENDER:          str   = field(default_factory=lambda: os.getenv("RENDER", "off"))
    SEED:            int   = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    DEBUG:           bool  = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "t"))
    DEBUG_LOG_EVERY: int   = field(default_factory=lambda: int(os.getenv("DEBUG_LOG_EVERY", "1")))
    FRICTION_MU:     float = field(default_factory=lambda: float(os.getenv("FRICTION_MU", "5.0")))


@dataclass
class ExperimentConfig:
    TOP_N:    int  = field(default_factory=lambda: int(os.getenv("TOP_N", "10")))
    EXECUTOR: str  = field(default_factory=lambda: os.getenv("EXECUTOR", "descend"))
    VIDEO:    bool = field(default_factory=lambda: os.getenv("VIDEO", "false").lower() in ("true", "1", "t"))


@dataclass
class Config:
    paths: PathsConfig      = field(default_factory=PathsConfig)
    sim:   SimConfig        = field(default_factory=SimConfig)
    exp:   ExperimentConfig = field(default_factory=ExperimentConfig)

    def __post_init__(self):
        for attr, path in [
            ("MODEL_DIR",   self.paths.MODEL_DIR),
            ("HAND_ASSETS", self.paths.HAND_ASSETS),
        ]:
            if path and not os.path.exists(path):
                raise FileNotFoundError(
                    f"Config.paths.{attr} = {path!r} does not exist. "
                    "Check your .env file."
                )

    def __str__(self):
        return (
            f"Config:\n"
            f"  Paths:\n"
            f"    GRASPNET_DATASET: {self.paths.GRASPNET_DATASET}\n"
            f"    SCENE_ID:         {self.paths.SCENE_ID}\n"
            f"    MODEL_DIR:        {self.paths.MODEL_DIR}\n"
            f"    HAND_ASSETS:      {self.paths.HAND_ASSETS}\n"
            f"    GRASP2SIM_OUTPUT: {self.paths.GRASP2SIM_OUTPUT}\n"
            f"    CAMERA:           {self.paths.CAMERA}\n"
            f"    SCENE_DIR:        {self.paths.SCENE_DIR}\n"
            f"    GRASPS_NPY:       {self.paths.GRASPS_NPY}\n"
            f"    CAMERA_EXTR:      {self.paths.CAMERA_EXTR}\n"
            f"    CAMERA_POSE:      {self.paths.CAMERA_POSE}\n"
            f"    OUTPUT_DIR:       {self.paths.OUTPUT_DIR}\n"
            f"    SCENE_XML:        {self.paths.SCENE_XML}\n"
            f"    OUTPUT_XML:       {self.paths.OUTPUT_XML}\n"
            f"  Simulation:\n"
            f"    RENDER:          {self.sim.RENDER}\n"
            f"    SEED:            {self.sim.SEED}\n"
            f"    DEBUG:           {self.sim.DEBUG}\n"
            f"    DEBUG_LOG_EVERY: {self.sim.DEBUG_LOG_EVERY}\n"
            f"    FRICTION_MU:     {self.sim.FRICTION_MU}\n"
            f"  Experiment:\n"
            f"    TOP_N:           {self.exp.TOP_N}\n"
            f"    EXECUTOR:        {self.exp.EXECUTOR}\n"
            f"    VIDEO:           {self.exp.VIDEO}"
        )

config = Config()
