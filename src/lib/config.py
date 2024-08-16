from dataclasses import dataclass
from pathlib import Path

import toml


@dataclass
class Config:
    training: "TrainingConfig"

    debug_dir: Path

    det_weights: Path
    reco_weights: Path

    det_arch: str
    reco_arch: str

    det_input_size: int

    @classmethod
    def load(cls, data: dict):
        d = data.copy()

        for fp_key in [
            "debug_dir",
            "det_weights",
            "reco_weights",
        ]:
            d[fp_key] = Path(d[fp_key])

        d["training"] = TrainingConfig.load(d["training"])

        return cls(**d)

    @classmethod
    def load_toml(cls, fp: Path):
        data = toml.loads(fp.read_text())
        return cls.load(data)


@dataclass
class TrainingConfig:
    font_dir: Path
    image_dir: Path
    vocab_file: Path

    det_arch: str
    det_dataset_dir: Path
    det_input_size: int
    det_model_dir: Path

    reco_arch: str
    reco_dataset_dir: Path
    reco_model_dir: Path

    @classmethod
    def load(cls, data: dict):
        d = data.copy()

        for fp_key in [
            "font_dir",
            "image_dir",
            "vocab_file",
            "det_dataset_dir",
            "det_model_dir",
            "reco_dataset_dir",
            "reco_model_dir",
        ]:
            d[fp_key] = Path(d[fp_key])

        return cls(**d)
