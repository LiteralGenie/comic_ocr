from dataclasses import dataclass
from pathlib import Path

import toml


@dataclass
class Config:
    debug_dir: Path
    font_dir: Path
    image_dir: Path
    vocab_file: Path

    det_arch: str
    det_dataset_dir: Path
    det_model_dir: Path

    reco_arch: str
    reco_dataset_dir: Path
    reco_model_dir: Path

    @classmethod
    def load(cls, data: dict) -> "Config":
        d = data.copy()

        for fp_key in [
            "debug_dir",
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

    @classmethod
    def load_toml(cls, fp: Path) -> "Config":
        data = toml.loads(fp.read_text())
        return cls.load(data)
