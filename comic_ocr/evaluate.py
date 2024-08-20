import argparse
from pathlib import Path

from doctr import models
from doctr.datasets import DetectionDataset
import numpy as np
from torchvision.transforms.v2 import Normalize
from doctr.utils.metrics import LocalizationConfusion
import torch
from doctr import transforms as T
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm


def run(args):
    if args.mode in ["detection", "det"]:
        eval_det(args)
    else:
        raise NotImplementedError()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "mode",
        choices=["detection", "recognition", "det", "reco"],
    )
    parser.add_argument(
        "arch",
        type=str,
        help="https://mindee.github.io/doctr/modules/models.html",
    )
    parser.add_argument(
        "weights",
        type=Path,
    )
    # @todo: script should generate labels but this'll require font / image dir inputs
    parser.add_argument(
        "dataset_dir",
        type=Path,
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=200,
    )

    return parser.parse_args()


def eval_det(args):
    val_set = DetectionDataset(
        img_folder=args.dataset_dir,
        label_path=args.dataset_dir / "val_labels.json",
        sample_transforms=T.SampleCompose(
            (
                [
                    T.Resize(
                        (args.input_size, args.input_size),
                        preserve_aspect_ratio=True,
                        symmetric_pad=True,
                    )
                ]
            )
        ),
    )

    val_loader = DataLoader(
        val_set,  # type: ignore
        batch_size=1,
        drop_last=False,
        sampler=SequentialSampler(val_set),
        pin_memory=torch.cuda.is_available(),
        collate_fn=val_set.collate_fn,
    )

    model = models.detection.__dict__[args.arch](
        pretrained=False,
        class_names=val_set.class_names,
    )

    checkpoint = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(checkpoint)

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        model = model.cuda()

    model.eval()

    val_loss, recall, precision, mean_iou = eval_det_samples(
        args,
        model,
        val_loader,
    )
    print(f"val_loss={val_loss:.6f}")
    print(f"recall={recall:.2%}")
    print(f"precision={precision:.2%}")
    print(f"mean_iou={mean_iou:.2%}")


# adapted from https://github.com/mindee/doctr/blob/0f1f7f627e55a07b7f89b8c019b2da3b3378dda5/references/detection/train_pytorch.py
def eval_det_samples(args, model, val_loader):

    batch_transforms = Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287))
    val_metric = LocalizationConfusion()

    val_loss, batch_cnt = 0, 0
    for idx, (images, targets) in enumerate(tqdm(val_loader, total=args.max_batches)):
        if idx >= args.max_batches:
            break

        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        out = model(images, targets, return_preds=True)
        # Compute metric
        loc_preds = out["preds"]
        for target, loc_pred in zip(targets, loc_preds):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values()):
                val_metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])

        val_loss += out["loss"].item()
        batch_cnt += 1

    val_loss /= batch_cnt
    recall, precision, mean_iou = val_metric.summary()
    return val_loss, recall, precision, mean_iou


if __name__ == "__main__":
    args = parse_args()
    run(args)
