# adapted from https://github.com/mindee/doctr/blob/ed927bd8f7a7eac800c5b965b55aab72e262571e/references/classification/utils.py

# ---

# Copyright (C) 2021-2024, Mindee.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://opensource.org/licenses/Apache-2.0> for full license details.

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import MultiplicativeLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def plot_samples(images, targets):
    # Unnormalize image
    num_samples = min(len(images), 12)
    num_cols = min(len(images), 8)
    num_rows = int(math.ceil(num_samples / num_cols))
    _, axes = plt.subplots(num_rows, num_cols, figsize=(20, 5))
    for idx in range(num_samples):
        img = (255 * images[idx].numpy()).round().clip(0, 255).astype(np.uint8)
        if img.shape[0] == 3 and img.shape[2] != 3:
            img = img.transpose(1, 2, 0)

        row_idx = idx // num_cols
        col_idx = idx % num_cols

        ax = axes[row_idx] if num_rows > 1 else axes
        ax = ax[col_idx] if num_cols > 1 else ax

        ax.imshow(img)
        ax.set_title(targets[idx])
    # Disable axis
    for ax in axes.ravel():
        ax.axis("off")
    plt.show()


def plot_recorder(lr_recorder, loss_recorder, beta: float = 0.95, **kwargs) -> None:
    """Display the results of the LR grid search.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py

    Args:
    ----
        lr_recorder: list of LR values
        loss_recorder: list of loss values
        beta (float, optional): smoothing factor
        **kwargs: keyword arguments from `matplotlib.pyplot.show`
    """
    if len(lr_recorder) != len(loss_recorder) or len(lr_recorder) == 0:
        raise AssertionError(
            "Both `lr_recorder` and `loss_recorder` should have the same length"
        )

    # Exp moving average of loss
    smoothed_losses = []
    avg_loss = 0.0
    for idx, loss in enumerate(loss_recorder):
        avg_loss = beta * avg_loss + (1 - beta) * loss
        smoothed_losses.append(avg_loss / (1 - beta ** (idx + 1)))

    # Properly rescale Y-axis
    data_slice = slice(
        min(len(loss_recorder) // 10, 10),
        (
            -min(len(loss_recorder) // 20, 5)
            if len(loss_recorder) >= 20
            else len(loss_recorder)
        ),
    )
    vals = np.array(smoothed_losses[data_slice])
    min_idx = vals.argmin()
    max_val = vals.max() if min_idx is None else vals[: min_idx + 1].max()  # type: ignore[misc]
    delta = max_val - vals[min_idx]

    plt.plot(lr_recorder[data_slice], smoothed_losses[data_slice])
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Training loss")
    plt.ylim(vals[min_idx] - 0.1 * delta, max_val + 0.2 * delta)
    plt.grid(True, linestyle="--", axis="x")
    plt.show(**kwargs)


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit_one_epoch(
    model,
    train_loader,
    batch_transforms,
    optimizer,
    scheduler,
    amp=False,
):
    if amp:
        scaler = torch.cuda.amp.GradScaler()

    losses = []
    loss_window_width = max(len(train_loader) // 100, 15)

    model.train()
    # Iterate over the batches of the dataset
    pbar = tqdm(train_loader, position=1)
    for images, targets in pbar:
        if torch.cuda.is_available():
            images = images.cuda()
        images = batch_transforms(images)

        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        scheduler.step()

        losses.append(train_loss.item())
        if (overflow := len(losses) - loss_window_width) > 0:
            losses = losses[overflow:]

        loss_avg = sum(losses) / len(losses)

        pbar.set_description(f"Training loss: {loss_avg:.6f}")


def record_lr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    batch_transforms,
    optimizer,
    start_lr: float = 1e-7,
    end_lr: float = 1,
    num_it: int = 100,
    amp: bool = False,
):
    """Gridsearch the optimal learning rate for the training.
    Adapted from https://github.com/frgfm/Holocron/blob/master/holocron/trainer/core.py
    """
    if num_it > len(train_loader):
        raise ValueError(
            "the value of `num_it` needs to be lower than the number of available batches"
        )

    model = model.train()
    # Update param groups & LR
    optimizer.defaults["lr"] = start_lr
    for pgroup in optimizer.param_groups:
        pgroup["lr"] = start_lr

    gamma = (end_lr / start_lr) ** (1 / (num_it - 1))
    scheduler = MultiplicativeLR(optimizer, lambda step: gamma)

    lr_recorder = [start_lr * gamma**idx for idx in range(num_it)]
    loss_recorder = []

    if amp:
        scaler = torch.cuda.amp.GradScaler()

    for batch_idx, (images, targets) in enumerate(train_loader):
        if torch.cuda.is_available():
            images = images.cuda()

        images = batch_transforms(images)

        # Forward, Backward & update
        optimizer.zero_grad()
        if amp:
            with torch.cuda.amp.autocast():
                train_loss = model(images, targets)["loss"]
            scaler.scale(train_loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            # Update the params
            scaler.step(optimizer)
            scaler.update()
        else:
            train_loss = model(images, targets)["loss"]
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        # Update LR
        scheduler.step()

        # Record
        if not torch.isfinite(train_loss):
            if batch_idx == 0:
                raise ValueError("loss value is NaN or inf.")
            else:
                break
        loss_recorder.append(train_loss.item())
        # Stop after the number of iterations
        if batch_idx + 1 == num_it:
            break

    return lr_recorder[: len(loss_recorder)], loss_recorder
