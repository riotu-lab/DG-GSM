#!/usr/bin/env python3
"""Run native-resolution DG-GSM inference and U3D evaluation in one command.

Protocol used for Table VII:
  1. Load each U3D image at its original 3840x2160 resolution.
  2. Enhance it in one full-image DG-GSM forward pass.
  3. Save a lossless PNG with exactly the same dimensions.
  4. Compute NIQE, BRISQUE, and U3D-PI for the input and enhanced sets.

No resizing, cropping, tiling, or test-time fine-tuning is performed.
U3D-PI is defined as (NIQE + BRISQUE) / 2; lower is better.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import time

import numpy as np
from PIL import Image
import pyiqa
import torch
import yaml

from models.DG_GSM import DGGSM


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/config.yaml"),
        help="Configuration containing the model section; image_size is ignored.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("weights/iSAID-dark_best.pth"),
        help="Path to the pretrained DG-GSM checkpoint.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the original U3D low-light test images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/U3D/DG-GSM"),
        help="Directory for native-resolution enhanced PNG images.",
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=Path("results/U3D/metrics"),
        help="Directory for inference and metric CSV files.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--amp", action="store_true", help="Use CUDA automatic mixed precision."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing enhanced images."
    )
    parser.add_argument(
        "--expected-count", type=int, default=100, help="Expected U3D image count."
    )
    parser.add_argument(
        "--expected-width", type=int, default=3840, help="Expected native width."
    )
    parser.add_argument(
        "--expected-height", type=int, default=2160, help="Expected native height."
    )
    return parser.parse_args()


def image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def index_by_stem(files: list[Path], set_name: str) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    for path in files:
        if path.stem in indexed:
            raise ValueError(f"{set_name}: duplicate image stem {path.stem!r}.")
        indexed[path.stem] = path
    return indexed


def validate_file_set(
    files: list[Path],
    set_name: str,
    expected_count: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, Path]:
    if len(files) != expected_count:
        raise ValueError(
            f"{set_name}: expected {expected_count} images, found {len(files)}."
        )

    indexed = index_by_stem(files, set_name)
    for path in files:
        with Image.open(path) as image:
            width, height = image.size
        if (width, height) != (expected_width, expected_height):
            raise ValueError(
                f"{set_name}/{path.name}: found {width}x{height}; expected "
                f"{expected_width}x{expected_height}."
            )
    return indexed


def read_rgb_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.array(image.convert("RGB"), dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def save_lossless_png(tensor: torch.Tensor, path: Path) -> None:
    array = (
        tensor.detach()
        .squeeze(0)
        .clamp(0, 1)
        .permute(1, 2, 0)
        .mul(255.0)
        .round()
        .byte()
        .cpu()
        .numpy()
    )
    Image.fromarray(array, mode="RGB").save(path, format="PNG", compress_level=1)


def extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net", "params"):
            value = checkpoint.get(key)
            if isinstance(value, dict) and value:
                checkpoint = value
                break

    if not isinstance(checkpoint, dict) or not checkpoint:
        raise TypeError("Checkpoint does not contain a valid model state dictionary.")

    state_dict: dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        if not isinstance(value, torch.Tensor):
            continue
        while key.startswith("module.") or key.startswith("model."):
            key = key.split(".", 1)[1]
        state_dict[key] = value

    if not state_dict:
        raise TypeError("No tensor parameters were found in the checkpoint.")
    return state_dict


def load_model(
    config_path: Path, checkpoint_path: Path, device: torch.device
) -> DGGSM:
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or "model" not in config:
        raise KeyError(f"Missing 'model' section in configuration: {config_path}")

    model = DGGSM(**config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint), strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def infer_full_image(
    model: torch.nn.Module, image: torch.Tensor, use_amp: bool
) -> torch.Tensor:
    amp_enabled = use_amp and image.device.type == "cuda"
    with torch.autocast(device_type=image.device.type, enabled=amp_enabled):
        output = model(image)
    return output.float().clamp(0, 1)


def run_inference(
    model: torch.nn.Module,
    input_files: list[Path],
    output_dir: Path,
    device: torch.device,
    use_amp: bool,
    overwrite: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, source in enumerate(input_files, start=1):
        destination = output_dir / f"{source.stem}.png"
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {destination}. Use --overwrite to replace it."
            )

        image = read_rgb_tensor(source).to(device, non_blocking=True)
        input_height, input_width = image.shape[-2:]

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        output = infer_full_image(model, image, use_amp)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        if not torch.isfinite(output).all():
            raise RuntimeError(f"Non-finite output detected for {source.name}.")
        if output.shape[-2:] != image.shape[-2:]:
            raise RuntimeError(
                f"Model changed the resolution of {source.name}: "
                f"input={image.shape[-2:]}, output={output.shape[-2:]}."
            )

        save_lossless_png(output, destination)
        with Image.open(destination) as saved:
            output_width, output_height = saved.size
        if (output_width, output_height) != (input_width, input_height):
            raise RuntimeError(
                f"Saved resolution mismatch for {source.name}: "
                f"input={(input_width, input_height)}, "
                f"saved={(output_width, output_height)}."
            )

        peak_gib = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        )
        rows.append(
            {
                "image": source.name,
                "output": destination.name,
                "input_width": input_width,
                "input_height": input_height,
                "output_width": output_width,
                "output_height": output_height,
                "seconds": f"{elapsed:.6f}",
                "peak_memory_gib": f"{peak_gib:.4f}",
            }
        )
        print(
            f"Inference [{index:03d}/{len(input_files):03d}] "
            f"{source.name} -> {destination.name} "
            f"({input_width}x{input_height}, {elapsed:.2f}s, {peak_gib:.2f} GiB)"
        )
    return rows


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def evaluate_sets(
    sets: list[tuple[str, dict[str, Path]]], device: torch.device
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    print("Loading IQA-PyTorch NIQE and BRISQUE metrics...")
    niqe_metric = pyiqa.create_metric("niqe", device=device)
    brisque_metric = pyiqa.create_metric("brisque", device=device)

    per_image_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for set_name, indexed_files in sets:
        niqe_values: list[float] = []
        brisque_values: list[float] = []
        pi_values: list[float] = []

        for index, stem in enumerate(sorted(indexed_files), start=1):
            image_path = indexed_files[stem]
            with torch.inference_mode():
                niqe = float(niqe_metric(str(image_path)).item())
                brisque = float(brisque_metric(str(image_path)).item())
            u3d_pi = (niqe + brisque) / 2.0

            niqe_values.append(niqe)
            brisque_values.append(brisque)
            pi_values.append(u3d_pi)
            per_image_rows.append(
                {
                    "method": set_name,
                    "image": image_path.name,
                    "niqe": f"{niqe:.6f}",
                    "brisque": f"{brisque:.6f}",
                    "u3d_pi": f"{u3d_pi:.6f}",
                }
            )
            print(
                f"Evaluation {set_name} [{index:03d}/{len(indexed_files):03d}] "
                f"NIQE={niqe:.4f}, BRISQUE={brisque:.4f}, PI={u3d_pi:.4f}",
                end="\r",
            )
        print()

        niqe_mean, niqe_std = mean_std(niqe_values)
        brisque_mean, brisque_std = mean_std(brisque_values)
        pi_mean, pi_std = mean_std(pi_values)
        summary_rows.append(
            {
                "method": set_name,
                "images": len(indexed_files),
                "niqe_mean": f"{niqe_mean:.4f}",
                "niqe_std": f"{niqe_std:.4f}",
                "brisque_mean": f"{brisque_mean:.4f}",
                "brisque_std": f"{brisque_std:.4f}",
                "u3d_pi_mean": f"{pi_mean:.4f}",
                "u3d_pi_std": f"{pi_std:.4f}",
            }
        )
        print(
            f"{set_name}: NIQE={niqe_mean:.4f} +/- {niqe_std:.4f}, "
            f"BRISQUE={brisque_mean:.4f} +/- {brisque_std:.4f}, "
            f"U3D-PI={pi_mean:.4f} +/- {pi_std:.4f}"
        )

    return per_image_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows available for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    metrics_dir = args.metrics_dir.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    if not config_path.is_file():
        raise SystemExit(f"Configuration not found: {config_path}")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    input_files = image_files(input_dir)
    input_index = validate_file_set(
        input_files,
        "Input",
        args.expected_count,
        args.expected_width,
        args.expected_height,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(
        "Protocol: one native-resolution forward pass per image; "
        "no resizing, cropping, or tiling"
    )

    model = load_model(config_path, checkpoint_path, device)
    inference_rows = run_inference(
        model,
        input_files,
        output_dir,
        device,
        args.amp,
        args.overwrite,
    )
    write_csv(metrics_dir / "u3d_inference_summary.csv", inference_rows)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output_files = image_files(output_dir)
    output_index = validate_file_set(
        output_files,
        "DG-GSM",
        args.expected_count,
        args.expected_width,
        args.expected_height,
    )
    if set(output_index) != set(input_index):
        missing = sorted(set(input_index) - set(output_index))
        extra = sorted(set(output_index) - set(input_index))
        raise ValueError(
            "Input/output filenames do not match. "
            f"Missing output stems: {missing[:10]}; extra output stems: {extra[:10]}."
        )

    per_image_rows, summary_rows = evaluate_sets(
        [("Input", input_index), ("DG-GSM", output_index)], device
    )
    write_csv(metrics_dir / "u3d_per_image_metrics.csv", per_image_rows)
    write_csv(metrics_dir / "u3d_summary_metrics.csv", summary_rows)

    print(f"Enhanced images: {output_dir}")
    print(f"CSV results: {metrics_dir}")


if __name__ == "__main__":
    main()
