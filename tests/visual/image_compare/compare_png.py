import argparse
import json
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOImage import vtkPNGReader, vtkPNGWriter


def load_png(path):
    path = Path(path).resolve()
    reader = vtkPNGReader()
    reader.SetFileName(str(path))
    reader.Update()

    image = reader.GetOutput()
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise ValueError(f"Failed to read PNG scalar data from {path}.")

    width, height, _ = image.GetDimensions()
    num_channels = scalars.GetNumberOfComponents()
    array = vtk_to_numpy(scalars).reshape(height, width, num_channels)
    return array, path


def save_png(path, image_array):
    image_array = np.asarray(image_array)
    if image_array.ndim != 3:
        raise ValueError(
            "Expected an image array with shape (height, width, channels)."
        )

    height, width, num_channels = image_array.shape
    image_array = np.ascontiguousarray(image_array.astype(np.uint8))

    image = vtkImageData()
    image.SetDimensions(width, height, 1)

    vtk_array = numpy_to_vtk(
        image_array.reshape(-1, num_channels),
        deep=True,
        array_type=None,
    )
    vtk_array.SetNumberOfComponents(num_channels)
    vtk_array.SetName("ImageScalars")
    image.GetPointData().SetScalars(vtk_array)

    writer = vtkPNGWriter()
    writer.SetFileName(str(Path(path).resolve()))
    writer.SetInputData(image)
    writer.Write()


def compare_images(image_a, image_b):
    if image_a.shape != image_b.shape:
        raise ValueError(f"Image shape mismatch: {image_a.shape} vs {image_b.shape}.")

    diff = np.abs(image_a.astype(np.int16) - image_b.astype(np.int16))
    diff_gray = diff.max(axis=2)

    return {
        "shape": list(image_a.shape),
        "mean_abs_error": float(diff.mean()),
        "max_abs_error": int(diff.max()),
        "rmse": float(np.sqrt(np.mean(diff.astype(np.float64) ** 2))),
        "nonzero_pixel_ratio": float(np.count_nonzero(diff_gray) / diff_gray.size),
        "per_channel_mean_abs_error": [
            float(diff[:, :, channel].mean()) for channel in range(diff.shape[2])
        ],
        "diff_image": diff.astype(np.uint8),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare two PNG images and report how different they are."
    )
    parser.add_argument("reference", help="Path to the reference PNG.")
    parser.add_argument("candidate", help="Path to the candidate PNG.")
    parser.add_argument(
        "--diff-out",
        help="Optional path to save an absolute-difference PNG.",
    )
    parser.add_argument(
        "--mean-threshold",
        type=float,
        default=None,
        help="Fail if mean absolute error exceeds this value.",
    )
    parser.add_argument(
        "--max-threshold",
        type=float,
        default=None,
        help="Fail if maximum absolute error exceeds this value.",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=None,
        help="Fail if nonzero pixel ratio exceeds this value.",
    )
    args = parser.parse_args()

    image_ref, path_ref = load_png(args.reference)
    image_candidate, path_candidate = load_png(args.candidate)

    result = compare_images(image_ref, image_candidate)
    diff_image = result.pop("diff_image")
    result["reference"] = str(path_ref)
    result["candidate"] = str(path_candidate)

    if args.diff_out:
        diff_out = Path(args.diff_out).resolve()
        diff_out.parent.mkdir(parents=True, exist_ok=True)
        save_png(diff_out, diff_image)
        result["diff_out"] = str(diff_out)

    print(json.dumps(result, indent=2))

    failures = []
    if (
        args.mean_threshold is not None
        and result["mean_abs_error"] > args.mean_threshold
    ):
        failures.append(
            f"mean_abs_error={result['mean_abs_error']:.6f} > {args.mean_threshold}"
        )
    if args.max_threshold is not None and result["max_abs_error"] > args.max_threshold:
        failures.append(
            f"max_abs_error={result['max_abs_error']} > {args.max_threshold}"
        )
    if (
        args.ratio_threshold is not None
        and result["nonzero_pixel_ratio"] > args.ratio_threshold
    ):
        failures.append(
            "nonzero_pixel_ratio="
            f"{result['nonzero_pixel_ratio']:.6f} > {args.ratio_threshold}"
        )

    if failures:
        raise SystemExit("Image comparison failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
