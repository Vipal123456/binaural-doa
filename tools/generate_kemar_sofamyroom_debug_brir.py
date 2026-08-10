#!/usr/bin/env python3
"""Generate a small KEMAR/SofaMyRoom BRIR debug set.

This is intentionally a smoke-test/debug tool, not the full dataset generator.
It verifies the SofaMyRoom executable, KEMAR SOFA file, coordinate convention,
and output WAV metadata before we scale up.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import soundfile as sf


DEFAULT_SOFAMYROOM_BIN = Path("/disk2/bywang/data/sofamyroom/build_doa/sofamyroom")
DEFAULT_KEMAR_SOFA = Path("/disk2/bywang/data/sofamyroom/data/MIT_KEMAR_normal_pinna.sofa")
DEFAULT_CONDA_LIB = Path("/home/bywang/miniconda3/envs/doa/lib")


def kemar_azimuth_to_room_offset(azimuth_deg: float, distance_m: float) -> Tuple[float, float, float]:
    """Map MIT KEMAR horizontal azimuth to SofaMyRoom room coordinates.

    For SofaMyRoom with receiver.orientation = [0 0 0], the cardinal smoke test
    indicates that +Y is listener-left and -Y is listener-right. We therefore
    use +X as frontal direction:

      KEMAR 0 deg   front -> +X
      KEMAR 90 deg  right -> -Y
      KEMAR 180 deg back  -> -X
      KEMAR 270 deg left  -> +Y
    """
    rad = math.radians(float(azimuth_deg))
    return (
        distance_m * math.cos(rad),
        -distance_m * math.sin(rad),
        0.0,
    )


def write_setup(
    path: Path,
    output_prefix: Path,
    sofa_path: Path,
    source_xyz: Tuple[float, float, float],
    receiver_xyz: Tuple[float, float, float],
    room_dims: Tuple[float, float, float],
    fs: int,
    response_duration: float,
    reflection_order: int,
    simulate_diffuse: bool,
) -> None:
    sx, sy, sz = source_xyz
    rx, ry, rz = receiver_xyz
    lx, ly, lz = room_dims
    text = f"""room.dimension              = [ {lx:.6f} {ly:.6f} {lz:.6f} ];
room.humidity               = 0.42;
room.temperature            = 20;

room.surface.frequency      = [  125       250       500       1000      2000      4000];
room.surface.absorption     = [  0.20      0.25      0.30      0.35      0.40      0.45;
                                 0.20      0.25      0.30      0.35      0.40      0.45;
                                 0.20      0.25      0.30      0.35      0.40      0.45;
                                 0.20      0.25      0.30      0.35      0.40      0.45;
                                 0.25      0.30      0.35      0.40      0.45      0.50;
                                 0.25      0.30      0.35      0.40      0.45      0.50 ];
room.surface.diffusion      = [  0.3       0.3       0.3       0.3       0.3       0.3;
                                 0.3       0.3       0.3       0.3       0.3       0.3;
                                 0.3       0.3       0.3       0.3       0.3       0.3;
                                 0.3       0.3       0.3       0.3       0.3       0.3;
                                 0.3       0.3       0.3       0.3       0.3       0.3;
                                 0.3       0.3       0.3       0.3       0.3       0.3 ];

options.fs                  = {int(fs)};
options.responseduration    = {float(response_duration):.6f};
options.bandsperoctave      = 1;
options.referencefrequency  = 125;
options.airabsorption       = false;
options.distanceattenuation = true;
options.subsampleaccuracy   = false;
options.highpasscutoff      = 0;
options.verbose             = false;

options.simulatespecular    = true;
options.reflectionorder     = [ {reflection_order} {reflection_order} {reflection_order} ];

options.simulatediffuse     = {"true" if simulate_diffuse else "false"};
options.numberofrays        = 2000;
options.diffusetimestep     = 0.010;
options.rayenergyfloordB    = -80;
options.uncorrelatednoise   = true;

options.outputname          = '{output_prefix}';
options.mex_saveaswav       = false;

source(1).location           = [ {sx:.6f} {sy:.6f} {sz:.6f} ];
source(1).orientation        = [ 0 0 0 ];
source(1).description        = 'omnidirectional';

receiver(1).location         = [ {rx:.6f} {ry:.6f} {rz:.6f} ];
receiver(1).orientation      = [ 0 0 0 ];
receiver(1).description      = 'SOFA {sofa_path} interp=0 norm=0 resampling=1';
"""
    path.write_text(text, encoding="utf-8")


def parse_angles(values: str) -> Iterable[int]:
    for part in values.split(","):
        part = part.strip()
        if part:
            yield int(part)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sofamyroom_bin", type=Path, default=DEFAULT_SOFAMYROOM_BIN)
    parser.add_argument("--sofa_path", type=Path, default=DEFAULT_KEMAR_SOFA)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/kemar_sofamyroom_debug_brir"))
    parser.add_argument("--angles", type=str, default="0,90,180,270")
    parser.add_argument("--distance", type=float, default=1.5)
    parser.add_argument("--room_dims", type=float, nargs=3, default=(6.0, 5.0, 3.0))
    parser.add_argument("--receiver_xyz", type=float, nargs=3, default=(3.0, 2.5, 1.4))
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--response_duration", type=float, default=1.2)
    parser.add_argument("--reflection_order", type=int, default=10)
    parser.add_argument("--simulate_diffuse", action="store_true")
    parser.add_argument("--conda_lib", type=Path, default=DEFAULT_CONDA_LIB)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{args.conda_lib}:{env.get('LD_LIBRARY_PATH', '')}"

    receiver = np.asarray(args.receiver_xyz, dtype=np.float64)
    room_dims = np.asarray(args.room_dims, dtype=np.float64)
    for az in parse_angles(args.angles):
        offset = np.asarray(kemar_azimuth_to_room_offset(az, args.distance), dtype=np.float64)
        source = receiver + offset
        if np.any(source <= 0.0) or np.any(source >= room_dims):
            raise ValueError(f"Source for az={az} is outside room: {source.tolist()}")

        stem = f"az{az:03d}"
        setup_path = args.output_dir / f"{stem}.txt"
        output_prefix = args.output_dir / stem
        write_setup(
            setup_path,
            output_prefix,
            args.sofa_path,
            tuple(source.tolist()),
            tuple(receiver.tolist()),
            tuple(room_dims.tolist()),
            args.fs,
            args.response_duration,
            args.reflection_order,
            args.simulate_diffuse,
        )
        subprocess.run([str(args.sofamyroom_bin), str(setup_path)], check=True, env=env)

        wav_path = args.output_dir / f"{stem}_receiver_0.wav"
        audio, sr = sf.read(str(wav_path), always_2d=True, dtype="float32")
        if audio.shape[1] != 2:
            raise RuntimeError(f"Expected stereo BRIR, got shape {audio.shape} for {wav_path}")
        left = audio[:, 0]
        right = audio[:, 1]
        rows.append({
            "kemar_azimuth_deg": az,
            "source_x": f"{source[0]:.6f}",
            "source_y": f"{source[1]:.6f}",
            "source_z": f"{source[2]:.6f}",
            "receiver_x": f"{receiver[0]:.6f}",
            "receiver_y": f"{receiver[1]:.6f}",
            "receiver_z": f"{receiver[2]:.6f}",
            "sample_rate": sr,
            "num_samples": audio.shape[0],
            "duration_sec": f"{audio.shape[0] / sr:.6f}",
            "peak_index_l": int(np.argmax(np.abs(left))),
            "peak_index_r": int(np.argmax(np.abs(right))),
            "max_abs_l": f"{float(np.max(np.abs(left))):.9f}",
            "max_abs_r": f"{float(np.max(np.abs(right))):.9f}",
            "rms_l": f"{float(np.sqrt(np.mean(left ** 2))):.9f}",
            "rms_r": f"{float(np.sqrt(np.mean(right ** 2))):.9f}",
            "wav_path": str(wav_path),
            "setup_path": str(setup_path),
        })

    report_path = args.output_dir / "metadata.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} BRIRs")
    print(f"Metadata: {report_path}")


if __name__ == "__main__":
    main()
