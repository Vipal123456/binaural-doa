# KEMAR + SofaMyRoom Reverberant DOA Dataset Plan

## Current Toolchain Status

- SofaMyRoom source: `/disk2/bywang/data/sofamyroom`
- Built executable: `/disk2/bywang/data/sofamyroom/build_doa/sofamyroom`
- KEMAR SOFA: `/disk2/bywang/data/sofamyroom/data/MIT_KEMAR_normal_pinna.sofa`
- Conda FFTW runtime: `/home/bywang/miniconda3/envs/doa/lib`
- Debug script: `tools/generate_kemar_sofamyroom_debug_brir.py`
- Verified output:
  - 48 kHz BRIR generation works.
  - `specular + diffuse` generation works.
  - Four cardinal azimuths pass left/right sanity checks.

Run environment:

```bash
LD_LIBRARY_PATH=/home/bywang/miniconda3/envs/doa/lib:$LD_LIBRARY_PATH
```

## Coordinate Convention

Use MIT KEMAR horizontal azimuth classes:

- `0 deg`: front
- `90 deg`: right
- `180 deg`: back
- `270 deg`: left

For SofaMyRoom with `receiver.orientation = [0 0 0]`, the checked coordinate mapping is:

- KEMAR `0 deg` front -> room `+X`
- KEMAR `90 deg` right -> room `-Y`
- KEMAR `180 deg` back -> room `-X`
- KEMAR `270 deg` left -> room `+Y`

Source offset from the receiver:

```python
x_offset = distance * cos(azimuth_rad)
y_offset = -distance * sin(azimuth_rad)
z_offset = 0
```

Store both angle forms in metadata:

- `kemar_azimuth_deg`: `[0, 5, ..., 355]`
- `azimuth_deg`: compatible wrapped form in `[-180, 180)`

## Dataset Scope

- Task: static horizontal binaural DOA classification.
- Classes: 72 classes at 5 degree spacing.
- Elevation: fixed `0 deg`.
- Listener: fixed KEMAR dummy head.
- Ear height: fixed `1.4 m`.
- Speech: LibriSpeech.
- Noise: DEMAND, added after binaural reverberant rendering.
- BRIR method: SofaMyRoom full room BRIR with specular and diffuse components.

## Training Rooms

Training should sample rooms continuously within size-dependent ranges:

| Room size | Length | Width | Height | RT60 target |
|---|---:|---:|---:|---:|
| small | 4.0-5.0 m | 3.5-4.5 m | 2.5-3.0 m | 0.25-0.45 s |
| medium | 5.5-7.0 m | 4.5-6.0 m | 2.8-3.2 m | 0.35-0.65 s |
| large | 8.0-10.0 m | 6.0-8.0 m | 3.0-4.0 m | 0.50-0.80 s |

Room size should be sampled uniformly across `small`, `medium`, and `large`.

## Test Rooms

Use fixed rooms for clean plotting. Recommended two rooms per size:

| Room ID | Size | Dimensions | RT60 target |
|---|---|---:|---:|
| S1 | small | 4.2 x 3.8 x 2.6 m | 0.30 s |
| S2 | small | 4.8 x 4.2 x 2.8 m | 0.40 s |
| M1 | medium | 5.8 x 4.8 x 3.0 m | 0.45 s |
| M2 | medium | 6.8 x 5.8 x 3.2 m | 0.60 s |
| L1 | large | 8.2 x 6.2 x 3.2 m | 0.65 s |
| L2 | large | 9.8 x 7.8 x 3.8 m | 0.80 s |

Main paper plots should aggregate by `room_size` with mean and error bars across the two room IDs. Appendix plots can show each `room_id` separately.

## Source Distance and Geometry

Training distance:

| Room size | Distance range |
|---|---:|
| small | 1.0-1.5 m |
| medium | 1.0-1.8 m |
| large | 1.0-2.0 m |

Testing distance:

| Room size | Distance values |
|---|---:|
| small | 1.0, 1.25, 1.5 m |
| medium | 1.0, 1.5, 1.8 m |
| large | 1.0, 1.5, 2.0 m |

Geometry constraints:

- Receiver wall clearance: at least `0.8 m`.
- Source wall clearance: at least `0.55 m`.
- Source and receiver height: `1.4 m`.
- If a sampled angle or distance violates clearance, resample geometry.

## Reverberation Generation

Formal generation should not use the debug script absorption values directly. The debug absorption is only for toolchain verification and produces a very weak late tail in the tested room.

Recommended formal approach:

1. Sample room dimensions and target RT60.
2. Convert target RT60 to frequency-independent or mildly frequency-shaped absorption using Sabine/Eyring.
3. Apply a SofaMyRoom calibration scale to the Eyring absorption. The current generator uses profile-specific defaults:
   - small rooms: `0.60`
   - medium rooms: `0.55`
   - large rooms: `0.45`
   A pilot generated with the older global `0.35` scale produced RT60 values that were too long, especially in small and medium rooms.
4. Clamp absorption to a physically plausible range, for example `0.02-0.85`.
5. Generate SofaMyRoom BRIR at 48 kHz with:
   - `options.simulatespecular = true`
   - `options.reflectionorder = [10 10 10]`
   - `options.simulatediffuse = true`
   - `options.numberofrays = 2000`
   - `options.diffusetimestep = 0.010`
   - `options.rayenergyfloordB = -80`
   - `options.uncorrelatednoise = true`
   - `options.responseduration = 1.2`
6. Estimate actual RT60 from the generated BRIR.
7. Downsample rendered binaural speech to 16 kHz for model training.
8. Store both `target_rt60` and `estimated_rt60` in metadata.

RT60 should be treated as a target, not a guaranteed exact value.

## Noise

Use additive binaural post-mix noise:

- Render clean reverberant binaural speech first.
- Sample two DEMAND channels or two decorrelated segments.
- Scale noise to target SNR using the binaural reverberant speech power.
- Mix after BRIR convolution.

Suggested scene split:

- Train/val seen scenes: `OOFFICE`, `PCAFETER`, `TMETRO`
- Test unseen scenes: `TBUS`, `SPSQUARE`, `NPARK`

SNR:

- Train: continuous uniform `[-10, 10] dB`
- Test: `clean`, `10`, `5`, `0`, `-5`, `-10 dB`

## Metadata Fields

Minimum required fields:

- `file_id`
- `split`
- `kemar_azimuth_deg`
- `azimuth_deg`
- `doa_class`
- `elevation_deg`
- `room_size`
- `room_id`
- `room_dims_m`
- `target_rt60`
- `estimated_rt60`
- `receiver_xyz`
- `source_xyz`
- `source_distance_m`
- `source_wall_clearance_m`
- `receiver_wall_clearance_m`
- `sofamyroom_setup_path`
- `brir_path`
- `speech_path`
- `noise_scene`
- `noise_path_l`
- `noise_path_r`
- `snr_db`
- `sample_rate`
- `duration_sec`

## Plotting Plan

Main plots:

- Overall circular MAE and accuracy.
- MAE versus SNR.
- MAE versus room size.
- MAE versus estimated RT60 bin.
- MAE versus source distance.
- Azimuth-sector error plot: front, right, back, left, oblique sectors.
- Confusion matrix over 72 azimuth classes.

For two fixed test rooms per size:

- Main figure: aggregate by `room_size`, error bars over room IDs and utterances.
- Supplement: separate curves for `S1`, `S2`, `M1`, `M2`, `L1`, `L2`.

## Immediate Next Implementation Step

Formal generator:

```text
tools/generate_kemar_sofamyroom_dataset.py
```

It should reuse existing project utilities for LibriSpeech loading, DEMAND loading, SNR mixing, peak normalization, and balanced class scheduling, but it should use SofaMyRoom as the only BRIR generator.

Smoke train generation:

```bash
LD_LIBRARY_PATH=/home/bywang/miniconda3/envs/doa/lib:$LD_LIBRARY_PATH \
/home/bywang/miniconda3/envs/doa/bin/python tools/generate_kemar_sofamyroom_dataset.py \
  --output_root outputs/kemar_sofamyroom_dataset_smoke \
  --split train \
  --num_samples 10 \
  --overwrite
```

Pilot generation and automatic sanity check:

```bash
tools/run_kemar_sofamyroom_pilot.sh outputs/kemar_sofamyroom_pilot
```

This creates:

- `outputs/kemar_sofamyroom_pilot_train`: 720 train samples
- `outputs/kemar_sofamyroom_pilot_val`: 144 val samples
- `outputs/kemar_sofamyroom_pilot_testgrid432`: 432 structured test-grid samples
- `outputs/kemar_sofamyroom_pilot_sanity_summary.json`: metadata/audio sanity summary

Full structured test grid:

```bash
LD_LIBRARY_PATH=/home/bywang/miniconda3/envs/doa/lib:$LD_LIBRARY_PATH \
/home/bywang/miniconda3/envs/doa/bin/python tools/generate_kemar_sofamyroom_dataset.py \
  --output_root outputs/kemar_sofamyroom_dataset_test_grid \
  --split test \
  --test_grid \
  --num_samples 0 \
  --overwrite
```

The full test grid contains `6 rooms x 3 distances x 6 SNR conditions x 72 azimuths = 7776` samples.
