#!/usr/bin/env python3
"""Build paper-ready diffusefg result tables (CSV + LaTeX)."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import Config
from models.binaural_doa_net import build_model
from tools.benchmark_models import count_parameters, measure_flops_thop, build_dummy_input


MODELS = [
    {
        "label": "v7\\_dualcue\\_fbfocus",
        "config": "configs/train_kemar_v7_dualcue_diffusefg_fbfocus_g5.yaml",
        "overall": "outputs/grouped_eval_runs/v7_dualcue_diffusefg_fbfocus_g5/overall.json",
        "by_snr": "outputs/grouped_eval_runs/v7_dualcue_diffusefg_fbfocus_g5/by_snr.csv",
    },
    {
        "label": "v7\\_dualcue\\_liteenc\\_v1",
        "config": "configs/train_kemar_v7_dualcue_liteenc_v1_diffusefg_g5.yaml",
        "overall": "outputs/grouped_eval_runs/v7_dualcue_liteenc_v1_diffusefg_g5/overall.json",
        "by_snr": "outputs/grouped_eval_runs/v7_dualcue_liteenc_v1_diffusefg_g5/by_snr.csv",
    },
    {
        "label": "SDEL",
        "config": "configs/train_kemar_sdel_doa_cls_fbaux_diffusefg_retry_g5.yaml",
        "overall": "outputs/eval_kemar_sdel_doa_cls_fbaux_diffusefg_retry_g5_diffuse/overall.json",
        "by_snr": "outputs/eval_kemar_sdel_doa_cls_fbaux_diffusefg_retry_g5_diffuse/by_snr.csv",
    },
    {
        "label": "FN-SSL",
        "config": "outputs/logs_kemar_fnssl_diffusefg_g5_stable/resolved_config.yaml",
        "overall": "outputs/grouped_eval_runs/fnssl_diffusefg_g5_stable/overall.json",
        "by_snr": "outputs/grouped_eval_runs/fnssl_diffusefg_g5_stable/by_snr.csv",
    },
    {
        "label": "DP-RTF",
        "config": "configs/train_kemar_dprtf_doa_cls_diffusefg_g5.yaml",
        "overall": "outputs/grouped_eval_runs/dprtf_diffusefg_g5/overall.json",
        "by_snr": "outputs/grouped_eval_runs/dprtf_diffusefg_g5/by_snr.csv",
    },
    {
        "label": "BiL",
        "config": "configs/train_kemar_bilstyle_gccphat_crn72_diffusefg_g7.yaml",
        "overall": "outputs/grouped_eval_runs/bilstyle_gccphat_crn72_diffusefg_retry_g6/overall.json",
        "by_snr": "outputs/grouped_eval_runs/bilstyle_gccphat_crn72_diffusefg_retry_g6/by_snr.csv",
    },
    {
        "label": "FAViT",
        "config": "configs/train_kemar_favitstyle_ildipd_diffusefg_g5_batch8.yaml",
        "overall": "outputs/grouped_eval_runs/favitstyle_ildipd_diffusefg_g5_batch8/overall.json",
        "by_snr": "outputs/grouped_eval_runs/favitstyle_ildipd_diffusefg_g5_batch8/by_snr.csv",
    },
]

SNR_ORDER = ["clean", "10", "5", "0", "-5", "-10"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def fmt4(x: float) -> str:
    return f"{x:.4f}"


def fmt2(x: float) -> str:
    return f"{x:.2f}"


def fmt_pct2(x: float) -> str:
    return f"{x * 100:.2f}"


def escape_tex(s: str) -> str:
    return s.replace("_", "\\_")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_complexity(config_path: Path) -> tuple[int, float | None]:
    cfg = Config.from_yaml(str(config_path))
    cfg.train.device = "cpu"
    model = build_model(cfg)
    dummy = build_dummy_input(cfg, batch_size=1, device="cpu")
    params = count_parameters(model)["total"]
    flops = measure_flops_thop(model, dummy, "cpu")
    flops_g = None if flops is None else flops / 1e9
    return params, flops_g


def build_main_table() -> list[dict]:
    rows = []
    for spec in MODELS:
        overall = read_json(ROOT / spec["overall"])
        rows.append(
            {
                "model": spec["label"],
                "acc": fmt_pct2(overall["accuracy"]),
                "mae_deg": fmt4(overall["mae"]),
                "acc_at_5": fmt_pct2(overall["acc_at_5"]),
                "acc_at_10": fmt_pct2(overall["acc_at_10"]),
            }
        )
    return rows


def build_by_snr_table() -> list[dict]:
    rows = []
    for spec in MODELS:
        data = {row["snr"]: row for row in read_csv(ROOT / spec["by_snr"])}
        row = {"model": spec["label"]}
        for snr in SNR_ORDER:
            item = data[snr]
            row[f"{snr}_acc"] = fmt_pct2(float(item["accuracy"]))
            row[f"{snr}_mae_deg"] = fmt4(float(item["mae"]))
        rows.append(row)
    return rows


def build_complexity_table() -> list[dict]:
    rows = []
    for spec in MODELS:
        params, flops_g = compute_complexity(ROOT / spec["config"])
        rows.append(
            {
                "model": spec["label"],
                "params_m": f"{params / 1e6:.3f}",
                "flops_g": "" if flops_g is None else f"{flops_g:.3f}",
            }
        )
    return rows


def write_main_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Main results on the diffusefg test set.}",
        "\\label{tab:diffusefg_main_results}",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Model & Acc (\\%) & MAE ($^\\circ$) & Acc@5 (\\%) & Acc@10 (\\%) \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(
            f"{escape_tex(row['model'])} & {row['acc']} & {row['mae_deg']} & "
            f"{row['acc_at_5']} & {row['acc_at_10']} \\\\"
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_complexity_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Model complexity comparison on the diffusefg benchmark. FLOPs are estimated on a single sample input.}",
        "\\label{tab:diffusefg_complexity}",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Model & Params (M) & FLOPs (G) \\\\",
        "\\hline",
    ]
    for row in rows:
        flops = row["flops_g"] if row["flops_g"] else "N/A"
        lines.append(f"{escape_tex(row['model'])} & {row['params_m']} & {flops} \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_by_snr_tex(path: Path, rows: list[dict]) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Per-SNR results on the diffusefg test set. Each cell reports Acc (\\%) / MAE ($^\\circ$).}",
        "\\label{tab:diffusefg_by_snr}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "Model & clean & 10 dB & 5 dB & 0 dB & -5 dB & -10 dB \\\\",
        "\\hline",
    ]
    for row in rows:
        vals = []
        for snr in SNR_ORDER:
            vals.append(f"{row[f'{snr}_acc']} / {row[f'{snr}_mae_deg']}")
        lines.append(f"{escape_tex(row['model'])} & " + " & ".join(vals) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table*}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_rows = build_main_table()
    by_snr_rows = build_by_snr_table()
    complexity_rows = build_complexity_table()

    write_csv(
        out_dir / "paper_diffusefg_main_results.csv",
        main_rows,
        ["model", "acc", "mae_deg", "acc_at_5", "acc_at_10"],
    )
    write_csv(
        out_dir / "paper_diffusefg_by_snr.csv",
        by_snr_rows,
        ["model"]
        + [f"{snr}_acc" for snr in SNR_ORDER]
        + [f"{snr}_mae_deg" for snr in SNR_ORDER],
    )
    write_csv(
        out_dir / "paper_diffusefg_complexity.csv",
        complexity_rows,
        ["model", "params_m", "flops_g"],
    )

    write_main_tex(out_dir / "paper_diffusefg_main_results.tex", main_rows)
    write_by_snr_tex(out_dir / "paper_diffusefg_by_snr.tex", by_snr_rows)
    write_complexity_tex(out_dir / "paper_diffusefg_complexity.tex", complexity_rows)

    print("Saved:")
    print(out_dir / "paper_diffusefg_main_results.csv")
    print(out_dir / "paper_diffusefg_main_results.tex")
    print(out_dir / "paper_diffusefg_by_snr.csv")
    print(out_dir / "paper_diffusefg_by_snr.tex")
    print(out_dir / "paper_diffusefg_complexity.csv")
    print(out_dir / "paper_diffusefg_complexity.tex")


if __name__ == "__main__":
    main()
