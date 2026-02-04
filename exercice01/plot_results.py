from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np


class StrideRow(NamedTuple):
    stride: int
    sum_value: float
    time_ms: float
    bandwidth_mb_s: float


def _read_text_lines(path: Path, encoding: str | None = None) -> list[str]:
    if encoding:
        return path.read_text(encoding=encoding).splitlines()

    for candidate in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=candidate).splitlines()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace").splitlines()


def read_stride_results(path: Path, encoding: str | None = None) -> list[StrideRow]:
    lines = _read_text_lines(path, encoding=encoding)
    if not lines:
        return []

    # Expected header: "stride , sum, time (msec), rate (MB/s)"
    rows: list[StrideRow] = []

    reader = csv.reader(lines)
    for idx, parts in enumerate(reader):
        if idx == 0:
            continue

        parts = [p.strip() for p in parts if p is not None]
        if len(parts) < 4:
            continue

        try:
            stride = int(parts[0])
            sum_value = float(parts[1])
            time_ms = float(parts[2])
            bandwidth = float(parts[3])
        except ValueError:
            continue

        if not math.isfinite(bandwidth):
            bandwidth = float("nan")
        rows.append(StrideRow(stride, sum_value, time_ms, bandwidth))

    return rows


def _as_arrays(rows: Iterable[StrideRow]):
    strides = np.array([r.stride for r in rows], dtype=int)
    times = np.array([r.time_ms for r in rows], dtype=float)
    rates = np.array([r.bandwidth_mb_s for r in rows], dtype=float)
    return strides, times, rates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot stride experiment results (time and bandwidth vs stride)."
    )
    parser.add_argument("--o0", default="results_O0.txt", help="Input results file for -O0.")
    parser.add_argument("--o2", default="results_O2.txt", help="Input results file for -O2.")
    parser.add_argument(
        "--output",
        default="stride_analysis.png",
        help="Output image path (PNG).",
    )
    parser.add_argument(
        "--encoding",
        default=None,
        help="Force input encoding (otherwise tries common encodings).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open a GUI window (useful on headless machines).",
    )
    args = parser.parse_args()

    o0_path = Path(args.o0)
    o2_path = Path(args.o2)

    o0_rows = read_stride_results(o0_path, encoding=args.encoding)
    o2_rows = read_stride_results(o2_path, encoding=args.encoding)

    if not o0_rows or not o2_rows:
        missing = []
        if not o0_rows:
            missing.append(str(o0_path))
        if not o2_rows:
            missing.append(str(o2_path))
        raise SystemExit(f"No usable rows parsed from: {', '.join(missing)}")

    strides_o0, times_o0, rates_o0 = _as_arrays(o0_rows)
    strides_o2, times_o2, rates_o2 = _as_arrays(o2_rows)

    fig, (ax_time, ax_bw) = plt.subplots(1, 2, figsize=(14, 5))

    ax_time.plot(
        strides_o0,
        times_o0,
        "o-",
        label="-O0 (no optimization)",
        linewidth=2,
        markersize=6,
    )
    ax_time.plot(
        strides_o2,
        times_o2,
        "s-",
        label="-O2 (optimized)",
        linewidth=2,
        markersize=6,
    )
    ax_time.set_xlabel("Stride")
    ax_time.set_ylabel("Execution time (ms)")
    ax_time.set_title("Execution time vs stride")
    ax_time.legend()
    ax_time.grid(True, alpha=0.3)

    ax_bw.plot(
        strides_o0,
        rates_o0,
        "o-",
        label="-O0 (no optimization)",
        linewidth=2,
        markersize=6,
    )
    ax_bw.plot(
        strides_o2,
        rates_o2,
        "s-",
        label="-O2 (optimized)",
        linewidth=2,
        markersize=6,
    )
    ax_bw.set_xlabel("Stride")
    ax_bw.set_ylabel("Bandwidth (MB/s)")
    ax_bw.set_title("Estimated bandwidth vs stride")
    ax_bw.legend()
    ax_bw.grid(True, alpha=0.3)

    out_path = Path(args.output)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")

    mean_o0 = float(np.nanmean(times_o0))
    mean_o2 = float(np.nanmean(times_o2))
    bw_o0 = float(np.nanmean(rates_o0))
    bw_o2 = float(np.nanmean(rates_o2))

    print("\n=== Summary ===")
    print(f"Avg time (ms):   -O0={mean_o0:.2f}  -O2={mean_o2:.2f}  speedup={mean_o0/mean_o2:.2f}x")
    if bw_o0 > 0 and bw_o2 > 0:
        print(
            f"Avg bandwidth:   -O0={bw_o0:.2f} MB/s  -O2={bw_o2:.2f} MB/s  change={(bw_o2/bw_o0-1)*100:.1f}%"
        )

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
