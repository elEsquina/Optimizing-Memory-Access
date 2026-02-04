from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class BlockRow:
    block_size: int
    time_ms: float
    bandwidth_mb_s: float
    label: str


def _read_text_lines(path: Path, encoding: str | None = None) -> list[str]:
    if encoding:
        return path.read_text(encoding=encoding).splitlines()

    for candidate in ("utf-8", "utf-8-sig", "utf-16", "latin-1"):
        try:
            return path.read_text(encoding=candidate).splitlines()
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="replace").splitlines()


def _infer_matrix_size(lines: list[str]) -> int | None:
    # Expected: "Matrix size: 512 x 512"
    for line in lines[:10]:
        m = re.search(r"Matrix\s+size:\s*(\d+)\s*x\s*(\d+)", line, flags=re.IGNORECASE)
        if m and m.group(1) == m.group(2):
            return int(m.group(1))
    return None


def parse_results(path: Path, encoding: str | None = None) -> tuple[list[BlockRow], int | None]:
    lines = _read_text_lines(path, encoding=encoding)
    if not lines:
        return [], None

    n = _infer_matrix_size(lines)
    no_block_x = n if n is not None else 512

    rows: list[BlockRow] = []
    for raw in lines:
        line = raw.strip()
        if not line or "," not in line:
            continue
        if line.lower().startswith("block size") or line.lower().startswith("version"):
            continue

        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue

        label = parts[0]
        if "standard" in label.lower():
            block_size = no_block_x
            label = "No blocking"
        else:
            try:
                block_size = int(label)
                label = f"Block {block_size}"
            except ValueError:
                continue

        try:
            time_ms = float(parts[1])
            bandwidth = float(parts[2])
        except ValueError:
            continue

        if not math.isfinite(time_ms) or time_ms <= 0:
            continue
        if not math.isfinite(bandwidth):
            bandwidth = float("nan")

        rows.append(BlockRow(block_size=block_size, time_ms=time_ms, bandwidth_mb_s=bandwidth, label=label))

    rows.sort(key=lambda r: r.block_size)
    return rows, n


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot blocked matrix multiplication results (time and bandwidth vs block size)."
    )
    parser.add_argument("--input", default="mxm_bloc_results.txt", help="Input results file.")
    parser.add_argument("--output", default="block_size_analysis.png", help="Output image path (PNG).")
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

    in_path = Path(args.input)
    rows, n = parse_results(in_path, encoding=args.encoding)
    if not rows:
        raise SystemExit(f"No usable rows parsed from: {in_path}")

    block_sizes = np.array([r.block_size for r in rows], dtype=int)
    times = np.array([r.time_ms for r in rows], dtype=float)
    bandwidths = np.array([r.bandwidth_mb_s for r in rows], dtype=float)

    # Use the "No blocking" row as baseline when available; otherwise fall back to the last entry.
    baseline_mask = np.array([r.label == "No blocking" for r in rows], dtype=bool)
    if np.any(baseline_mask):
        baseline_time = float(times[baseline_mask][0])
    else:
        baseline_time = float(times[-1])

    speedups = baseline_time / times

    # Plot
    fig, (ax_time, ax_bw) = plt.subplots(1, 2, figsize=(14, 6))

    label_no_block = f"{n} (= No blocking)" if n else "No blocking"

    ax_time.plot(block_sizes, times, "o-", linewidth=2, markersize=8, color="#e74c3c")
    ax_time.set_xlabel(f"Block size ({label_no_block})")
    ax_time.set_ylabel("Execution time (ms)")
    ax_time.set_title("Execution time vs block size")
    ax_time.grid(True, alpha=0.3, linestyle="--")
    ax_time.set_xscale("log", base=2)
    ax_time.set_xticks(block_sizes)
    ax_time.set_xticklabels(
        ["No block" if (n and bs == n) or (not n and bs == 512) else str(bs) for bs in block_sizes],
        rotation=45,
    )

    best_time_idx = int(np.argmin(times))
    ax_time.plot(block_sizes[best_time_idx], times[best_time_idx], "g*", markersize=18)
    ax_time.legend([f"Best: {times[best_time_idx]:.2f} ms"], loc="best")

    ax_bw.plot(block_sizes, bandwidths, "s-", linewidth=2, markersize=8, color="#3498db")
    ax_bw.set_xlabel(f"Block size ({label_no_block})")
    ax_bw.set_ylabel("Bandwidth (MB/s)")
    ax_bw.set_title("Estimated bandwidth vs block size")
    ax_bw.grid(True, alpha=0.3, linestyle="--")
    ax_bw.set_xscale("log", base=2)
    ax_bw.set_xticks(block_sizes)
    ax_bw.set_xticklabels(
        ["No block" if (n and bs == n) or (not n and bs == 512) else str(bs) for bs in block_sizes],
        rotation=45,
    )

    best_bw_idx = int(np.nanargmax(bandwidths))
    ax_bw.plot(block_sizes[best_bw_idx], bandwidths[best_bw_idx], "g*", markersize=18)
    ax_bw.legend([f"Best: {bandwidths[best_bw_idx]:.2f} MB/s"], loc="best")

    out_path = Path(args.output)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {out_path}")

    # Print a compact table
    print("\n=== Summary (speedup vs no blocking) ===")
    for r, sp in zip(rows, speedups):
        name = "No blocking" if r.label == "No blocking" else f"Block {r.block_size:>4d}"
        print(f"{name}: time={r.time_ms:8.2f} ms  bw={r.bandwidth_mb_s:10.2f} MB/s  speedup={sp:5.2f}x")

    # Working-set estimate (3 blocks of doubles: A, B, C)
    chosen_bs = int(block_sizes[best_time_idx])
    bytes_working_set = 3 * chosen_bs * chosen_bs * 8
    print("\n=== Working-set estimate (best time) ===")
    print(f"Block size: {chosen_bs}")
    print(f"Approx working set: {bytes_working_set:,} bytes ({bytes_working_set/1024:.1f} KiB)")

    if not args.no_show:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
