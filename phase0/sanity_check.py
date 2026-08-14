"""
sanity_check.py

Phase 0 telemetry sanity check for the Wattson project.

What this does:
1. Launches burn.exe (a CUDA load generator) in a background process.
2. Polls `nvidia-smi` power draw as fast as possible in the foreground,
   with a timestamp on every sample.
3. Once the burn finishes, plots power-vs-time so you can see:
   - your GPU's actual power sampling rate (how often the value changes)
   - idle baseline power
   - lag between kernel start and power ramping up
   - lag between kernel end and power dropping back to idle

Usage:
    python sanity_check.py --seconds 15 --intensity 8

Requires:
    pip install matplotlib
    burn.exe compiled and in the same directory (see build instructions below)

Build burn.exe (run in this directory, from a Developer/CUDA-enabled shell):
    nvcc -O3 burn.cu -o burn.exe
"""

import argparse
import subprocess
import threading
import time
import csv
import sys
import os

def poll_power(stop_event, samples, interval_s=0.05):
    """
    Repeatedly call `nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits`
    and record (timestamp, watts) until stop_event is set.

    interval_s is the requested polling interval. Actual achieved interval
    will be logged too, since nvidia-smi subprocess overhead may dominate
    at high polling rates -- that overhead is itself useful data for Phase 0.
    """
    query = [
        "nvidia-smi",
        "--query-gpu=power.draw",
        "--format=csv,noheader,nounits",
    ]
    while not stop_event.is_set():
        t0 = time.time()
        try:
            out = subprocess.check_output(query, stderr=subprocess.DEVNULL)
            watts = float(out.decode().strip())
        except Exception as e:
            watts = float("nan")
        t1 = time.time()
        # timestamp = midpoint of the query call
        samples.append((t0, watts, t1 - t0))
        # sleep to approximate requested interval, accounting for call cost
        sleep_left = interval_s - (t1 - t0)
        if sleep_left > 0:
            time.sleep(sleep_left)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=15.0,
                         help="How long the burn kernel should run")
    parser.add_argument("--intensity", type=int, default=8,
                         help="Load intensity 1-10")
    parser.add_argument("--poll-interval", type=float, default=0.05,
                         help="Requested seconds between nvidia-smi polls")
    parser.add_argument("--idle-pad", type=float, default=5.0,
                         help="Seconds of idle polling before/after the burn, "
                              "to establish baseline power")
    parser.add_argument("--out-csv", type=str, default="power_trace.csv")
    parser.add_argument("--out-png", type=str, default="power_trace.png")
    args = parser.parse_args()

    burn_exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "burn.exe")
    if not os.path.exists(burn_exe):
        print(f"ERROR: {burn_exe} not found. Build it first with:")
        print("    nvcc -O3 burn.cu -o burn.exe")
        sys.exit(1)

    samples = []
    stop_event = threading.Event()
    poll_thread = threading.Thread(
        target=poll_power, args=(stop_event, samples, args.poll_interval)
    )

    print(f"Polling idle baseline for {args.idle_pad}s...")
    poll_thread.start()
    time.sleep(args.idle_pad)

    print(f"Launching burn.exe for {args.seconds}s at intensity {args.intensity}...")
    proc = subprocess.Popen(
        [burn_exe, str(args.seconds), str(args.intensity)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    stdout, stderr = proc.communicate()
    print(stdout)
    if stderr.strip():
        print("burn.exe stderr:", stderr)

    print(f"Polling idle cooldown for {args.idle_pad}s...")
    time.sleep(args.idle_pad)

    stop_event.set()
    poll_thread.join()

    # Write raw samples to CSV
    t_start = samples[0][0]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["t_rel_seconds", "watts", "query_latency_seconds"])
        for t, watts, qlat in samples:
            writer.writerow([t - t_start, watts, qlat])
    print(f"Wrote {len(samples)} samples to {args.out_csv}")

    # Basic stats on sampling behavior
    intervals = [samples[i][0] - samples[i-1][0] for i in range(1, len(samples))]
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        print(f"Average achieved poll interval: {avg_interval*1000:.1f} ms "
              f"(requested {args.poll_interval*1000:.1f} ms)")
        avg_qlat = sum(s[2] for s in samples) / len(samples)
        print(f"Average nvidia-smi query call latency: {avg_qlat*1000:.1f} ms")

    # Plot
    try:
        import matplotlib.pyplot as plt
        times = [s[0] - t_start for s in samples]
        watts = [s[1] for s in samples]
        plt.figure(figsize=(10, 5))
        plt.plot(times, watts, marker=".", linewidth=1)
        plt.axvline(args.idle_pad, color="green", linestyle="--", label="burn start (requested)")
        plt.axvline(args.idle_pad + args.seconds, color="red", linestyle="--", label="burn end (requested)")
        plt.xlabel("Time (s)")
        plt.ylabel("Power draw (W)")
        plt.title("GPU power trace -- Phase 0 sanity check")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out_png, dpi=150)
        print(f"Saved plot to {args.out_png}")
    except ImportError:
        print("matplotlib not installed -- skipping plot. "
              "Run `pip install matplotlib` and re-run to generate power_trace.png")


if __name__ == "__main__":
    main()