#!/usr/bin/env python3

import subprocess
import os
import glob
import matplotlib.pyplot as plt

# Configurations to test
JOINT_LOCKS = [
    "none",
    "lock_index",
    "lock_middle",
    "lock_ring",
    "lock_little",
    "lock_thumb",
    "lock_ring_little",
    "lock_middle_ring_little",
    "lock_index_middle",
    "lock_index_middle_ring"
]

# Experiment settings
NBR = 20000
OBJ = "ycb_tennis_ball"
LOG_PATH = "finger_locking_experiment_2"
os.makedirs(LOG_PATH, exist_ok=True)

def run_test(joint_lock):
    folder_name = f"test_{joint_lock}"
    cmd = [
        "python", "run_qd_grasp.py",
        "-a", "contact_me_scs",
        "-r", "shadow",
        "-nbr", str(NBR),
        "-o", OBJ,
        "-jl", joint_lock,
        "-f", folder_name,
        "-l", LOG_PATH,
        "-ll"
    ]
    print(f"\n[Running test: {joint_lock}]")
    subprocess.run(cmd)

def find_latest_run_folder(joint_lock):
    prefix = f"test_{joint_lock}"
    pattern = os.path.join(LOG_PATH, f"{prefix}[0-9]*")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None

def parse_run_infos(path):
    n_success = n_eval = None
    try:
        with open(path, "r") as f:
            for line in f:
                if "Number of successful individuals" in line:
                    n_success = int(line.split(":")[-1].strip())
                elif "Number of evaluations (progression_monitoring.n_eval)" in line:
                    n_eval = int(line.split(":")[-1].strip())
    except Exception:
        pass
    return n_success, n_eval

def read_stats(run_folder):
    path = os.path.join(run_folder, "run_infos.yaml")
    if not os.path.exists(path):
        print(f"[!] Missing run_infos.yaml in {run_folder}")
        return None, None
    return parse_run_infos(path)

def save_txt(results, path):
    with open(path, "w") as f:
        f.write("=== Grasping Results Summary ===\n")
        for name, success, total, rate in results:
            f.write(f"{name:<30} {success}/{total} successful grasps ({rate:.1f}%)\n")
    print(f"[✔] Saved summary to: {path}")

def save_plot(results, path):
    """
    Draws a line plot with labeled points from experiment results.

    Parameters:
        results: List of tuples (label, n_success, n_total, success_rate)
        path: Path to save the plot image
    """
    names = [name for name, _, _, _ in results]
    rates = [rate for _, _, _, rate in results]

    plt.figure(figsize=(12, 6))
    plt.plot(names, rates, marker='x', markersize=10, linewidth=2, color='brown', label='Success Rate')

    # Annotate each point with its value
    for i, (x, y) in enumerate(zip(names, rates)):
        plt.text(i, y + 0.3, f"{y:.2f}", ha='center', fontsize=9, color='brown')

    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Success Rate (%)")
    plt.ylim(0, 10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.title("Grasping Success Rate per Joint Lock Configuration")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"[✔] Line plot saved to: {path}")

def main():
    results = []

    for joint_lock in JOINT_LOCKS:
        run_test(joint_lock)
        run_folder = find_latest_run_folder(joint_lock)
        if run_folder:
            success, total = read_stats(run_folder)
            if success is not None and total is not None:
                rate = (success / total) * 100
                results.append((joint_lock, success, total, rate))

    print("\n=== Grasping Results Summary ===")
    for name, success, total, rate in results:
        print(f"{name:<30} {success}/{total} successful grasps ({rate:.1f}%)")

    summary_path = os.path.join(LOG_PATH, "summary.txt")
    plot_path = os.path.join(LOG_PATH, "success_plot.png")
    save_txt(results, summary_path)
    save_plot(results, plot_path)

if __name__ == "__main__":
    main()
