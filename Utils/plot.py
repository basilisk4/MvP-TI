import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def centered_moving_average(x, window):
    x = np.asarray(x)
    if window % 2 == 0:
        raise ValueError("Window size must be odd for centered moving average")
    if len(x) < window:
        return x, np.arange(len(x))

    half = window // 2
    y = np.convolve(x, np.ones(window) / window, mode="valid")
    idx = np.arange(half, len(x) - half)
    return y, idx


def plot(LOG_FILE):
    # ----------------------------------------------------
    # Regex patterns
    # ----------------------------------------------------
    epoch_end_pattern = re.compile(
        r"Epoch:\s*\[(\d+)\]\[1509/1509\].*?"
        r"loss_ce:\s*[\d.]+\s*\(([\d.]+)\).*?"
        r"loss_pose_perjoint:\s*[\d.]+\s*\(([\d.]+)\).*?"
        r"loss_pose_perprojection:\s*[\d.]+\s*\(([\d.]+)\)",
        re.DOTALL
    )

    test_end_pattern = re.compile(
        r"Test:\s*\[760/760\].*?"
        r"loss_ce:\s*[\d.]+\s*\(([\d.]+)\).*?"
        r"loss_pose_perjoint:\s*[\d.]+\s*\(([\d.]+)\).*?"
        r"loss_pose_perprojection:\s*[\d.]+\s*\(([\d.]+)\)",
        re.DOTALL
    )

    ap_pattern = re.compile(
        r"\|\s*AP\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
    )

    recall_pattern = re.compile(
        r"\|\s*Recall\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)"
    )

    mpjpe_pattern = re.compile(r"MPJPE:\s*([\d.]+)mm")

    thresholds = [25, 50, 75, 100, 125, 150]

    # ----------------------------------------------------
    # Containers
    # ----------------------------------------------------
    epochs = []

    train_loss, train_ce, train_pj, train_pp = [], [], [], []
    val_loss, val_ce, val_pj, val_pp = [], [], [], []

    ap_all, recall_all, mpjpe_all = [], [], []

    # ----------------------------------------------------
    # Read log
    # ----------------------------------------------------
    with open(LOG_FILE, "r") as f:
        log = f.read()

    epoch_matches = list(epoch_end_pattern.finditer(log))

    for match in epoch_matches:
        epoch, ce, pj, pp = match.groups()
        start = match.end()

        test_match = test_end_pattern.search(log, start)
        ap_match = ap_pattern.search(log, start)
        recall_match = recall_pattern.search(log, start)
        mpjpe_match = mpjpe_pattern.search(log, start)

        if not (test_match and ap_match and recall_match and mpjpe_match):
            continue

        # Training
        epochs.append(int(epoch))
        train_ce.append(float(ce))
        train_pj.append(float(pj))
        train_pp.append(float(pp))
        train_loss.append(2*float(ce)+5*float(pj)+5*float(pp))

        # Validation
        v_ce, v_pj, v_pp = test_match.groups()
        val_ce.append(float(v_ce))
        val_pj.append(float(v_pj))
        val_pp.append(float(v_pp))
        val_loss.append(2*float(v_ce)+5*float(v_pj)+5*float(v_pp))

        # Metrics
        ap_all.append(list(map(float, ap_match.groups())))
        recall_all.append(list(map(float, recall_match.groups())))
        mpjpe_all.append(float(mpjpe_match.group(1)))

    WINDOW = 9
    val_loss_smooth, idx = centered_moving_average(val_loss, WINDOW)
    epochs_smooth = [epochs[i] for i in idx]
    WINDOW = 5
    val_loss_smooth, idx = centered_moving_average(val_loss_smooth, WINDOW)
    epochs_smooth = [epochs_smooth[i] for i in idx]

    # ----------------------------------------------------
    # Plot 1: Train vs Val losses
    # ----------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_loss, "-o", label="Training loss")
    plt.plot(epochs, val_loss, "--o", label="Validation Loss")
    #plt.plot(epochs_smooth,val_loss_smooth,"-o", label=f"Validation loss (smoothed, w={WINDOW})")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.title("Losses")
    plt.legend()
    plt.minorticks_on()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, mpjpe_all, marker="o")
    plt.title("MPJPE")
    plt.xlabel("Epoch")
    plt.ylabel("MPJPE (mm)")
    plt.minorticks_on()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))

    plt.subplot(2, 1, 1)
    for i, t in enumerate(thresholds):
        plt.plot(epochs, [ap[i] for ap in ap_all], label=f"AP@{t}mm")
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.title("Average Precision")
    plt.legend()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    plt.subplot(2, 1, 2)
    for i, t in enumerate(thresholds):
        plt.plot(epochs, [r[i] for r in recall_all], label=f"Recall@{t}mm")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall")
    plt.legend()
    plt.grid(True, which="major")
    plt.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a file")
    parser.add_argument("log", help="Path to the log file")
    args = parser.parse_args()
    plot(args.log)  