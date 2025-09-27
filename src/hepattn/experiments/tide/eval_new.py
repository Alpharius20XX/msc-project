from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binned_statistic
from tqdm import tqdm

# --- Matplotlib Configuration ---
plt.rcParams["text.usetex"] = False
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    # --- Configuration ---

    base_dir = Path("/home/xucabeeh/maxnewcopy/hepattn/logs")
    folder = "big_rff_allpos"
    file_name = "epoch=003-train_loss=1.69316_test_eval.h5"
    eval_path = base_dir / folder / "ckpts" / file_name
    output_dir = base_dir / folder / "plots"
    output_dir.mkdir(exist_ok=True)

    print(f"Processing file: {eval_path}")

    # --- Initialization ---
    pred_names = ["sudo", "sisp", "reco", "pred"]
    colors = {
        "sudo": "black",
        "sisp": "red",
        "reco": "blue",
        "pred": "purple",
    }

    trk_qtys = [
        ("sudo_pt", r"Track $p_\mathrm{T}$ (GeV)", "log", np.geomspace(1, 3.5e3, 24)),
        ("sudo_eta", r"Track $\eta$", "linear", np.linspace(-4, 4, 24)),
        ("sudo_phi", r"Track $\phi$", "linear", np.linspace(-np.pi, np.pi, 24)),
    ]

    trk_all_bins = {p: {q[0]: np.zeros(len(q[3]) - 1) for q in trk_qtys} for p in pred_names}
    trk_eff_bins = {p: {q[0]: np.zeros(len(q[3]) - 1) for q in trk_qtys} for p in pred_names}

    total_tracks = {pred_name: 0 for pred_name in pred_names}
    efficient_tracks = {pred_name: 0 for pred_name in pred_names}

    # --- Data Processing ---
    with h5py.File(eval_path) as file:
        num_samples = len(file.keys())
        for i, sample_id in tqdm(enumerate(file.keys()), total=num_samples, desc="Processing Samples"):
            if i == num_samples:
                break

            targets = file[sample_id]["targets"]
            outputs = file[sample_id]["outputs"]["final"]

            true_valid = targets["sudo_valid"][0].astype(bool)
            true_pix_valid = targets["sudo_pix_valid"][0].astype(bool)
            true_sct_valid = targets["sudo_sct_valid"][0].astype(bool)

            num_true_in_sample = np.sum(true_valid)

            preds = {}
            for pred_name in ["sudo", "sisp", "reco"]:
                preds[f"{pred_name}_valid"] = targets[f"{pred_name}_valid"][0].astype(bool)
                preds[f"{pred_name}_pix_valid"] = targets[f"{pred_name}_pix_valid"][0].astype(bool)
                preds[f"{pred_name}_sct_valid"] = targets[f"{pred_name}_sct_valid"][0].astype(bool)

            preds["pred_valid"] = (sigmoid(outputs["pred_valid"]["pred_logit"][0]) >= 0.5)
            preds["pred_pix_valid"] = (sigmoid(outputs["pred_pix_assignment"]["pred_pix_logit"][0]) >= 0.5)
            preds["pred_sct_valid"] = (sigmoid(outputs["pred_sct_assignment"]["pred_sct_logit"][0]) >= 0.5)

            for pred_name in pred_names:
                pred_valid = preds[f"{pred_name}_valid"]
                pred_pix_valid = preds[f"{pred_name}_pix_valid"]
                pred_sct_valid = preds[f"{pred_name}_sct_valid"]

                # --- Compute overlaps ---
                pix_tp = np.einsum("nc,mc->nm", true_pix_valid, pred_pix_valid)
                pix_fp = np.einsum("nc,mc->nm", true_pix_valid, ~pred_pix_valid)
                pix_fn = np.einsum("nc,mc->nm", ~true_pix_valid, pred_pix_valid)

                sct_tp = np.einsum("nc,mc->nm", true_sct_valid, pred_sct_valid)
                sct_fp = np.einsum("nc,mc->nm", true_sct_valid, ~pred_sct_valid)
                sct_fn = np.einsum("nc,mc->nm", ~true_sct_valid, pred_sct_valid)

                eps = 1e-6
                metric = "iou"  # keep eval logic from script 2
                score_threshold = 0.75

                if metric == "tmp":
                    scores = (2 * pix_tp + sct_tp) / (2 * (pix_tp + pix_fp) + sct_tp + sct_fp + eps)
                elif metric == "iou":
                    scores = (2 * pix_tp + sct_tp) / (
                        2 * (pix_tp + pix_fp + pix_fn) + sct_tp + sct_fp + sct_fn + eps
                    )
                elif metric == "dice":
                    scores = 2 * (pix_tp + sct_tp) / (
                        2 * pix_tp + pix_fp + pix_fn + 2 * sct_tp + sct_fp + sct_fn + eps
                    )

                _, pred_idx = linear_sum_assignment(scores)
                scores = scores[:, pred_idx]

                true_is_eff_mask = (scores >= score_threshold).any(-1)
                true_is_eff_for_binning = true_is_eff_mask[true_valid]

                efficient_tracks[pred_name] += np.sum(true_is_eff_mask)
                total_tracks[pred_name] += num_true_in_sample

                for qty_name, _, _, bins in trk_qtys:
                    qty = targets[qty_name][0][true_valid]
                    num_all, _, _ = binned_statistic(qty, true_is_eff_for_binning, statistic="count", bins=bins)
                    num_eff, _, _ = binned_statistic(qty, true_is_eff_for_binning, statistic="sum", bins=bins)
                    trk_all_bins[pred_name][qty_name] += num_all
                    trk_eff_bins[pred_name][qty_name] += num_eff

    # --- Print Overall Efficiency to Console ---
    print("\n" + "=" * 50)
    print(f"Overall Tracking Efficiencies for {eval_path.name}:")
    for pred_name in pred_names:
        total = total_tracks[pred_name]
        efficient = efficient_tracks[pred_name]
        if total > 0:
            overall_eff = efficient / total
            print(f"  - {pred_name.capitalize():<5}: {overall_eff:.4f} ({efficient}/{total})")
        else:
            print(f"  - {pred_name.capitalize():<5}: N/A (0 total tracks)")
    print("=" * 50 + "\n")

    # --- Save Overall Efficiency to File ---
    efficiency_filename = output_dir / f"{Path(file_name).stem}_efficiencies_new.txt"
    with open(efficiency_filename, 'w') as f:
        f.write(f"Overall Tracking Efficiencies for {eval_path.name}\n")
        f.write("=" * 50 + "\n")
        for pred_name in pred_names:
            total = total_tracks[pred_name]
            efficient = efficient_tracks[pred_name]
            if total > 0:
                overall_eff = efficient / total
                f.write(f"{pred_name.capitalize():<5}: {overall_eff:.4f} ({efficient}/{total})\n")
            else:
                f.write(f"{pred_name.capitalize():<5}: N/A (0 total tracks)\n")
    print(f"Saved efficiencies to: {efficiency_filename}")

    # --- Plotting (script 2 style) ---
    for qty_name, qty_label, scale, bins in trk_qtys:
        fig, ax = plt.subplots()
        fig.set_size_inches(8, 3)

        for pred_name in pred_names:
            with np.errstate(divide="ignore", invalid="ignore"):
                freq_e = trk_eff_bins[pred_name][qty_name] / trk_all_bins[pred_name][qty_name]
                freq_e = np.nan_to_num(freq_e)

            for bin_idx in range(len(bins) - 1):
                px = np.array([bins[bin_idx], bins[bin_idx + 1]])
                py = np.array([freq_e[bin_idx], freq_e[bin_idx]])
                ax.plot(px, py, color=colors[pred_name], linewidth=1.0)

        ax.set_xscale(scale)
        ax.grid(zorder=0, alpha=0.25, linestyle="--")
        ax.set_xlabel(qty_label)
        ax.set_ylabel("Track Efficiency")

        fig.tight_layout()
        plot_filename = f"{Path(file_name).stem}_new{qty_name}.png"
        fig.savefig(output_dir / plot_filename)
        plt.close(fig)
        print(f"Saved plot: {output_dir / plot_filename}")


if __name__ == "__main__":
    main()
