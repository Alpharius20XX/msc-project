import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import binned_statistic
from tqdm import tqdm
import glob
import logging

# --- Matplotlib Configuration ---
# Setting plot parameters for better aesthetics.
plt.rcParams["text.usetex"] = False  # Set to True if you have a LaTeX installation
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10
plt.rcParams["figure.constrained_layout.use"] = True


def sigmoid(x):
    """
    Computes the sigmoid function. A clipping range is used for numerical stability.
    """
    return 1 / (1 + np.exp(-np.clip(x, -10, 10)))


def main():
    """
    Main function to parse arguments, process H5 files, and generate plots.
    """
    # --- Argument Parsing ---
    # Set up command-line argument parsing to get the output directory.
    parser = argparse.ArgumentParser(description="Plot tracking efficiency from evaluation H5 files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory containing the '/ckpts/*.test_eval.h5' files and where plots will be saved."
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Optional file to save the console output."
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    # --- Logging Setup ---
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, mode='w')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)


    # --- File Searching ---
    # Search for all evaluation files within the specified directory.
    search_path = str(output_dir / "ckpts/*test_eval.h5")
    eval_paths = glob.glob(search_path)

    if not eval_paths:
        logging.error(f"Error: No matching files found for pattern: {search_path}")
        return

    logging.info(f"Found {len(eval_paths)} files to process.")

    for eval_path in eval_paths:
        logging.info(f"Processing: {eval_path}")

        # --- Initialization ---
        # Define prediction types and corresponding plot colors.
        pred_names = ["sudo", "sisp", "reco", "pred"]
        colors = {
            "sudo": "black",
            "sisp": "red",
            "reco": "blue",
            "pred": "purple",
        }

        # Define track quantities to be plotted, including labels, scales, and binning.
        trk_qtys = [
            ("sudo_pt", r"Track $p_\mathrm{T}$ (GeV)", "log", np.geomspace(1, 3.5e3, 24)),
            ("sudo_eta", r"Track $\eta$", "linear", np.linspace(-4, 4, 24)),
            ("sudo_phi", r"Track $\phi$", "linear", np.linspace(-np.pi, np.pi, 24)),
        ]

        # Initialize dictionaries to store binned statistics.
        trk_all_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in trk_qtys} for pred_name in pred_names}
        trk_eff_bins = {pred_name: {qty: np.zeros(len(bins) - 1) for qty, _, _, bins in trk_qtys} for pred_name in pred_names}
        
        # Initialize dictionary to store overall efficiency counts.
        overall_counts = {pred_name: {"eff": 0, "all": 0} for pred_name in pred_names}


        # --- Data Processing ---
        # Open the H5 file and loop through samples.
        with h5py.File(eval_path) as file:
            # Using tqdm for a progress bar. Limited to 100 samples for quick testing.
            num_samples = len(file.keys()) #min(100, len(file.keys()))
            for i, sample_id in tqdm(enumerate(file.keys()), total=num_samples, desc="Processing Samples"):
                if i == num_samples:
                    break

                targets = file[sample_id]["targets"]
                outputs = file[sample_id]["outputs"]["final"]

                true_valid = targets["sudo_valid"][0]
                true_pix_valid = targets["sudo_pix_valid"][0]
                true_sct_valid = targets["sudo_sct_valid"][0]

                preds = {}
                for pred_name in ["sudo", "sisp", "reco"]:
                    preds[f"{pred_name}_valid"] = targets[f"{pred_name}_valid"][0]
                    preds[f"{pred_name}_pix_valid"] = targets[f"{pred_name}_pix_valid"][0]
                    preds[f"{pred_name}_sct_valid"] = targets[f"{pred_name}_sct_valid"][0]

                # Calculate validity from model logits.
                preds["pred_valid"] = sigmoid(outputs["pred_valid"]["pred_logit"][0]) >= 0.5
                preds["pred_pix_valid"] = sigmoid(outputs["pred_pix_assignment"]["pred_pix_logit"][0]) >= 0.5
                preds["pred_sct_valid"] = sigmoid(outputs["pred_sct_assignment"]["pred_sct_logit"][0]) >= 0.5

                for pred_name in pred_names:
                    pred_valid = preds[f"{pred_name}_valid"]
                    pred_pix_valid = preds[f"{pred_name}_pix_valid"]
                    pred_sct_valid = preds[f"{pred_name}_sct_valid"]

                    pix_tp = np.einsum("nc,mc->nm", true_pix_valid.astype(int), pred_pix_valid.astype(int))
                    sct_tp = np.einsum("nc,mc->nm", true_sct_valid.astype(int), pred_sct_valid.astype(int))

                    # Calculate Intersection over Union (IoU) score matrix.
                    eps = 1e-6
                    denominator = (
                        np.sum(true_pix_valid, axis=1, keepdims=True) + np.sum(pred_pix_valid, axis=1) - pix_tp +
                        np.sum(true_sct_valid, axis=1, keepdims=True) + np.sum(pred_sct_valid, axis=1) - sct_tp +
                        eps
                    )
                    scores = (pix_tp + sct_tp) / denominator

                    # Use the Hungarian algorithm for optimal assignment between true and predicted tracks.
                    true_idx, pred_idx = linear_sum_assignment(-scores) # Use negative for max weight matching
                    
                    matched_scores = scores[true_idx, pred_idx]
                    
                    # Create a mask for efficient tracks based on the score threshold.
                    true_is_eff = np.zeros_like(true_valid, dtype=bool)
                    score_threshold = 0.75
                    efficient_matches = matched_scores >= score_threshold
                    true_is_eff[true_idx[efficient_matches]] = True
                    
                    # Update overall efficiency counts for valid tracks.
                    overall_counts[pred_name]["all"] += np.sum(true_valid)
                    overall_counts[pred_name]["eff"] += np.sum(true_is_eff[true_valid])

                    # Bin the data for each quantity.
                    for qty_name, _, _, bins in trk_qtys:
                        qty = targets[qty_name][0][true_valid]
                        is_eff_for_qty = true_is_eff[true_valid]

                        num_all, _, _ = binned_statistic(qty, is_eff_for_qty, statistic="count", bins=bins)
                        num_eff, _, _ = binned_statistic(qty, is_eff_for_qty, statistic="sum", bins=bins)

                        trk_all_bins[pred_name][qty_name] += num_all
                        trk_eff_bins[pred_name][qty_name] += num_eff

        # --- Overall Efficiency Calculation & Output ---
        logging.info("\n" + "="*40)
        logging.info(f"Overall Efficiency for: {Path(eval_path).name}")
        logging.info("="*40)
        # Calculate and print the overall efficiency specifically for 'pred'
        pred_name = "pred"
        total_all = overall_counts[pred_name]["all"]
        total_eff = overall_counts[pred_name]["eff"]
        if total_all > 0:
            efficiency = total_eff / total_all
            logging.info(f"Overall 'Pred' Efficiency: {efficiency:.4f} ({total_eff} / {total_all})")
        else:
            logging.info(f"Overall 'Pred' Efficiency: N/A (0 total tracks)")
        logging.info("="*40 + "\n")


        # --- Plotting ---
        # Generate a plot for each track quantity.
        for qty_name, qty_label, scale, bins in trk_qtys:
            fig, ax = plt.subplots(figsize=(8, 4))

            for pred_name in pred_names:
                # Avoid division by zero.
                with np.errstate(divide='ignore', invalid='ignore'):
                    freq_e = trk_eff_bins[pred_name][qty_name] / trk_all_bins[pred_name][qty_name]
                    freq_e = np.nan_to_num(freq_e) # Replace NaN with 0

                # Plot data as a step plot.
                ax.step(bins, np.append(freq_e, freq_e[-1]), where='post', color=colors[pred_name], label=pred_name.capitalize())

            ax.set_xscale(scale)
            ax.set_ylim(0, 1.05)
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.set_xlabel(qty_label)
            ax.set_ylabel("Track Efficiency")
            ax.legend(title="Prediction Type")
            ax.set_title(f"Tracking Efficiency vs. {qty_label.split('$')[1]}")
            
            fig.tight_layout()
            
            # Construct a unique filename for each plot based on the input file.
            plot_filename = f"{Path(eval_path).stem}_{qty_name}.png"
            fig.savefig(output_dir / plot_filename)
            plt.close(fig)
            logging.info(f"Saved plot: {output_dir / plot_filename}")


if __name__ == "__main__":
    main()

