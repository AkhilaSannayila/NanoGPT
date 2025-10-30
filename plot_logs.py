import os
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

def try_open_file(file_path):
    """
    Tries to open a file with common PowerShell encodings.
    """
    encodings_to_try = ['utf-16-le', 'utf-8', 'latin-1']
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    print(f"Warning: Could not decode file {file_path}.")
    return None

def parse_losses_from_content(content, regex):
    """Helper to parse all losses (train or val) from log content."""
    losses = []
    for line in content.splitlines():
        match = regex.search(line)
        if match:
            try:
                losses.append(float(match.group(1)))
            except ValueError:
                pass
    return losses

def plot_all_logs():
    """
    Parses all 'training.log' files and generates plots for
    training and validation loss.
    """
    
    # Regex to find train and val loss
    train_loss_regex = re.compile(r"train loss.*?\s+(\d+\.\d+)")
    val_loss_regex = re.compile(r"val loss.*?\s+(\d+\.\d+)")
    
    experiment_dirs = glob.glob('out-g1-exp*')
    if not experiment_dirs:
        print("Error: No 'out-g1-exp...' directories found.")
        return

    print(f"Parsing logs from {len(experiment_dirs)} experiment directories...")

    all_train_losses = {}
    all_val_losses = {}

    for dir_name in experiment_dirs:
        log_file_path = os.path.join(dir_name, 'training.log')
        if not os.path.exists(log_file_path):
            continue

        content = try_open_file(log_file_path)
        if not content:
            continue

        # Extract a short, readable name for the legend
        # e.g., "exp1-nh4-ne128-b8-mi1000-d0.1"
        try:
            parts = dir_name.split('-')
            short_name = f"{parts[2]}-{parts[4]}-{parts[5]}-{parts[6]}-{parts[7]}-{parts[8]}"
        except Exception:
            short_name = dir_name # Fallback
            
        train_losses = parse_losses_from_content(content, train_loss_regex)
        val_losses = parse_losses_from_content(content, val_loss_regex)
        
        if train_losses:
            all_train_losses[short_name] = train_losses
        if val_losses:
            all_val_losses[short_name] = val_losses

    # --- Plot Training Loss ---
    print("Generating training_loss_plot.png...")
    plt.figure(figsize=(20, 12))
    for name, losses in all_train_losses.items():
        # Only plot the first 100 steps to avoid clutter
        steps = np.arange(len(losses)) * 100 
        plt.plot(steps, losses, label=name, alpha=0.7)
    
    plt.title('Training Loss per 100 Steps (All 32 Experiments)', fontsize=20)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Training Loss', fontsize=14)
    plt.grid(True)
    plt.ylim(0, 10) # Set Y-axis limit to see details (adjust if needed)
    # Optional: Add a legend. Comment out if it's too cluttered.
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig('training_loss_plot.png')
    print("Saved training_loss_plot.png")

    # --- Plot Validation Loss ---
    print("Generating validation_loss_plot.png...")
    plt.figure(figsize=(20, 12))
    for name, losses in all_val_losses.items():
        steps = np.arange(len(losses)) * 100
        plt.plot(steps, losses, label=name, alpha=0.7)
    
    plt.title('Validation Loss per 100 Steps (All 32 Experiments)', fontsize=20)
    plt.xlabel('Training Step', fontsize=14)
    plt.ylabel('Validation Loss', fontsize=14)
    plt.grid(True)
    plt.ylim(2.0, 6.0) # Zoom in on the important part of the y-axis
    
    # Add legend for the Top 5 Models
    top_5_names = [
        "exp30-nh8-ne256-b16-mi1000-d0.2",
        "exp32-nh8-ne256-b16-mi2000-d0.2",
        "exp14-nh4-ne256-b16-mi1000-d0.2",
        "exp16-nh4-ne256-b16-mi2000-d0.2",
        "exp28-nh8-ne256-b8-mi2000-d0.2"
    ]
    
    for name, losses in all_val_losses.items():
        if name in top_5_names:
            steps = np.arange(len(losses)) * 100
            plt.plot(steps, losses, label=f"TOP 5: {name}", linewidth=3)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='medium')
    plt.tight_layout()
    plt.savefig('validation_loss_plot.png')
    print("Saved validation_loss_plot.png")
    print("Done.")

if __name__ == '__main__':
    plot_all_logs()