import os
import glob
import re

def try_open_file(file_path):
    """
    Tries to open a file with common PowerShell encodings (utf-16-le, utf-8, latin-1).
    Returns the file content as a string, or None if all fail.
    """
    encodings_to_try = ['utf-16-le', 'utf-8', 'latin-1']
    
    for enc in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue # Try the next encoding
        except Exception as e:
            print(f"Error reading {file_path} with {enc}: {e}")
            return None # Other error
            
    print(f"Warning: Could not decode file {file_path} with any known encoding.")
    return None

def parse_logs():
    """
    Parses all 'training.log' files in 'out-g1-exp...' directories,
    finds the minimum validation loss for each, and prints the top 5.
    """
    
    # Regex to find validation loss. Handles junk/color codes.
    val_loss_regex = re.compile(r"val loss.*?\s+(\d+\.\d+)")
    
    results = []
    experiment_dirs = glob.glob('out-g1-exp*')

    if not experiment_dirs:
        print("Error: No 'out-g1-exp...' directories found.")
        print("Please run this script from your 'nanoGPT' directory.")
        return

    print(f"Parsing logs from {len(experiment_dirs)} experiment directories...")

    for dir_name in experiment_dirs:
        log_file_path = os.path.join(dir_name, 'training.log')
        
        if not os.path.exists(log_file_path):
            print(f"Warning: '{log_file_path}' not found. Skipping.")
            continue

        content = try_open_file(log_file_path)
        
        if content:
            # Find all validation loss values in the file
            losses = [float(loss) for loss in val_loss_regex.findall(content)]
            
            if losses:
                # Find the minimum validation loss
                min_val_loss = min(losses)
                results.append((dir_name, min_val_loss))
            else:
                print(f"Warning: No 'val loss' numbers found in '{log_file_path}'.")
        else:
            print(f"Warning: Could not read content from '{log_file_path}'.")

    # --- Sort and Print Results ---
    if not results:
        print("No results found. Check log files manually for 'val loss'.")
        return

    # Sort the results by the validation loss (lowest first)
    results.sort(key=lambda x: x[1])

    print("\n" + "---" * 15)
    print("--- 🏆 Top 5 Best Models (by Validation Loss) ---")
    print("---" * 15)
    for i, (dir_name, loss) in enumerate(results[:5]):
        print(f"\n#{i+1}: {dir_name}\n     Min Val Loss: {loss:.4f}")
        
    print("\n" + "---" * 15)

if __name__ == '__main__':
    parse_logs()