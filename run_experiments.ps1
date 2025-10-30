<#
This script runs the 32 experiments for Group Member One
Fixed Parameters:
  - block_size: 64
  - n_layer: 4

Varying Parameters:
  - n_head: 4, 8
  - n_embd: 128, 256
  - batch_size: 8, 16
  - max_iters: 1000, 2000
  - dropout: 0.1, 0.2
#>

Write-Host "Starting nanoGPT 32 experiments for Group Member One..." -ForegroundColor Green

# --- Define Hyperparameters ---

# Fixed parameters for Group 1
$block_size = 64
$n_layer = 4

# Arrays for varying parameters
$n_heads = @(4, 8)
$n_embds = @(128, 256)
$batch_sizes = @(8, 16)
$max_iters_list = @(1000, 2000)
$dropouts = @(0.1, 0.2)

$experiment_num = 1

# --- Loop Through All Combinations ---
foreach ($n_head in $n_heads) {
    foreach ($n_embd in $n_embds) {
        
        # Check for compatibility as mentioned in the assignment
        if ($n_embd % $n_head -ne 0) {
            Write-Host "Skipping invalid combo: n_embd=$n_embd, n_head=$n_head. n_embd must be divisible by n_head."
            continue
        }

        foreach ($batch_size in $batch_sizes) {
            foreach ($max_iters in $max_iters_list) {
                foreach ($dropout in $dropouts) {
                    
                    # Create a unique, descriptive output directory name
                    $out_dir = "out-g1-exp${experiment_num}-bs${block_size}-nl${n_layer}-nh${n_head}-ne${n_embd}-b${batch_size}-mi${max_iters}-d${dropout}"
                    
                    # --- Create a unique log file name inside the output directory ---
                    $log_file = ".\${out_dir}\training.log"
                    
                    # --- Print and Run ---
                    Write-Host "---" -ForegroundColor Yellow
                    Write-Host "Starting Experiment $experiment_num / 32: $out_dir" -ForegroundColor Cyan
                    Write-Host "Logs will be saved to: $log_file" -ForegroundColor Cyan
                    Write-Host "---" -ForegroundColor Yellow

                    # Wrap the command in 'Measure-Command' to track wall-clock time
                    Measure-Command { `
                        python train.py --dataset=shakespeare `
                        --out_dir=$out_dir `
                        --block_size=$block_size `
                        --n_layer=$n_layer `
                        --n_head=$n_head `
                        --n_embd=$n_embd `
                        --batch_size=$batch_size `
                        --max_iters=$max_iters `
                        --dropout=$dropout `
                        --eval_interval=100 `
                        --log_interval=10 `
                        --device=cuda `
                        --compile=False `
                        *>&1 | Tee-Object -FilePath $log_file
                    }
                    
                    Write-Host "Completed Experiment $experiment_num / 32." -ForegroundColor Green
                    $experiment_num++
                }
            }
        }
    }
}

Write-Host "--- ALL 32 EXPERIMENTS COMPLETE ---" -ForegroundColor Green