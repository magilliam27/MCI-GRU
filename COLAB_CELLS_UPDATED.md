# Updated Colab Workflow Cells

## Cell 2: Clone Repository (Updated for modular_correct branch)
```python
# Clone repository (modular_correct branch)
import os

if not os.path.exists('/content/MCI-GRU'):
    !git clone -b modular_correct https://github.com/yourusername/MCI-GRU.git /content/MCI-GRU
    print("✓ Repository cloned (modular_correct branch)")
else:
    print("✓ Repository already exists")
    # Pull latest changes
    %cd /content/MCI-GRU
    !git fetch origin
    !git checkout modular_correct
    !git pull origin modular_correct
    print("✓ Updated to latest modular_correct branch")

%cd /content/MCI-GRU

# Verify branch
!git branch --show-current

# Install requirements
!pip install -q -r requirements.txt
print("✓ Dependencies installed")
```

## Cell 3: Run Training (Simplified - uses correct defaults now)
```python
# Run training experiment
GDRIVE_BASE = '/content/drive/MyDrive/MCI-GRU-Experiments'

# Make sure we're in the repo directory
%cd /content/MCI-GRU

# Run training - all defaults are now correct!
!python run_experiment.py \
    output_dir={GDRIVE_BASE} \
    experiment_name=baseline \
    training.num_epochs=100 \
    training.num_models=10

print("\n" + "="*80)
print("✓ Training complete!")
print(f"Check outputs in: {GDRIVE_BASE}/baseline/")
print("="*80)
```

## Cell 4: Find Latest Run
```python
# Find the most recent run
import glob
import os

def find_latest_run(base_dir, experiment_name):
    """Find the most recent run for an experiment."""
    experiment_dir = f"{base_dir}/{experiment_name}"
    run_dirs = sorted(glob.glob(f"{experiment_dir}/*/"))
    if run_dirs:
        latest = run_dirs[-1].rstrip('/')
        print(f"Latest run: {latest}")
        return latest
    else:
        print(f"No runs found for: {experiment_name}")
        return None

# Find latest baseline run
LATEST_RUN = find_latest_run(GDRIVE_BASE, 'baseline')

if LATEST_RUN:
    PREDICTIONS_PATH = f"{LATEST_RUN}/averaged_predictions"
    print(f"\nPredictions path: {PREDICTIONS_PATH}")
    
    # Check what was generated
    print(f"\nGenerated files:")
    !ls -lh {LATEST_RUN}
```

## Cell 5: Run Backtesting (Simplified - uses correct defaults now)
```python
# Backtest WITHOUT transaction costs
print("Running backtest WITHOUT transaction costs...\n")

!python evaluate_sp500.py \
    --predictions_dir {PREDICTIONS_PATH} \
    --auto_save \
    --plot

print(f"\n✓ Backtest complete (no transaction costs)")
print(f"Results saved to: {LATEST_RUN}/backtest/")
```

## Cell 6: Backtest WITH Transaction Costs
```python
# Backtest WITH transaction costs
print("\nRunning backtest WITH transaction costs...\n")

!python evaluate_sp500.py \
    --predictions_dir {PREDICTIONS_PATH} \
    --auto_save \
    --backtest_suffix _with_costs \
    --transaction_costs \
    --spread 10 \
    --slippage 5 \
    --plot

print(f"\n✓ Backtest complete (with transaction costs)")
print(f"Results saved to: {LATEST_RUN}/backtest_with_costs/")
```

## Cell 7: View Results
```python
# View summary text file
summary_file = f"{LATEST_RUN}/backtest/summary.txt"
if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        print(f.read())
else:
    print("Summary file not found")
```

## Cell 8: Display Equity Curve
```python
# Display equity curve
from IPython.display import Image, display

equity_plot = f"{LATEST_RUN}/backtest/equity_curve.png"
if os.path.exists(equity_plot):
    display(Image(filename=equity_plot))
else:
    print("Equity curve plot not found")
```

## Cell 9: Load and Display Results DataFrame
```python
# Load and display results
import pandas as pd

results_file = f"{LATEST_RUN}/backtest/backtest_results.csv"
if os.path.exists(results_file):
    results_df = pd.read_csv(results_file)
    display(results_df.T)  # Transpose for easier reading
else:
    print("Results file not found")
```

## Optional: LSEG Setup (if using LSEG data)
```python
# Cell 1.5: Setup LSEG credentials (insert after Cell 1)
import os
from google.colab import userdata

# Store LSEG API key in Colab Secrets (left sidebar -> Key icon)
try:
    LSEG_API_KEY = userdata.get('LSEG_API_KEY')
    os.environ['LSEG_API_KEY'] = LSEG_API_KEY
    print("✓ LSEG API key loaded from Colab Secrets")
except:
    print("⚠️  LSEG API key not found in Colab Secrets")
    print("   Add 'LSEG_API_KEY' in the Secrets panel if using LSEG data source")
```

## Notes:
1. No need to specify `data.filename` anymore - defaults are correct
2. No need to specify `--data_file` or `--test_start/end` - defaults match training
3. Simplified commands that "just work"
4. All paths respect Google Drive output directory
