"""
GLAAM Hyperparameter Search with Optuna
Bayesian optimization for finding best GLAAM configuration.

Usage:
    python scripts/optuna_glaam_search.py --n-trials 50 --timeout 48

Key Parameters Being Searched:
- attention_stages: Which MobileNetV2 layers get GLAAM
- reduction: Channel reduction ratio in attention (4, 8, 16, 32)
- dropout_rate: CRITICAL - your 0.3 is too high, testing 0.0-0.2
- learning_rate: 1e-4 to 5e-4
- weight_decay: 0.01 to 0.05
"""

import optuna
from optuna.samplers import TPESampler
import json
import argparse
from pathlib import Path
import subprocess
import pickle
import numpy as np

# Configuration
TEMPLATE_CONFIG = "configs/odir_glaam_template.json"
RESULTS_DIR = Path("optuna_results")
RESULTS_DIR.mkdir(exist_ok=True)

def create_config_for_trial(trial, template_config):
    """
    Generate a config dict based on Optuna trial suggestions.
    """
    # Suggest hyperparameters
    attention_stages_idx = trial.suggest_categorical('attention_stages_idx', [0, 1, 2, 3])
    attention_options = [
        [6, 13, 17],       # Original
        [10, 13, 17],      # Deeper start
        [10, 13, 17, 18],  # Include final layer (best for DR)
        [6, 10, 13, 17],   # Balanced
    ]
    attention_stages = attention_options[attention_stages_idx]
    
    reduction = trial.suggest_categorical('reduction', [4, 8, 16, 32])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.0, 0.1, 0.2])
    learning_rate = trial.suggest_categorical('learning_rate', [1e-4, 3e-4, 5e-4])
    weight_decay = trial.suggest_categorical('weight_decay', [0.01, 0.03, 0.05])
    
    # Create config
    config = template_config.copy()
    config['model_name'] = f"glaam_optuna_trial_{trial.number:03d}"
    config['attention_stages'] = attention_stages
    config['reduction_ratio'] = reduction
    config['dropout_rate'] = dropout_rate
    config['learning_rate'] = learning_rate
    config['weight_decay'] = weight_decay
    
    # Quick run settings for search
    config['epochs'] = 15  # Quick evaluation
    config['early_stopping_patience'] = 5
    
    return config

def run_trial_locally(config, trial_number):
    """
    Run a single trial using Modal or locally.
    Returns best validation AUC.
    """
    # Save trial config
    config_path = RESULTS_DIR / f"trial_{trial_number:03d}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"🧪 Trial {trial_number}")
    print(f"{'='*60}")
    print(f"   attention_stages: {config['attention_stages']}")
    print(f"   reduction_ratio: {config['reduction_ratio']}")
    print(f"   dropout_rate: {config['dropout_rate']}")
    print(f"   learning_rate: {config['learning_rate']}")
    print(f"   weight_decay: {config['weight_decay']}")
    
    # Run training via Modal
    try:
        result = subprocess.run(
            ['modal', 'run', 'modal_train_glaam_odir.py', '--config', str(config_path)],
            capture_output=True,
            text=True,
            timeout=3600 * 2  # 2 hour timeout per trial
        )
        
        # Parse output for best validation AUC
        output = result.stdout + result.stderr
        
        # Look for "Best Val AUC:" in output
        import re
        match = re.search(r'Best Val AUC:\s*([0-9.]+)', output)
        if match:
            val_auc = float(match.group(1))
        else:
            # Fallback: look for test AUC
            match = re.search(r'Test AUC:\s*([0-9.]+)', output)
            if match:
                val_auc = float(match.group(1))
            else:
                print(f"   ⚠️ Could not parse AUC from output")
                val_auc = 0.5  # Penalty for failed trials
        
        print(f"   Result: Val AUC = {val_auc:.4f}")
        
        # Save result
        result_path = RESULTS_DIR / f"trial_{trial_number:03d}_result.json"
        with open(result_path, 'w') as f:
            json.dump({
                'trial_number': trial_number,
                'config': config,
                'val_auc': val_auc
            }, f, indent=2)
        
        return val_auc
        
    except subprocess.TimeoutExpired:
        print(f"   ⚠️ Trial timed out")
        return 0.5
    except Exception as e:
        print(f"   ⚠️ Trial failed: {e}")
        return 0.5

def objective(trial):
    """
    Optuna objective function.
    """
    # Load template config
    with open(TEMPLATE_CONFIG, 'r') as f:
        template_config = json.load(f)
    
    # Create trial config
    config = create_config_for_trial(trial, template_config)
    
    # Run trial
    val_auc = run_trial_locally(config, trial.number)
    
    # Optuna minimizes, so return negative AUC
    return -val_auc

def run_search(n_trials=50, timeout_hours=48):
    """
    Run Bayesian hyperparameter search.
    """
    print("="*70)
    print("🔬 GLAAM Bayesian Hyperparameter Search")
    print("="*70)
    
    # Create Optuna study
    study = optuna.create_study(
        study_name="glaam_hyperparam_search",
        sampler=TPESampler(n_startup_trials=5),  # Random for first 5 trials
        direction="minimize",  # Minimizing negative AUC
        storage=f"sqlite:///{RESULTS_DIR}/optuna_study.db",
        load_if_exists=True
    )
    
    print(f"\n📋 Search Configuration:")
    print(f"   n_trials: {n_trials}")
    print(f"   timeout: {timeout_hours} hours")
    print(f"   storage: {RESULTS_DIR}/optuna_study.db")
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout_hours * 3600,
        n_jobs=1,
        show_progress_bar=True
    )
    
    # Print best trial
    print("\n" + "="*70)
    print("🎯 BEST TRIAL FOUND")
    print("="*70)
    
    best_trial = study.best_trial
    print(f"Trial number: {best_trial.number}")
    print(f"Val AUC: {-best_trial.value:.4f}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"   {key}: {value}")
    
    # Save best config
    with open(TEMPLATE_CONFIG, 'r') as f:
        template_config = json.load(f)
    
    best_config = create_config_for_trial(best_trial, template_config)
    best_config['model_name'] = "glaam_optuna_best"
    best_config['epochs'] = 50  # Full training
    best_config['early_stopping_patience'] = 10
    
    best_config_path = "configs/glaam_optuna_best.json"
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\n💾 Saved best config to: {best_config_path}")
    
    # Create importance plot
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.write_html(str(RESULTS_DIR / "param_importance.html"))
        print(f"📊 Saved importance plot to: {RESULTS_DIR}/param_importance.html")
    except Exception as e:
        print(f"⚠️ Could not create visualization: {e}")
    
    return study

def analyze_study(study_path=None):
    """
    Analyze an existing Optuna study.
    """
    if study_path is None:
        study_path = f"sqlite:///{RESULTS_DIR}/optuna_study.db"
    
    study = optuna.load_study(
        study_name="glaam_hyperparam_search",
        storage=study_path
    )
    
    print("\n📊 Study Analysis:")
    print(f"   Total trials: {len(study.trials)}")
    print(f"   Best Val AUC: {-study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")
    
    # Print top 5 trials
    print("\n🏆 Top 5 Trials:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else float('inf'))
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value is not None:
            print(f"   {i+1}. Trial {trial.number}: AUC={-trial.value:.4f}")
            for key, value in trial.params.items():
                print(f"      {key}: {value}")
    
    return study

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run')
    parser.add_argument('--timeout', type=float, default=48,
                        help='Timeout in hours')
    parser.add_argument('--analyze', action='store_true',
                        help='Analyze existing study instead of running')
    args = parser.parse_args()
    
    if args.analyze:
        analyze_study()
    else:
        study = run_search(n_trials=args.n_trials, timeout_hours=args.timeout)
