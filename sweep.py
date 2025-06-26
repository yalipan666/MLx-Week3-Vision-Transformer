#!/usr/bin/env python3
"""
Weights & Biases Hyperparameter Sweep Script

This script programmatically creates and runs wandb sweeps for the HackerNews score prediction model.
It provides more control over the sweep process compared to the CLI-based approach.
"""

import wandb
import os

from mnist_transformer import TrainingHyperparameters
from mnist_transformer import ModelHyperparameters
from mnist_transformer import train_model

# Sweep configuration - equivalent to wandb_sweep.yaml but in Python
SWEEP_CONFIG = {
    'method': 'bayes',  # Can be 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'test_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {
            'values': [128]
        },
        'num_epochs':{
            'values': [10]
        },
        'num_heads':{
            'values': [2,4,8]
        },
        'num_layers':{
            'values': [2,4,6]
        },
        'learning_rate': {
            'min': 1e-5,
            'max': 1e-3,
            'distribution': 'log_uniform'
        },
        'weight_decay':{
            'min': 1e-6,
            'max': 1e-3,
            'distribution': 'log_uniform'
        },
        'drop_rate': {
            'min': 0.1,
            'max': 0.5,
            'distribution': 'uniform'
        }
    }
}

def train_sweep_run():
    """
    Single training run for wandb sweep.
    This function is called by the sweep agent for each hyperparameter combination.
    """
    import torch
    from prepare_dataset import Combine
    from mnist_transformer import ViT, evaluate_model

    # Initialize wandb run
    wandb.init()
    
    try:
        config = wandb.config
        print(f"\n🚀 Starting sweep run")

        # Build dataclasses from wandb.config, using defaults for missing values
        train_cfg = TrainingHyperparameters(
            batch_size = config.batch_size,
            num_epochs = config.num_epochs,
            learning_rate = config.learning_rate,
            weight_decay = config.weight_decay,
            drop_rate = config.drop_rate
        )
        model_cfg = ModelHyperparameters(
            num_heads = config.num_heads,
            num_layers = config.num_layers
            # Other model parameters can use defaults or be added to SWEEP_CONFIG if you want to sweep them
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data
        train_dataset = Combine(train=True)
        test_dataset = Combine(train=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

        # Update model_cfg with dataset info
        tmp = next(iter(train_dataset))
        model_cfg.img_width = tmp[0].shape[0]
        model_cfg.img_channels = 1
        model_cfg.patch_size = tmp[1].shape[1]

        # Model, optimizer, loss
        model = ViT(model_cfg, train_cfg).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

        # Training loop
        for epoch in range(train_cfg.num_epochs):
            train_model(model, train_loader, criterion, optimizer, device)
            seq_acc = evaluate_model(model, test_loader, device)
            wandb.log({'epoch': epoch, 'sequence_accuracy': seq_acc})

        print(f"✅ Sweep run completed!")
        
    except Exception as e:
        print(f"❌ Sweep run failed: {e}")
        # Log the failure
        wandb.log({"status": "failed", "error": str(e)})
        raise
    
    finally:
        # Ensure wandb run is properly finished
        wandb.finish()


def create_and_run_sweep(config, project_name, count=10):
    """
    Create and run a wandb sweep programmatically.
    
    Args:
        config: Sweep configuration dictionary (defaults to SWEEP_CONFIG)
        project_name: W&B project name
        count: Number of runs to execute in the sweep
    """
    print(f"🔧 Creating sweep with {config['method']} optimization...")
    print(f"📊 Target metric: {config['metric']['name']} ({config['metric']['goal']})")
    
    # Create the sweep
    sweep_id = wandb.sweep(config, project=project_name)
    print(f"✅ Sweep created with ID: {sweep_id}")
    print(f"🌐 View sweep at: https://wandb.ai/{wandb.api.default_entity}/{project_name}/sweeps/{sweep_id}")
    
    # Run the sweep
    print(f"🏃 Starting sweep agent with {count} runs...")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)
    
    print(f"🎉 Sweep completed!")
    return sweep_id


def run_existing_sweep(sweep_id, project_name, count=10):
    """
    Run an existing sweep by ID.
    
    Args:
        sweep_id: The ID of an existing sweep
        count: Number of additional runs to execute
    """
    print(f"🔄 Joining existing sweep: {sweep_id} against {project_name}")
    wandb.agent(sweep_id, train_sweep_run, project=project_name, count=count)


def main():
    """
    Main function with different sweep options.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hyperparameter sweeps for HackerNews model')
    parser.add_argument('--project', default='hackernews-score-prediction',
                        help='W&B project name (default: hackernews-score-prediction)')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of sweep runs (default: 20)')
    parser.add_argument('--sweep-id', type=str,
                        help='Join existing sweep by ID instead of creating new one')
    parser.add_argument('--dry-run', action='store_true',
                        help='Just show the configuration without running')
    
    args = parser.parse_args()
    
    # Select configuration
    config = SWEEP_CONFIG
    print("📋 Using default Bayesian optimization configuration")
    
    if args.dry_run:
        print("\n🔍 Sweep configuration:")
        import json
        print(json.dumps(config, indent=2))
        return
    
    # Make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Run sweep
    if args.sweep_id:
        run_existing_sweep(args.sweep_id, args.project, args.count)
    else:
        sweep_id = create_and_run_sweep(config, args.project, args.count)
        print(f"\n💾 Save this sweep ID for future use: {sweep_id}")


if __name__ == '__main__':
    main()
