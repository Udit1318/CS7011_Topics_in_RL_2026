import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import glob

def plot_averaged_learning_curves(env_name, cost_threshold):
    # Verify the directory exists before attempting to plot
    if not os.path.isdir(env_name):
        print(f"Error: Could not find directory '{env_name}'. Make sure you have trained the agent first.")
        return

    # Find all CSV files inside the nested seed directories
    csv_pattern = os.path.join(env_name, "seed_*", "*.csv")
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching '{csv_pattern}'.")
        return
        
    print(f"Found {len(csv_files)} CSV files in '{env_name}'. Aggregating data...")
    
    # Load and concatenate all seed data into a single DataFrame
    df_list = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # --- Calculate Final Metrics ---
    # Extract data from the very last training epoch across all seeds
    max_epoch = df['Epoch'].max()
    final_epoch_data = df[df['Epoch'] == max_epoch]
    
    # Calculate Mean and Std for Reward and Cost
    final_reward_mean = final_epoch_data['Average_Returns'].mean()
    final_reward_std = final_epoch_data['Average_Returns'].std()
    
    final_cost_mean = final_epoch_data['Average_Discounted_Cost'].mean()
    final_cost_std = final_epoch_data['Average_Discounted_Cost'].std()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    sns.set_theme(style="whitegrid")

    # --- Setup Text Box Styling ---
    bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.85)

    # --- TOP PLOT: Average Reward ---
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Average_Returns',
        ax=ax1,
        color='tab:purple',
        errorbar=('ci', 95),
        n_boot=1000,
        seed=0,
    )
    ax1.set_title(f'{env_name}: Average Discounted Reward', fontsize=14)
    ax1.set_ylabel('Reward Return')
    
    # Add Reward Annotation
    reward_text = f"Final Epoch (Mean ± Std):\n{final_reward_mean:.2f} ± {final_reward_std:.2f}"
    ax1.text(0.02, 0.85, reward_text, transform=ax1.transAxes, fontsize=11, 
             verticalalignment='top', bbox=bbox_props)

    # --- MIDDLE PLOT: Average Cost ---
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Average_Discounted_Cost',
        ax=ax2,
        color='tab:blue',
        label='Agent Speed (Cost)',
        errorbar=('ci', 95),
        n_boot=1000,
        seed=0,
    )
    ax2.axhline(y=cost_threshold, color='tab:red', linestyle='--', linewidth=2, label=f'Limit ({cost_threshold})')
    ax2.set_title(f'{env_name}: Average Discounted Cost (Across Seeds)', fontsize=14)
    ax2.set_ylabel('Cost Return')
    ax2.legend(loc='upper right') # Moved to the right to avoid overlapping text
    
    # Add Cost Annotation
    cost_text = f"Final Epoch (Mean ± Std):\n{final_cost_mean:.2f} ± {final_cost_std:.2f}"
    ax2.text(0.02, 0.85, cost_text, transform=ax2.transAxes, fontsize=11, 
             verticalalignment='top', bbox=bbox_props)

    # --- BOTTOM PLOT: Dual Variable (nu) ---
    sns.lineplot(
        data=df,
        x='Epoch',
        y='Dual_Variable_nu',
        ax=ax3,
        color='tab:green',
        errorbar=('ci', 95),
        n_boot=1000,
        seed=0,
    )
    ax3.set_title(f'{env_name}: FOCOPS Penalty Multiplier (nu) (Across Seeds)', fontsize=14)
    ax3.set_xlabel('Training Epoch')
    ax3.set_ylabel('Value of nu')

    plt.tight_layout()
    
    # Dynamically name the saved plot and save it inside the env directory
    save_path = os.path.join(env_name, f'focops_{env_name}_learning_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot successfully saved as '{save_path}'")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot FOCOPS training curves from a folder of CSV logs.")
    parser.add_argument('--env', type=str, required=True, help="Name of the environment (e.g., Walker2d-v5)")
    parser.add_argument('--threshold', type=float, required=True, help="Cost threshold limit for the environment")
    
    args = parser.parse_args()
    
    plot_averaged_learning_curves(args.env, args.threshold)