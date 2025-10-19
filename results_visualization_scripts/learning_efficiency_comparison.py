"""
Learning Efficiency Comparison: Classic ACO vs LLM ACO
Shows learning efficiency progression over number of trials
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

def find_latest_results(results_type):
    """Find the most recent results file for the given type (aco or aco_llm)."""
    results_dir = f"../results/{results_type}"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory {results_dir} not found!")
        return None
    
    pattern = os.path.join(results_dir, f"multi_run_{results_type}_*.json")
    files = glob.glob(pattern)
    
    if not files:
        print(f"❌ No {results_type} result files found")
        return None
    
    latest_file = max(files, key=os.path.getctime)
    print(f"📊 Using latest {results_type} results: {os.path.basename(latest_file)}")
    return latest_file

def load_results(file_path):
    """Load and extract real learning efficiency data from individual trial results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract individual trial results - the real data!
        individual_results = data.get('individual_results', [])
        if not individual_results:
            print(f"❌ No individual results found in {file_path}")
            return None
        
        learning_efficiencies = []
        for trial in individual_results:
            # Extract actual learning efficiency from swarm_performance
            swarm_perf = trial.get('swarm_performance', {})
            efficiency = swarm_perf.get('learning_efficiency', 0)
            learning_efficiencies.append(efficiency)
        
        print(f"✅ Extracted {len(learning_efficiencies)} real trial learning efficiency values")
        return learning_efficiencies
    
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def create_learning_efficiency_comparison():
    """Create comparison visualization for learning efficiency."""
    # Load data
    aco_file = find_latest_results('aco')
    aco_llm_file = find_latest_results('aco_llm')
    
    if not aco_file or not aco_llm_file:
        print("❌ Missing required result files")
        return
    
    aco_efficiencies = load_results(aco_file)
    aco_llm_efficiencies = load_results(aco_llm_file)
    
    if not aco_efficiencies or not aco_llm_efficiencies:
        print("❌ Failed to load learning efficiency data")
        return
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    trials_aco = range(1, len(aco_efficiencies) + 1)
    trials_aco_llm = range(1, len(aco_llm_efficiencies) + 1)
    
    # Plot individual trial results
    plt.scatter(trials_aco, aco_efficiencies, alpha=0.6, color='lightblue', 
                label='Classic ACO', s=50, edgecolors='darkblue')
    plt.scatter(trials_aco_llm, aco_llm_efficiencies, alpha=0.6, color='lightcoral', 
                label='LLM ACO', s=50, edgecolors='darkred')
    
    # Add trend lines
    z_aco = np.polyfit(trials_aco, aco_efficiencies, 1)
    p_aco = np.poly1d(z_aco)
    plt.plot(trials_aco, p_aco(trials_aco), 'darkblue', linewidth=2, linestyle='--', alpha=0.8)
    
    z_aco_llm = np.polyfit(trials_aco_llm, aco_llm_efficiencies, 1)
    p_aco_llm = np.poly1d(z_aco_llm)
    plt.plot(trials_aco_llm, p_aco_llm(trials_aco_llm), 'darkred', linewidth=2, linestyle='--', alpha=0.8)
    
    # Add mean lines
    aco_mean = np.mean(aco_efficiencies)
    aco_llm_mean = np.mean(aco_llm_efficiencies)
    
    plt.axhline(y=aco_mean, color='darkblue', linestyle='-', alpha=0.5, linewidth=2)
    plt.axhline(y=aco_llm_mean, color='darkred', linestyle='-', alpha=0.5, linewidth=2)
    
    # Formatting
    plt.xlabel('Trial Number', fontsize=14)
    plt.ylabel('Learning Efficiency', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Add statistics text
    stats_text = f"Classic ACO Mean: {aco_mean:.3f}\nLLM ACO Mean: {aco_llm_mean:.3f}\nDifference: {aco_llm_mean - aco_mean:+.3f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11)
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"learning_efficiency_comparison_{timestamp}.png"
    save_path = os.path.join("..", "results", "visualizations", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Learning efficiency comparison saved to: {save_path}")
    
    plt.show()

if __name__ == "__main__":
    print("🎯 Learning Efficiency Comparison: Classic ACO vs LLM ACO")
    print("=" * 60)
    create_learning_efficiency_comparison()
