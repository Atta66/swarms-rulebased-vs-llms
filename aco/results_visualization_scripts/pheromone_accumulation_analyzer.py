"""
Pheromone Accumulation Visualizer: Classic ACO vs LLM ACO
Shows how pheromone levels accumulate on short and long paths over iterations
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

def extract_pheromone_data(file_path, max_trials=5):
    """Extract pheromone accumulation data from trial results."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        individual_results = data.get('individual_results', [])
        if not individual_results:
            print(f"❌ No individual results found in {file_path}")
            return None
        
        pheromone_data = []
        
        # Extract data from first few trials to avoid overcrowding
        trials_to_analyze = min(max_trials, len(individual_results))
        
        for trial_idx in range(trials_to_analyze):
            trial = individual_results[trial_idx]
            detailed_steps = trial.get('detailed_step_data', [])
            
            if not detailed_steps:
                continue
            
            trial_data = {
                'trial_number': trial_idx + 1,
                'iterations': [],
                'short_pheromone': [],
                'long_pheromone': [],
                'pheromone_ratio': []
            }
            
            for step in detailed_steps:
                iteration = step.get('step', 0)
                short_pher = step.get('short_pheromone', 0)
                long_pher = step.get('long_pheromone', 0)
                
                trial_data['iterations'].append(iteration)
                trial_data['short_pheromone'].append(short_pher)
                trial_data['long_pheromone'].append(long_pher)
                
                # Calculate ratio (short/long) to show preference
                ratio = short_pher / long_pher if long_pher > 0 else short_pher
                trial_data['pheromone_ratio'].append(ratio)
            
            pheromone_data.append(trial_data)
        
        print(f"✅ Extracted pheromone data from {len(pheromone_data)} trials")
        return pheromone_data
    
    except Exception as e:
        print(f"❌ Error extracting pheromone data from {file_path}: {e}")
        return None

def create_pheromone_accumulation_visualization():
    """Create visualization showing pheromone accumulation patterns."""
    # Load data from both ACO types
    aco_file = find_latest_results('aco')
    aco_llm_file = find_latest_results('aco_llm')
    
    if not aco_file or not aco_llm_file:
        print("❌ Missing required result files")
        return
    
    aco_pheromone_data = extract_pheromone_data(aco_file, max_trials=3)
    aco_llm_pheromone_data = extract_pheromone_data(aco_llm_file, max_trials=3)
    
    if not aco_pheromone_data or not aco_llm_pheromone_data:
        print("❌ Failed to extract pheromone data")
        return
    
    # Create subplot layout: 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Colors for different trials
    colors = ['darkblue', 'blue', 'lightblue']
    aco_llm_colors = ['darkred', 'red', 'lightcoral']
    
    # Plot 1: Classic ACO - Short Path Pheromone
    ax1.set_title('Classic ACO: Short Path Pheromone Accumulation', fontsize=14, fontweight='bold')
    for i, trial_data in enumerate(aco_pheromone_data):
        ax1.plot(trial_data['iterations'], trial_data['short_pheromone'], 
                color=colors[i], linewidth=2, alpha=0.8, 
                label=f'Trial {trial_data["trial_number"]}')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Short Path Pheromone Level')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: LLM ACO - Short Path Pheromone
    ax2.set_title('LLM ACO: Short Path Pheromone Accumulation', fontsize=14, fontweight='bold')
    for i, trial_data in enumerate(aco_llm_pheromone_data):
        ax2.plot(trial_data['iterations'], trial_data['short_pheromone'], 
                color=aco_llm_colors[i], linewidth=2, alpha=0.8, 
                label=f'Trial {trial_data["trial_number"]}')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Short Path Pheromone Level')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Pheromone Ratio Comparison (Short/Long)
    ax3.set_title('Pheromone Ratio: Short/Long Path Preference', fontsize=14, fontweight='bold')
    
    # Average ratio across trials for cleaner comparison
    if aco_pheromone_data and aco_llm_pheromone_data:
        # Calculate average ratios
        max_iterations = min(len(aco_pheromone_data[0]['iterations']), 
                           len(aco_llm_pheromone_data[0]['iterations']))
        
        aco_avg_ratios = []
        aco_llm_avg_ratios = []
        iterations = list(range(max_iterations))
        
        for iter_idx in range(max_iterations):
            aco_ratios_at_iter = []
            aco_llm_ratios_at_iter = []
            
            for trial_data in aco_pheromone_data:
                if iter_idx < len(trial_data['pheromone_ratio']):
                    aco_ratios_at_iter.append(trial_data['pheromone_ratio'][iter_idx])
            
            for trial_data in aco_llm_pheromone_data:
                if iter_idx < len(trial_data['pheromone_ratio']):
                    aco_llm_ratios_at_iter.append(trial_data['pheromone_ratio'][iter_idx])
            
            aco_avg_ratios.append(np.mean(aco_ratios_at_iter) if aco_ratios_at_iter else 0)
            aco_llm_avg_ratios.append(np.mean(aco_llm_ratios_at_iter) if aco_llm_ratios_at_iter else 0)
        
        ax3.plot(iterations, aco_avg_ratios, 'darkblue', linewidth=3, 
                label='Classic ACO (Avg)', alpha=0.8)
        ax3.plot(iterations, aco_llm_avg_ratios, 'darkred', linewidth=3, 
                label='LLM ACO (Avg)', alpha=0.8)
        
        # Add horizontal line at ratio = 1 (equal preference)
        ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Equal Preference')
    
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Pheromone Ratio (Short/Long)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')  # Log scale for better visualization of ratios
    
    # Plot 4: Both Paths Side by Side (First Trial Only)
    ax4.set_title('Path Pheromone Comparison (First Trial)', fontsize=14, fontweight='bold')
    
    if aco_pheromone_data and aco_llm_pheromone_data:
        aco_trial1 = aco_pheromone_data[0]
        aco_llm_trial1 = aco_llm_pheromone_data[0]
        
        # Classic ACO
        ax4.plot(aco_trial1['iterations'], aco_trial1['short_pheromone'], 
                'darkblue', linewidth=2, label='Classic ACO - Short Path')
        ax4.plot(aco_trial1['iterations'], aco_trial1['long_pheromone'], 
                'darkblue', linewidth=2, linestyle='--', alpha=0.7, label='Classic ACO - Long Path')
        
        # LLM ACO
        ax4.plot(aco_llm_trial1['iterations'], aco_llm_trial1['short_pheromone'], 
                'darkred', linewidth=2, label='LLM ACO - Short Path')
        ax4.plot(aco_llm_trial1['iterations'], aco_llm_trial1['long_pheromone'], 
                'darkred', linewidth=2, linestyle='--', alpha=0.7, label='LLM ACO - Long Path')
    
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Pheromone Level')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pheromone_accumulation_analysis_{timestamp}.png"
    save_path = os.path.join("..", "results", "visualizations", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Pheromone accumulation analysis saved to: {save_path}")
    
    plt.show()

def create_pheromone_statistics_summary():
    """Print statistical summary of pheromone accumulation patterns."""
    print("\n" + "="*70)
    print("📊 PHEROMONE ACCUMULATION ANALYSIS SUMMARY")
    print("="*70)
    
    # Load and analyze data
    aco_file = find_latest_results('aco')
    aco_llm_file = find_latest_results('aco_llm')
    
    if not aco_file or not aco_llm_file:
        return
    
    aco_data = extract_pheromone_data(aco_file, max_trials=10)
    aco_llm_data = extract_pheromone_data(aco_llm_file, max_trials=10)
    
    if not aco_data or not aco_llm_data:
        return
    
    # Calculate final pheromone statistics
    aco_final_short = [trial['short_pheromone'][-1] for trial in aco_data if trial['short_pheromone']]
    aco_final_long = [trial['long_pheromone'][-1] for trial in aco_data if trial['long_pheromone']]
    aco_final_ratios = [trial['pheromone_ratio'][-1] for trial in aco_data if trial['pheromone_ratio']]
    
    aco_llm_final_short = [trial['short_pheromone'][-1] for trial in aco_llm_data if trial['short_pheromone']]
    aco_llm_final_long = [trial['long_pheromone'][-1] for trial in aco_llm_data if trial['long_pheromone']]
    aco_llm_final_ratios = [trial['pheromone_ratio'][-1] for trial in aco_llm_data if trial['pheromone_ratio']]
    
    print(f"🔵 Classic ACO Final Pheromone Levels:")
    print(f"   Short Path: {np.mean(aco_final_short):.2f} ± {np.std(aco_final_short):.2f}")
    print(f"   Long Path:  {np.mean(aco_final_long):.2f} ± {np.std(aco_final_long):.2f}")
    print(f"   Ratio (S/L): {np.mean(aco_final_ratios):.2f} ± {np.std(aco_final_ratios):.2f}")
    
    print(f"\n🔴 LLM ACO Final Pheromone Levels:")
    print(f"   Short Path: {np.mean(aco_llm_final_short):.2f} ± {np.std(aco_llm_final_short):.2f}")
    print(f"   Long Path:  {np.mean(aco_llm_final_long):.2f} ± {np.std(aco_llm_final_long):.2f}")
    print(f"   Ratio (S/L): {np.mean(aco_llm_final_ratios):.2f} ± {np.std(aco_llm_final_ratios):.2f}")
    
    # Comparison
    ratio_diff = np.mean(aco_llm_final_ratios) - np.mean(aco_final_ratios)
    print(f"\n📈 Pheromone Preference Comparison:")
    print(f"   LLM ACO shows {ratio_diff:+.2f} higher short path preference ratio")
    if ratio_diff > 0:
        print(f"   ✅ LLM ACO develops stronger short path preference")
    else:
        print(f"   ⚠️ Classic ACO develops stronger short path preference")

if __name__ == "__main__":
    print("🧪 Pheromone Accumulation Analysis: Classic ACO vs LLM ACO")
    print("=" * 70)
    
    # Create main visualization
    create_pheromone_accumulation_visualization()
    
    # Print statistical summary
    create_pheromone_statistics_summary()
    
    print(f"\n✅ Pheromone accumulation analysis complete!")
