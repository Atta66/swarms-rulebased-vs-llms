"""
Phase Performance Summary: Classic ACO vs LLM ACO
Shows average short path selection percentage across phases for all trials
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

def find_500_iteration_aco_file():
    """Find the specific 500-iteration ACO file (second-to-latest multi_run_aco)."""
    results_dir = "../results/aco"
    if not os.path.exists(results_dir):
        print(f"❌ Results directory {results_dir} not found!")
        return None
    
    # Target the specific 500-iteration file (second-to-latest)
    target_file = os.path.join(results_dir, "multi_run_aco_20251020_163353.json")
    
    if os.path.exists(target_file):
        print(f"📊 Using 500-iteration ACO results: {os.path.basename(target_file)}")
        return target_file
    else:
        print(f"❌ 500-iteration file not found: {target_file}")
        return None

def extract_phase_data(file_path):
    """Extract phase performance data from all trials."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        individual_results = data.get('individual_results', [])
        if not individual_results:
            print(f"❌ No individual results found in {file_path}")
            return None
        
        phase_data = {
            'early_phase': [],
            'mid_phase': [],
            'late_phase': []
        }
        
        for trial in individual_results:
            phase_selections = trial.get('phase_selections', {})
            
            # Extract early phase percentage
            early_phase = phase_selections.get('early_phase', {})
            early_percentage = early_phase.get('short_percentage', 0)
            phase_data['early_phase'].append(early_percentage)
            
            # Extract mid phase percentage (transition phase)
            mid_phase = phase_selections.get('mid_phase', {})
            mid_percentage = mid_phase.get('short_percentage', 0)
            phase_data['mid_phase'].append(mid_percentage)
            
            # Extract late phase percentage (final phase)
            late_phase = phase_selections.get('late_phase', {})
            late_percentage = late_phase.get('short_percentage', 0)
            phase_data['late_phase'].append(late_percentage)
        
        print(f"✅ Extracted phase data from {len(individual_results)} trials")
        return phase_data
    
    except Exception as e:
        print(f"❌ Error extracting phase data from {file_path}: {e}")
        return None

def create_phase_performance_summary():
    """Create phase performance summary visualization."""
    # Load data
    aco_file = find_latest_results('aco')
    aco_llm_file = find_latest_results('aco_llm')
    aco_500_file = find_500_iteration_aco_file()
    
    if not aco_file or not aco_llm_file:
        print("❌ Missing required result files")
        return
    
    aco_phase_data = extract_phase_data(aco_file)
    aco_llm_phase_data = extract_phase_data(aco_llm_file)
    
    # Load 500-iteration data if available
    aco_500_phase_data = None
    if aco_500_file:
        aco_500_phase_data = extract_phase_data(aco_500_file)
        if aco_500_phase_data:
            print("✅ Added 500-iteration Classic ACO data to comparison")
    
    if not aco_phase_data or not aco_llm_phase_data:
        print("❌ Failed to extract phase data")
        return
    
    # Calculate averages and standard deviations
    phases = ['Early Phase', 'Mid Phase', 'Late Phase']
    phase_keys = ['early_phase', 'mid_phase', 'late_phase']
    
    aco_means = []
    aco_stds = []
    aco_llm_means = []
    aco_llm_stds = []
    aco_500_means = []
    aco_500_stds = []
    
    for phase_key in phase_keys:
        aco_values = aco_phase_data[phase_key]
        aco_llm_values = aco_llm_phase_data[phase_key]
        
        aco_means.append(np.mean(aco_values))
        aco_stds.append(np.std(aco_values))
        aco_llm_means.append(np.mean(aco_llm_values))
        aco_llm_stds.append(np.std(aco_llm_values))
        
        # Add 500-iteration data if available
        if aco_500_phase_data:
            aco_500_values = aco_500_phase_data[phase_key]
            aco_500_means.append(np.mean(aco_500_values))
            aco_500_stds.append(np.std(aco_500_values))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(phases))
    width = 0.25 if aco_500_phase_data else 0.35
    
    # Create bars
    bars1 = ax.bar(x - width, aco_means, width, yerr=aco_stds, 
                   label='Classic ACO (18 iter)', color='lightblue', alpha=0.8, 
                   edgecolor='darkblue', capsize=5, linewidth=1.5)
    bars2 = ax.bar(x, aco_llm_means, width, yerr=aco_llm_stds,
                   label='LLM ACO (18 iter)', color='lightcoral', alpha=0.8, 
                   edgecolor='darkred', capsize=5, linewidth=1.5)
    
    # Add 500-iteration bars if data is available
    bars3 = None
    if aco_500_phase_data:
        bars3 = ax.bar(x + width, aco_500_means, width, yerr=aco_500_stds,
                       label='Classic ACO (500 iter)', color='lightgreen', alpha=0.8, 
                       edgecolor='darkgreen', capsize=5, linewidth=1.5)
    
    # Add value labels on bars
    bar_series = [(bars1, aco_means), (bars2, aco_llm_means)]
    if bars3:
        bar_series.append((bars3, aco_500_means))
    
    for bars, means in bar_series:
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            # Smart positioning: inside bar if too high, outside if there's room
            if height > 90:  # If bar is very tall, put label inside
                y_pos = height - 5
                va = 'top'
            else:  # If there's room above, put label outside
                y_pos = height + 1
                va = 'bottom'
            
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{mean:.1f}%', ha='center', va=va, 
                    fontsize=10, fontweight='bold', color='black')
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('Optimal Path Selection (%)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=12)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)
    
    # Add phase descriptions with background colors (similar to your reference)
    phase_colors = ['lightblue', 'lightyellow', 'lightcoral']
    phase_alphas = [0.1, 0.1, 0.1]
    
    for i, (color, alpha) in enumerate(zip(phase_colors, phase_alphas)):
        ax.axvspan(i-0.4, i+0.4, alpha=alpha, color=color)
    
    plt.tight_layout()
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"phase_performance_summary_{timestamp}.png"
    save_path = os.path.join("..", "results", "visualizations", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Phase performance summary saved to: {save_path}")
    
    plt.show()
    
    return aco_phase_data, aco_llm_phase_data, aco_500_phase_data

def print_phase_statistics(aco_phase_data, aco_llm_phase_data, aco_500_phase_data=None):
    """Print detailed phase performance statistics."""
    print("\n" + "="*80)
    print("📊 PHASE PERFORMANCE ANALYSIS")
    print("="*80)
    
    phases = ['Early Phase (Exploration)', 'Mid Phase (Transition)', 'Late Phase (Exploitation)']
    phase_keys = ['early_phase', 'mid_phase', 'late_phase']
    
    for i, (phase_name, phase_key) in enumerate(zip(phases, phase_keys)):
        aco_values = aco_phase_data[phase_key]
        aco_llm_values = aco_llm_phase_data[phase_key]
        
        aco_mean = np.mean(aco_values)
        aco_std = np.std(aco_values)
        aco_llm_mean = np.mean(aco_llm_values)
        aco_llm_std = np.std(aco_llm_values)
        
        difference = aco_llm_mean - aco_mean
        
        print(f"\n🔍 {phase_name}:")
        print(f"   Classic ACO (18):   {aco_mean:6.1f}% ± {aco_std:4.1f}%")
        print(f"   LLM ACO (18):       {aco_llm_mean:6.1f}% ± {aco_llm_std:4.1f}%")
        
        # Add 500-iteration data if available
        if aco_500_phase_data:
            aco_500_values = aco_500_phase_data[phase_key]
            aco_500_mean = np.mean(aco_500_values)
            aco_500_std = np.std(aco_500_values)
            print(f"   Classic ACO (500):  {aco_500_mean:6.1f}% ± {aco_500_std:4.1f}%")
            
            difference_500 = aco_500_mean - aco_mean
            print(f"   18 vs 500 iter:     {difference_500:+6.1f}% ({'500 iter better' if difference_500 > 0 else '18 iter better'})")
        
        print(f"   18 iter difference:  {difference:+6.1f}% ({'LLM ACO better' if difference > 0 else 'Classic ACO better'})")
    
    # Overall progression analysis
    print(f"\n📈 Learning Progression Analysis:")
    
    # Classic ACO progression (18 iterations)
    aco_early = np.mean(aco_phase_data['early_phase'])
    aco_mid = np.mean(aco_phase_data['mid_phase'])
    aco_late = np.mean(aco_phase_data['late_phase'])
    aco_progression = aco_late - aco_early
    
    # LLM ACO progression (18 iterations)
    aco_llm_early = np.mean(aco_llm_phase_data['early_phase'])
    aco_llm_mid = np.mean(aco_llm_phase_data['mid_phase'])
    aco_llm_late = np.mean(aco_llm_phase_data['late_phase'])
    aco_llm_progression = aco_llm_late - aco_llm_early
    
    print(f"   Classic ACO (18):  {aco_early:.1f}% → {aco_mid:.1f}% → {aco_late:.1f}% (Δ{aco_progression:+.1f}%)")
    print(f"   LLM ACO (18):      {aco_llm_early:.1f}% → {aco_llm_mid:.1f}% → {aco_llm_late:.1f}% (Δ{aco_llm_progression:+.1f}%)")
    
    # 500-iteration progression if available
    if aco_500_phase_data:
        aco_500_early = np.mean(aco_500_phase_data['early_phase'])
        aco_500_mid = np.mean(aco_500_phase_data['mid_phase'])
        aco_500_late = np.mean(aco_500_phase_data['late_phase'])
        aco_500_progression = aco_500_late - aco_500_early
        
        print(f"   Classic ACO (500): {aco_500_early:.1f}% → {aco_500_mid:.1f}% → {aco_500_late:.1f}% (Δ{aco_500_progression:+.1f}%)")
        
        # Compare progressions
        if aco_500_progression > aco_llm_progression:
            print(f"   ✅ Classic ACO (500 iter) shows best learning progression (+{aco_500_progression - aco_llm_progression:.1f}% vs LLM)")
        elif aco_llm_progression > aco_progression:
            print(f"   ✅ LLM ACO shows better learning progression than Classic (18 iter) (+{aco_llm_progression - aco_progression:.1f}%)")
        else:
            print(f"   ⚠️ Classic ACO (18 iter) shows better learning progression (+{aco_progression - aco_llm_progression:.1f}% advantage)")
    else:
        if aco_llm_progression > aco_progression:
            print(f"   ✅ LLM ACO shows better learning progression (+{aco_llm_progression - aco_progression:.1f}% advantage)")
        else:
            print(f"   ⚠️ Classic ACO shows better learning progression (+{aco_progression - aco_llm_progression:.1f}% advantage)")

if __name__ == "__main__":
    print("📊 Phase Performance Summary: Classic ACO vs LLM ACO (with 500-iter comparison)")
    print("=" * 80)
    
    # Create visualization and get data
    result = create_phase_performance_summary()
    
    if result and len(result) >= 2:
        aco_data, aco_llm_data = result[0], result[1]
        aco_500_data = result[2] if len(result) > 2 else None
        
        # Print detailed statistics
        print_phase_statistics(aco_data, aco_llm_data, aco_500_data)
    
    print(f"\n✅ Phase performance analysis complete!")
