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
    
    if not aco_file or not aco_llm_file:
        print("❌ Missing required result files")
        return
    
    aco_phase_data = extract_phase_data(aco_file)
    aco_llm_phase_data = extract_phase_data(aco_llm_file)
    
    if not aco_phase_data or not aco_llm_phase_data:
        print("❌ Failed to extract phase data")
        return
    
    # Calculate averages and standard deviations
    phases = ['Early\n(Exploration)', 'Mid\n(Transition)', 'Late\n(Exploitation)']
    phase_keys = ['early_phase', 'mid_phase', 'late_phase']
    
    aco_means = []
    aco_stds = []
    aco_llm_means = []
    aco_llm_stds = []
    
    for phase_key in phase_keys:
        aco_values = aco_phase_data[phase_key]
        aco_llm_values = aco_llm_phase_data[phase_key]
        
        aco_means.append(np.mean(aco_values))
        aco_stds.append(np.std(aco_values))
        aco_llm_means.append(np.mean(aco_llm_values))
        aco_llm_stds.append(np.std(aco_llm_values))
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    
    x = np.arange(len(phases))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, aco_means, width, yerr=aco_stds, 
                   label='Classic ACO', color='lightblue', alpha=0.8, 
                   edgecolor='darkblue', capsize=5, linewidth=1.5)
    bars2 = ax.bar(x + width/2, aco_llm_means, width, yerr=aco_llm_stds,
                   label='LLM ACO', color='lightcoral', alpha=0.8, 
                   edgecolor='darkred', capsize=5, linewidth=1.5)
    
    # Add value labels on bars
    for bars, means in [(bars1, aco_means), (bars2, aco_llm_means)]:
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{mean:.1f}%', ha='center', va='bottom', 
                    fontsize=12, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('')
    ax.set_ylabel('Short Path Selection (%)', fontsize=14)
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
    
    return aco_phase_data, aco_llm_phase_data

def print_phase_statistics(aco_phase_data, aco_llm_phase_data):
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
        print(f"   Classic ACO:  {aco_mean:6.1f}% ± {aco_std:4.1f}%")
        print(f"   LLM ACO:      {aco_llm_mean:6.1f}% ± {aco_llm_std:4.1f}%")
        print(f"   Difference:   {difference:+6.1f}% ({'LLM ACO better' if difference > 0 else 'Classic ACO better'})")
    
    # Overall progression analysis
    print(f"\n📈 Learning Progression Analysis:")
    
    # Classic ACO progression
    aco_early = np.mean(aco_phase_data['early_phase'])
    aco_mid = np.mean(aco_phase_data['mid_phase'])
    aco_late = np.mean(aco_phase_data['late_phase'])
    aco_progression = aco_late - aco_early
    
    # LLM ACO progression
    aco_llm_early = np.mean(aco_llm_phase_data['early_phase'])
    aco_llm_mid = np.mean(aco_llm_phase_data['mid_phase'])
    aco_llm_late = np.mean(aco_llm_phase_data['late_phase'])
    aco_llm_progression = aco_llm_late - aco_llm_early
    
    print(f"   Classic ACO: {aco_early:.1f}% → {aco_mid:.1f}% → {aco_late:.1f}% (Δ{aco_progression:+.1f}%)")
    print(f"   LLM ACO:     {aco_llm_early:.1f}% → {aco_llm_mid:.1f}% → {aco_llm_late:.1f}% (Δ{aco_llm_progression:+.1f}%)")
    
    if aco_llm_progression > aco_progression:
        print(f"   ✅ LLM ACO shows better learning progression (+{aco_llm_progression - aco_progression:.1f}% advantage)")
    else:
        print(f"   ⚠️ Classic ACO shows better learning progression (+{aco_progression - aco_llm_progression:.1f}% advantage)")

if __name__ == "__main__":
    print("📊 Phase Performance Summary: Classic ACO vs LLM ACO")
    print("=" * 60)
    
    # Create visualization and get data
    aco_data, aco_llm_data = create_phase_performance_summary()
    
    if aco_data and aco_llm_data:
        # Print detailed statistics
        print_phase_statistics(aco_data, aco_llm_data)
    
    print(f"\n✅ Phase performance analysis complete!")
