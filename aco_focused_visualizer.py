#!/usr/bin/env python3
"""
ACO Performance Visualizer - Clean and Focused

This tool creates a 2x2 visualization showing key ACO performance metrics:
1. Path Selection Probabilities Over Time
2. Exploration-Exploitation Balance 
3. Pheromone Accumulation
4. Phase Performance Summary

Uses optimized parameters from config/aco_config.json:
- evaporation_rate: 0.008 (slower pheromone decay)
- pheromone_deposit: 0.5 (reduced reinforcement)  
- distance_weight: 1.5 (higher sensitivity)
- initial_short_boost: 0.8 (balanced exploration start)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import glob
import sys
from datetime import datetime

class ACOVisualizer:
    def __init__(self, config_file="config/aco_config.json"):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
    def find_latest_multi_run_results(self):
        """Find the latest multi-run ACO results file (including LLM)"""
        # Check both regular ACO and LLM ACO results
        aco_pattern = "results/aco/multi_run_aco_*.json"
        llm_pattern = "results/aco_llm/multi_run_aco_llm_*.json"
        
        aco_files = glob.glob(aco_pattern)
        llm_files = glob.glob(llm_pattern)
        
        all_files = aco_files + llm_files
        
        if not all_files:
            print("⚠️  No multi-run results found, using simulation")
            return None
            
        # Get the latest file by modification time
        latest_file = max(all_files, key=os.path.getmtime)
        
        # Identify the algorithm type
        if "aco_llm" in latest_file:
            algorithm_type = "ACO_LLM"
        else:
            algorithm_type = "ACO"
        
        print(f"📄 Found latest multi-run results: {latest_file}")
        print(f"🤖 Algorithm type: {algorithm_type}")
        return latest_file
    
    def load_multi_run_seed_data(self, seed_choice=0):
        """Load data from a specific seed in the latest multi-run results"""
        latest_file = self.find_latest_multi_run_results()
        
        if not latest_file:
            return None
            
        with open(latest_file, 'r') as f:
            multi_run_data = json.load(f)
        
        individual_results = multi_run_data.get('individual_results', [])
        
        if not individual_results:
            print("⚠️  No individual results found in multi-run data")
            return None
        
        if seed_choice >= len(individual_results):
            seed_choice = 0
            
        chosen_result = individual_results[seed_choice]
        chosen_seed = chosen_result['seed']
        
        print(f"🎲 Using seed {chosen_seed} from multi-run results (choice {seed_choice + 1}/{len(individual_results)})")
        
        # Check if detailed step data is available in the saved results
        if 'detailed_step_data' in chosen_result:
            print("📊 Using saved detailed step data from multi-run results")
            step_data = chosen_result['detailed_step_data']
            
            # Use all available steps (no more 15-step limit)
            total_steps = len(step_data)
            if total_steps > 0:
                print(f"   Using {total_steps} steps for dynamic phase analysis")
            
            # Return full data including phase_selections for visualizer dependency
            return {
                'detailed_step_data': step_data,
                'phase_selections': chosen_result.get('phase_selections'),
                'seed': chosen_seed,
                'algorithm': chosen_result.get('algorithm', 'ACO')
            }
        else:
            print("⚠️  No detailed step data found, re-simulating with same seed")
            # Fallback: re-run the simulation with the same seed
            return self.run_aco_tracking(seed=chosen_seed)
    
    def run_aco_tracking(self, seed=15151, max_iterations=15):
        """Run ACO with tracking"""
        np.random.seed(seed)
        random.seed(seed)
        
        paths = {
            "short": {
                "distance": self.config["paths"]["short"]["distance"],
                "pheromone": self.config["paths"]["short"]["initial_pheromone"]
            },
            "long": {
                "distance": self.config["paths"]["long"]["distance"], 
                "pheromone": self.config["paths"]["long"]["initial_pheromone"]
            }
        }
        
        # Parameters
        evaporation_rate = self.config["evaporation_rate"]
        pheromone_deposit = self.config["pheromone_deposit"]
        distance_weight = self.config["distance_weight"]
        
        step_data = []
        
        for step in range(max_iterations):
            # Calculate attraction
            short_attraction = paths["short"]["pheromone"] / (paths["short"]["distance"] ** distance_weight)
            long_attraction = paths["long"]["pheromone"] / (paths["long"]["distance"] ** distance_weight)
            total_attraction = short_attraction + long_attraction
            
            short_prob = short_attraction / total_attraction
            long_prob = long_attraction / total_attraction
            balance = min(short_prob, long_prob) * 2
            
            # Choose path
            chosen = np.random.choice(["short", "long"], p=[short_prob, long_prob])
            
            step_data.append({
                'step': step,
                'short_pheromone': paths["short"]["pheromone"],
                'long_pheromone': paths["long"]["pheromone"],
                'short_prob': short_prob,
                'long_prob': long_prob,
                'balance': balance,
                'chosen': chosen
            })
            
            # Update pheromones
            paths[chosen]["pheromone"] += pheromone_deposit
            
            # Apply evaporation
            for path in paths.values():
                path["pheromone"] *= (1 - evaporation_rate)
        
        return step_data
    
    def create_focused_visualization_with_choice(self, seed_choice=0):
        """Create focused 2x2 visualization with specified seed choice"""
        # Try to use data from latest multi-run, otherwise simulate
        multi_run_data = self.load_multi_run_seed_data(seed_choice=seed_choice)
        
        if multi_run_data is None:
            print(f"📊 Using default simulation with seed {12345 + seed_choice}")
            step_data = self.run_aco_tracking(seed=12345 + seed_choice)
            return self.create_visualization_from_data(step_data)
        else:
            # Use data from multi-run including saved phase data
            step_data = multi_run_data['detailed_step_data']
            phase_data = multi_run_data['phase_selections']
            return self.create_visualization_from_data(step_data, phase_data)
    
    def create_focused_visualization(self):
        """Create focused 2x2 visualization (legacy method)"""
        return self.create_focused_visualization_with_choice(seed_choice=0)
    
    def create_visualization_from_data(self, step_data, phase_data=None):
        """Create the actual visualization from step data and optional phase data"""
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Optimized ACO Performance: Key Metrics Analysis', fontsize=18, fontweight='bold')
        
        # Extract data
        steps = [d['step'] for d in step_data]
        short_probs = [d['short_prob'] for d in step_data]
        long_probs = [d['long_prob'] for d in step_data]
        balances = [d['balance'] for d in step_data]
        short_pheromones = [d['short_pheromone'] for d in step_data]
        long_pheromones = [d['long_pheromone'] for d in step_data]
        
        # Phase boundaries for equal 5-5-5 division
        early_phase = 4  # Steps 0-4 (5 steps)
        mid_phase = 9    # Steps 5-9 (5 steps)
                        # Steps 10-14 (5 steps)
        
        # 1. Path Selection Probabilities
        ax1 = axes[0, 0]
        ax1.plot(steps, short_probs, 'b-', linewidth=4, label='Short Path', marker='o', markersize=8)
        ax1.plot(steps, long_probs, 'r-', linewidth=4, label='Long Path', marker='s', markersize=8)
        
        # Add phase regions
        ax1.axvspan(0, early_phase, alpha=0.2, color='lightblue', label='Exploration')
        ax1.axvspan(early_phase, mid_phase, alpha=0.2, color='lightyellow', label='Transition')
        ax1.axvspan(mid_phase, 14, alpha=0.2, color='lightcoral', label='Exploitation')
        
        ax1.set_xlabel('Step', fontsize=14)
        ax1.set_ylabel('Selection Probability', fontsize=14)
        ax1.set_title('Path Selection Probabilities Over Time', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 2. Exploration Balance
        ax2 = axes[0, 1]
        ax2.plot(steps, balances, 'green', linewidth=4, marker='o', markersize=8, color='darkgreen')
        
        # Add target zones
        ax2.axhspan(0.4, 0.8, xmin=0, xmax=early_phase/14, alpha=0.3, color='blue', label='Target Early (0.4-0.8)')
        ax2.axhspan(0.2, 0.4, xmin=early_phase/14, xmax=mid_phase/14, alpha=0.3, color='orange', label='Target Mid (0.2-0.4)')
        ax2.axhspan(0.0, 0.3, xmin=mid_phase/14, xmax=1, alpha=0.3, color='red', label='Target Late (0.0-0.3)')
        
        ax2.set_xlabel('Step', fontsize=14)
        ax2.set_ylabel('Balance Score', fontsize=14)
        ax2.set_title('Exploration-Exploitation Balance', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # 3. Pheromone Evolution
        ax3 = axes[1, 0]
        ax3.plot(steps, short_pheromones, 'b-', linewidth=4, label='Short Path', marker='o', markersize=8)
        ax3.plot(steps, long_pheromones, 'r-', linewidth=4, label='Long Path', marker='s', markersize=8)
        
        # Add phase lines
        ax3.axvline(x=early_phase, color='green', linestyle='--', linewidth=3, alpha=0.8)
        ax3.axvline(x=mid_phase, color='green', linestyle='--', linewidth=3, alpha=0.8)
        
        ax3.set_xlabel('Step', fontsize=14)
        ax3.set_ylabel('Pheromone Level', fontsize=14)
        ax3.set_title('Pheromone Accumulation', fontsize=16, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # 4. Phase Performance Summary
        ax4 = axes[1, 1]
        
        # Use saved phase data if available, otherwise calculate from step_data
        if phase_data:
            # Use pre-calculated phase data from multi-run results
            early_short_ratio = phase_data['early_phase']['short_percentage'] / 100
            mid_short_ratio = phase_data['mid_phase']['short_percentage'] / 100
            late_short_ratio = phase_data['late_phase']['short_percentage'] / 100
            
            early_short_count = phase_data['early_phase']['short_count']
            early_long_count = phase_data['early_phase']['long_count']
            mid_short_count = phase_data['mid_phase']['short_count']
            mid_long_count = phase_data['mid_phase']['long_count']
            late_short_count = phase_data['late_phase']['short_count']
            late_long_count = phase_data['late_phase']['long_count']
            
            print(f"📊 Using saved phase data from multi-run results")
        else:
            # Calculate phase statistics with dynamic division (fallback)
            total_steps = len(step_data)
            phase_size = total_steps // 3
            remainder = total_steps % 3
            
            # Distribute remainder across phases
            early_size = phase_size + (1 if remainder > 0 else 0)
            mid_size = phase_size + (1 if remainder > 1 else 0)
            late_size = phase_size
            
            # Calculate dynamic phase boundaries
            early_end = early_size
            mid_end = early_size + mid_size
            
            fallback_early_data = step_data[:early_end]
            fallback_mid_data = step_data[early_end:mid_end]
            fallback_late_data = step_data[mid_end:]
            
            early_short_count = sum(1 for d in fallback_early_data if d['chosen'] == 'short')
            early_long_count = len(fallback_early_data) - early_short_count
            mid_short_count = sum(1 for d in fallback_mid_data if d['chosen'] == 'short')
            mid_long_count = len(fallback_mid_data) - mid_short_count
            late_short_count = sum(1 for d in fallback_late_data if d['chosen'] == 'short')
            late_long_count = len(fallback_late_data) - late_short_count
            
            early_short_ratio = early_short_count / len(fallback_early_data) if len(fallback_early_data) > 0 else 0
            mid_short_ratio = mid_short_count / len(fallback_mid_data) if len(fallback_mid_data) > 0 else 0
            late_short_ratio = late_short_count / len(fallback_late_data) if len(fallback_late_data) > 0 else 0
            
            print(f"📊 Calculated dynamic phase data from step data ({early_size}-{mid_size}-{late_size} division)")
        
        # For visualization, calculate dynamic phase boundaries
        total_steps = len(step_data)
        phase_size = total_steps // 3
        remainder = total_steps % 3
        
        # Distribute remainder across phases
        early_size = phase_size + (1 if remainder > 0 else 0)
        mid_size = phase_size + (1 if remainder > 1 else 0)
        late_size = phase_size
        
        # Calculate dynamic phase boundaries
        early_end = early_size
        mid_end = early_size + mid_size
        
        early_data = step_data[:early_end]
        mid_data = step_data[early_end:mid_end]
        late_data = step_data[mid_end:]
        
        # Dynamic phase labels
        phases = [f'Early\n(0-{early_end-1})', f'Mid\n({early_end}-{mid_end-1})', f'Late\n({mid_end}-{total_steps-1})']
        
        # Calculate metrics for each phase
        phase_metrics = []
        for phase_data in [early_data, mid_data, late_data]:
            if phase_data:
                short_pct = sum(1 for d in phase_data if d['chosen'] == 'short') / len(phase_data)
                avg_balance = np.mean([d['balance'] for d in phase_data])
                phase_metrics.append([short_pct, avg_balance])
            else:
                phase_metrics.append([0, 0])
        
        # Create bar chart
        x = np.arange(len(phases))
        width = 0.35
        
        short_ratios = [m[0] for m in phase_metrics]
        balance_scores = [m[1] for m in phase_metrics]
        
        bars1 = ax4.bar(x - width/2, short_ratios, width, label='Short Path Selection %', 
                       color='lightblue', alpha=0.8, edgecolor='black')
        bars2 = ax4.bar(x + width/2, balance_scores, width, label='Balance Score', 
                       color='lightgreen', alpha=0.8, edgecolor='black')
        
        # Add target lines
        ax4.axhline(y=0.6, color='blue', linestyle=':', alpha=0.7, label='Target 60%')
        ax4.axhline(y=0.75, color='orange', linestyle=':', alpha=0.7, label='Target 75%')
        ax4.axhline(y=0.9, color='red', linestyle=':', alpha=0.7, label='Target 90%')
        
        ax4.set_xlabel('Phase', fontsize=14)
        ax4.set_ylabel('Score', fontsize=14)
        ax4.set_title('Phase Performance Summary', fontsize=16, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(phases)
        ax4.legend(fontsize=11)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # Add value labels on bars with selection counts
        for i, bars in enumerate([bars1, bars2]):
            for j, bar in enumerate(bars):
                height = bar.get_height()
                if i == 0:  # Short path bars - add selection counts
                    phase_data = [early_data, mid_data, late_data][j]
                    short_count = sum(1 for d in phase_data if d['chosen'] == 'short')
                    long_count = len(phase_data) - short_count
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.1%}\n({short_count}S/{long_count}L)', 
                            ha='center', va='bottom', fontsize=10, fontweight='bold')
                else:  # Balance bars
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        # Add parameter info text with selection summary
        early_short_count = sum(1 for d in early_data if d['chosen'] == 'short')
        early_long_count = len(early_data) - early_short_count
        mid_short_count = sum(1 for d in mid_data if d['chosen'] == 'short')
        mid_long_count = len(mid_data) - mid_short_count
        late_short_count = sum(1 for d in late_data if d['chosen'] == 'short')
        late_long_count = len(late_data) - late_short_count
        
        param_text = f"Parameters: evap={self.config['evaporation_rate']}, deposit={self.config['pheromone_deposit']}, weight={self.config['distance_weight']}, boost={self.config['paths']['short']['initial_pheromone']}"
        selection_summary = f"Selection Counts: Early({early_short_count}S/{early_long_count}L) → Mid({mid_short_count}S/{mid_long_count}L) → Late({late_short_count}S/{late_long_count}L)"
        
        fig.text(0.5, 0.04, param_text, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        fig.text(0.5, 0.01, selection_summary, ha='center', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f'results/visualizations/aco_focused_analysis_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        print(f"📊 Focused visualization saved to: {save_path}")
        plt.show()
        
        # Print summary
        early_short = sum(1 for d in early_data if d['chosen'] == 'short') / len(early_data)
        mid_short = sum(1 for d in mid_data if d['chosen'] == 'short') / len(mid_data)
        late_short = sum(1 for d in late_data if d['chosen'] == 'short') / len(late_data)
        
        early_balance = np.mean([d['balance'] for d in early_data])
        late_exploitation = late_short
        learning_efficiency = (early_balance * 0.5 + late_exploitation * 0.5)
        
        # Print detailed selection counts
        print(f"\n📈 DETAILED SELECTION COUNTS:")
        print("=" * 60)
        
        display_early_short_count = sum(1 for d in early_data if d['chosen'] == 'short')
        display_early_long_count = len(early_data) - display_early_short_count
        display_mid_short_count = sum(1 for d in mid_data if d['chosen'] == 'short')
        display_mid_long_count = len(mid_data) - display_mid_short_count
        display_late_short_count = sum(1 for d in late_data if d['chosen'] == 'short')
        display_late_long_count = len(late_data) - display_late_short_count
        
        display_early_short_pct = display_early_short_count / len(early_data) if len(early_data) > 0 else 0
        display_mid_short_pct = display_mid_short_count / len(mid_data) if len(mid_data) > 0 else 0
        display_late_short_pct = display_late_short_count / len(late_data) if len(late_data) > 0 else 0
        
        print(f"📊 EARLY PHASE (Steps 0-{early_end-1}):")
        print(f"   Short Path: {display_early_short_count} selections ({display_early_short_pct:.1%})")
        print(f"   Long Path:  {display_early_long_count} selections ({(1-display_early_short_pct):.1%})")
        print(f"   Total:      {len(early_data)} steps")
        
        print(f"\n📊 MID PHASE (Steps {early_end}-{mid_end-1}):")
        print(f"   Short Path: {display_mid_short_count} selections ({display_mid_short_pct:.1%})")
        print(f"   Long Path:  {display_mid_long_count} selections ({(1-display_mid_short_pct):.1%})")
        print(f"   Total:      {len(mid_data)} steps")
        
        print(f"\n📊 LATE PHASE (Steps {mid_end}-{total_steps-1}):")
        print(f"   Short Path: {display_late_short_count} selections ({display_late_short_pct:.1%})")
        print(f"   Long Path:  {display_late_long_count} selections ({(1-display_late_short_pct):.1%})")
        print(f"   Total:      {len(late_data)} steps")
        
        print(f"\n📈 OVERALL SUMMARY:")
        print(f"Phase Progression: {display_early_short_pct:.1%} → {display_mid_short_pct:.1%} → {display_late_short_pct:.1%}")
        print(f"Phase Distribution: {early_size}-{mid_size}-{late_size} steps ({total_steps} total)")
        print(f"Learning Efficiency: {learning_efficiency:.3f}")
        print(f"Early Balance: {early_balance:.3f}")
        print(f"Late Exploitation: {late_exploitation:.3f}")
        
        # Show step-by-step selections
        print(f"\n🔍 STEP-BY-STEP SELECTIONS:")
        print("=" * 60)
        selections = [d['chosen'] for d in step_data]
        
        print("Early Phase:")
        dynamic_early_selections = selections[:early_end]
        print(f"  Steps 0-{early_end-1}: {' → '.join(dynamic_early_selections)}")
        
        print("Mid Phase:")
        dynamic_mid_selections = selections[early_end:mid_end]
        print(f"  Steps {early_end}-{mid_end-1}: {' → '.join(dynamic_mid_selections)}")
        
        print("Late Phase:")
        dynamic_late_selections = selections[mid_end:]
        print(f"  Steps {mid_end}-{total_steps-1}: {' → '.join(dynamic_late_selections)}")

def main():
    """Create focused visualization"""
    print("📊 ACO Performance Visualization")
    print("=" * 50)
    
    # Check for command line argument to specify seed choice
    seed_choice = 0  # Default
    if len(sys.argv) > 1:
        try:
            seed_choice = int(sys.argv[1])
            print(f"🎯 Using command-line seed choice: {seed_choice}")
        except ValueError:
            print("⚠️  Invalid seed choice argument, using default (0)")
            seed_choice = 0
    
    visualizer = ACOVisualizer()
    
    # Check if multi-run results exist and show options
    latest_file = visualizer.find_latest_multi_run_results()
    if latest_file:
        with open(latest_file, 'r') as f:
            multi_run_data = json.load(f)
        
        seeds = multi_run_data['metadata']['seeds_used']
        total_seeds = len(seeds)
        
        if seed_choice >= total_seeds:
            print(f"⚠️  Seed choice {seed_choice} too high, using {total_seeds-1}")
            seed_choice = total_seeds - 1
        
        print(f"🎲 Available seeds from latest multi-run ({total_seeds} total):")
        for i, seed in enumerate(seeds[:10]):  # Show first 10
            marker = " ← SELECTED" if i == seed_choice else ""
            print(f"   {i}: {seed}{marker}")
        if total_seeds > 10:
            print(f"   ... and {total_seeds-10} more")
        
        print(f"\n💡 To choose a different seed:")
        print(f"   python aco_focused_visualizer.py <seed_number>")
        print(f"   Example: python aco_focused_visualizer.py 2")
    
    # Create visualization with chosen seed
    visualizer.create_focused_visualization_with_choice(seed_choice)
    
    print("\n✅ Visualization completed!")

if __name__ == "__main__":
    main()
