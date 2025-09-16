#!/usr/bin/env python3
"""
Complete LLM Boids Multi-Run Analysis with Visualization
Runs the mini multi-run analysis and automatically generates visualization
"""

import subprocess
import sys
import os
from datetime import datetime

def run_mini_analysis(n_trials=5):
    """Run the mini multi-run analysis"""
    print(f"🚀 Running LLM Boids Multi-Run Analysis with {n_trials} trials...")
    print("⚠️  This requires OpenAI API key and will take several minutes")
    print()
    
    # Ask for confirmation
    try:
        proceed = input(f"🔑 Continue with {n_trials} LLM trials? (y/n): ").lower().strip()
        if proceed not in ['y', 'yes']:
            print("❌ Analysis cancelled.")
            return False
    except KeyboardInterrupt:
        print("\n❌ Analysis cancelled.")
        return False
    
    # Run the mini multi-run script
    cmd = [sys.executable, "mini_multi_run_llm.py", "-n", str(n_trials), "--auto"]
    
    try:
        print(f"🔄 Starting mini multi-run analysis...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Multi-run analysis completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Multi-run analysis failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def generate_visualization():
    """Generate the 4-panel visualization"""
    print("📊 Generating visualization...")
    
    try:
        result = subprocess.run([sys.executable, "create_llm_visualization.py"], 
                              check=True, capture_output=True, text=True)
        print("✅ Visualization generated successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Visualization generation failed: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Main execution function"""
    print("🎯 Complete LLM Boids Multi-Run Analysis & Visualization")
    print("=" * 60)
    print("This script will:")
    print("1. Run LLM boids simulations (headless mode)")
    print("2. Generate comprehensive 4-panel visualization")
    print("3. Save results and charts to files")
    print()
    
    # Get number of trials
    try:
        n_input = input("Enter number of trials (default 5, max 30): ").strip()
        if n_input:
            n_trials = int(n_input)
            if n_trials < 1 or n_trials > 30:
                print("❌ Number of trials must be between 1 and 30")
                return
        else:
            n_trials = 5
    except ValueError:
        print("❌ Invalid number of trials")
        return
    except KeyboardInterrupt:
        print("\n❌ Analysis cancelled.")
        return
    
    start_time = datetime.now()
    
    # Step 1: Run analysis
    success = run_mini_analysis(n_trials)
    
    if not success:
        print("❌ Cannot proceed to visualization without successful analysis")
        return
    
    # Step 2: Generate visualization
    viz_success = generate_visualization()
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print()
    print("=" * 60)
    print("🎉 COMPLETE ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"✅ Trials Completed: {n_trials}")
    print(f"📊 Visualization: {'✅ Generated' if viz_success else '❌ Failed'}")
    print(f"⏱️  Total Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()
    print("📂 Generated Files:")
    print("   • Multi-run results: results/visual_llm_boids/")
    if viz_success:
        print("   • Visualization: results/visualizations/")
    print()
    
    if viz_success:
        print("✅ Complete LLM Boids analysis finished successfully!")
        print("💡 Check the visualization PNG file for comprehensive results")
    else:
        print("⚠️  Analysis completed but visualization failed")
        print("💡 Try running: python create_llm_visualization.py manually")

if __name__ == "__main__":
    main()
