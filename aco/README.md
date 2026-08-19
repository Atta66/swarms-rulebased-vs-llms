# Swarms: Rule-based vs LLM-powered ACO

A comprehensive comparison framework for analyzing the performance differences between traditional rule-based Ant Colony Optimization (ACO) and Large Language Model (LLM)-powered ACO algorithms.

## 📋 Overview

This project implements and compares two approaches to solving the classic Ant Colony Optimization problem:

1. **Rule-Based ACO** (`aco.py`): Traditional algorithmic approach using deterministic pheromone update rules
2. **LLM-powered ACO** (`aco_llm_enhanced.py`): AI-powered approach leveraging Large Language Models for dynamic decision-making

The framework includes extensive performance tracking, visualization tools, and statistical analysis capabilities to evaluate both approaches across multiple dimensions.

## 🎯 Features

- **Dual Implementation**: Side-by-side comparison of rule-based and LLM-powered algorithms
- **Multi-Trial Analysis**: Run 30 (can be changed) trials for statistical significance
- **Comprehensive Metrics**:
  - Path selection efficiency
  - Pheromone accumulation patterns
  - Convergence speed
  - Exploration vs exploitation balance
  - Learning efficiency and stability
  - Solution quality
- **Resource Monitoring**: CPU, RAM, and GPU usage tracking
- **Visualization Suite**: 9 specialized analysis scripts
- **Configurable Parameters**: JSON-based configuration system

## 🗂️ Project Structure

```
swarms-rulebased-vs-llms/
├── aco.py                          # Rule-based ACO implementation
├── aco_llm_enhanced.py             # LLM-based ACO implementation
├── multi_run_aco.py                # Multi-trial runner for rule-based ACO
├── multi_run_aco_llm.py            # Multi-trial runner for LLM ACO
├── aco_focused_visualizer.py       # Real-time visualization tool
├── config/
│   ├── aco_config.json             # Rule-based ACO configuration
│   └── config.json                 # LLM ACO configuration
└── results_visualization_scripts/
    ├── aco_iteration_analyzer.py
    ├── aco_performance_comparison.py
    ├── convergence_speed_comparison.py
    ├── exploration_exploitation_analyzer.py
    ├── learning_efficiency_comparison.py
    ├── learning_stability_comparison.py
    ├── phase_performance_summary.py
    ├── pheromone_accumulation_analyzer.py
    └── solution_quality_comparison.py
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install numpy matplotlib psutil GPUtil swarm
```

### Configuration

Edit the configuration files to customize algorithm parameters:

**`config/aco_config.json`** (Rule-based ACO):
```json
{
    "max_iterations": 18,
    "num_trials": 30,
    "evaporation_rate": 0.02,
    "target_ratio": 236.2,
    "ratio_tolerance": 1.0,
    "paths": {
        "short": {"distance": 1, "initial_pheromone": 1.0},
        "long": {"distance": 2, "initial_pheromone": 2.0}
    }
}
```

**`config/config.json`** (LLM-based ACO):
- Similar structure with additional LLM-specific parameters

### Running Single Experiments

**Rule-based ACO:**
```bash
python aco.py
```

**LLM-powered ACO:**
```bash
python aco_llm_enhanced.py
```

### Running Multi-Trial Analysis

For statistical significance, run multiple trials:

**Rule-based ACO (30 trials):**
```bash
python multi_run_aco.py
```

**LLM-based ACO (30 trials):**
```bash
python multi_run_aco_llm.py
```

### Visualizing Results

After running experiments, use the visualization scripts:

```bash
# Compare overall performance
python results_visualization_scripts/aco_performance_comparison.py

# Analyze convergence speed
python results_visualization_scripts/convergence_speed_comparison.py

# Examine exploration vs exploitation
python results_visualization_scripts/exploration_exploitation_analyzer.py

# And more...
```

## 📊 Metrics & Analysis

### Performance Metrics

1. **Path Selection Ratio**: Ratio of short path selections to long path selections
2. **Convergence Speed**: Iterations required to reach target ratio
3. **Pheromone Accumulation**: Rate and pattern of pheromone buildup
4. **Exploration Efficiency**: Path diversity in early iterations
5. **Learning Stability**: Variance in performance across trials
6. **Solution Quality**: Final ratio accuracy vs target

### Resource Metrics

- CPU usage per iteration
- RAM consumption
- GPU utilization (if available)
- Execution time

## 🔬 Analysis Scripts

| Script | Purpose |
|--------|---------|
| `aco_iteration_analyzer.py` | Per-iteration performance breakdown |
| `aco_performance_comparison.py` | Overall performance comparison |
| `convergence_speed_comparison.py` | Time-to-convergence analysis |
| `exploration_exploitation_analyzer.py` | Balance between exploration and exploitation |
| `learning_efficiency_comparison.py` | Learning rate and improvement tracking |
| `learning_stability_comparison.py` | Variance and reliability metrics |
| `phase_performance_summary.py` | Performance across algorithm phases |
| `pheromone_accumulation_analyzer.py` | Pheromone trail dynamics |
| `solution_quality_comparison.py` | Final solution accuracy |

## 📈 Expected Outputs

### Data Files
- JSON results files with complete trial data
- Performance metrics per trial
- Resource usage logs

### Visualizations
- Convergence curves
- Pheromone evolution plots
- Path selection histograms
- Statistical comparison charts
- Resource utilization graphs

## 🎛️ Configuration Parameters

### Common Parameters

- `max_iterations`: Maximum iterations per trial (default: 18)
- `num_trials`: Number of trials to run (default: 30)
- `evaporation_rate`: Pheromone evaporation rate (default: 0.02)
- `target_ratio`: Target short/long path ratio (default: 236.2 - optional for experimentation for early convergence
- `ratio_tolerance`: Acceptable deviation from target (default: 1.0) - optional for experimentation for early convergence

### Path Configuration

- `short.distance`: Distance of short path (default: 1)
- `long.distance`: Distance of long path (default: 2)
- `initial_pheromone`: Starting pheromone level

## 🤖 LLM Integration

The LLM-powered version uses the `swarm` library to integrate AI decision-making:

- **Path Selection Agent**: Chooses paths based on pheromone and distance
- **Pheromone Update Agent**: Calculates optimal pheromone updates
- **Strategy Agent**: Adjusts exploration/exploitation balance

## 📝 Results Interpretation

### Good Convergence Indicators
- ✅ Reaches target ratio within tolerance
- ✅ Stable pheromone accumulation
- ✅ Balanced exploration in early phases
- ✅ Low variance across trials

### Performance Comparison Metrics
- Convergence speed (faster is better)
- Solution quality (closer to target is better)
- Resource efficiency (lower usage is better)
- Stability (lower variance is better)


## 📄 License

This project is open source and available for research and educational purposes.


## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub


**Note**: This framework is designed for research and educational purposes. Results may vary based on hardware, LLM model used, and random seed values.
