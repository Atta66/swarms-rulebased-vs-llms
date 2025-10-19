# ACO Comparison Tools

This folder contains tools for comparing the performance of different ACO implementations.

## Files

### `aco_performance_comparison.py`
**Purpose**: Compares Regular ACO vs LLM-Enhanced ACO performance metrics

**Features**:
- Loads latest results from both ACO and ACO LLM multi-run analyses
- Generates detailed performance comparison statistics
- Creates clean visualization with error bars showing performance across 5 key metrics:
  - Convergence Speed
  - Solution Quality  
  - Learning Efficiency
  - Learning Stability
  - Overall Fitness
- Provides timing and resource usage comparisons

**Usage**:
```bash
cd comparison_tools
python aco_performance_comparison.py
```

**Requirements**:
- Must have results from both `multi_run_aco.py` and `multi_run_aco_llm.py`
- Results are automatically loaded from `../results/aco/` and `../results/aco_llm/`

**Output**:
- Terminal summary with detailed statistics
- Visualization saved to `../results/visualizations/aco_performance_comparison_[timestamp].png`

## Output Interpretation

### Performance Metrics
- **Higher is better** for all metrics (0.0 to 1.0 scale)
- **Error bars** show standard deviation across all trials
- **Value labels** show exact mean performance scores

### Key Insights
- **ACO**: Typically faster convergence, lower resource usage
- **ACO LLM**: Often better solution quality and learning progression, but much slower

## Dependencies
- numpy
- matplotlib
- json
- os
- sys
- datetime
