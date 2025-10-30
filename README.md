# Swarms: Rule-based vs LLM-powered Boids

A comprehensive comparison framework for analyzing the performance differences between traditional rule-based Boids flocking simulation and LLM-powered Boids algorithms.

## 📋 Overview

This project implements and compares two approaches to the classic Boids flocking simulation:

1. **Classic Rule-based Boids** (`boids.py`): Traditional algorithmic approach using three rules (separation, cohesion, alignment)
2. **LLM-powered Boids** (`boids_llm.py`): AI-powered approach leveraging LLMs to make flocking decisions

The framework includes extensive performance tracking, visualization tools, and statistical analysis capabilities to evaluate both approaches across multiple dimensions.

## 🎯 Features

- **Dual Implementation**: Side-by-side comparison of rule-based and LLM-powered flocking
- **Multi-Trial Analysis**: Run 30 trials with different random seeds for statistical significance
- **Comprehensive Metrics**:
  - Swarm cohesion and coherence
  - Individual agent performance tracking
  - Separation, cohesion, and alignment forces
  - Flocking behavior quality
  - Resource usage (CPU, RAM, GPU)
  - Execution time analysis
- **Visual Simulations**: Real-time visualization using Pygame
- **Headless Mode**: Run experiments without GUI for faster batch processing
- **Reproducible Results**: Seeded random number generation
- **Statistical Analysis**: Mean, standard deviation, and comparative metrics

## 🗂️ Project Structure

```
swarms-rulebased-vs-llms/
├── boids.py                           # Classic rule-based Boids implementation
├── boids_llm.py                       # LLM-based Boids implementation
├── multi_run_boids.py                 # Multi-trial runner for classic Boids
├── multi_run_boids_llm.py             # Multi-trial runner for LLM Boids
├── boids_multi_run_comparison.py.py   # Primary comparison analyzer
├── boids_multi_run_comparison_2.py    # Alternative comparison analyzer
└── config/
    └── config.json                    # Simulation configuration
```

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
pip install pygame numpy matplotlib psutil GPUtil swarm
```

### Configuration

Edit `config/config.json` to customize simulation parameters:

```json
{
    "width": 600,
    "height": 400,
    "point_size": 3,
    "num_points": 3,
    "radius": 50,
    "perception_radius": 150,
    "simulation_steps": 5,
    "velocity_damping": 0.95,
    "position_spread": 0.3,
    "velocity_spread": 0.5,
    "random_seeds": [12345, 67890, ...],
    "prompts": {
        "separation": "...",
        "cohesion": "...",
        "alignment": "..."
    }
}
```

### Running Single Simulations

**Classic Rule-based Boids:**
```powershell
python boids.py
```

**LLM-powered Boids:**
```powershell
python boids_llm.py
```

**Headless Mode (No GUI):**
```powershell
# For batch processing without visualization
python boids.py --headless
python boids_llm.py --headless
```

### Running Multi-Trial Analysis

For statistical significance, run multiple trials:

**Classic Boids (30 trials with different seeds):**
```powershell
python multi_run_boids.py
```

**LLM-powered Boids (30 trials):**
```powershell
python multi_run_boids_llm.py
```

### Comparing Results

After running both classic and LLM experiments:

```powershell
# Primary comparison analysis
python boids_multi_run_comparison.py.py

# Alternative comparison analysis
python boids_multi_run_comparison_2.py
```

## 📊 Metrics & Analysis

### Flocking Behavior Metrics

1. **Swarm Cohesion**: How tightly the flock stays together
2. **Swarm Coherence**: How aligned the velocities are
3. **Separation Quality**: Effective collision avoidance
4. **Alignment Efficiency**: Velocity synchronization
5. **Positional Spread**: Distribution of agents in space
6. **Velocity Consistency**: Uniformity of movement

### Agent Performance Metrics

Each agent is tracked individually for:
- Separation force magnitude and direction
- Cohesion force contributions
- Alignment force effectiveness
- Decision quality (LLM agents)
- Response time per decision

### Resource Metrics

- **CPU Usage**: Per-process CPU consumption
- **RAM Usage**: Memory footprint (MB)
- **GPU Utilization**: GPU load and memory (if available)
- **Execution Time**: Total simulation time
- **Iterations per Second**: Performance throughput

## 🎮 Boids Flocking Rules

### Classic Boids (Reynolds 1987)

**1. Separation (Collision Avoidance)**
- Steer to avoid crowding local flockmates
- Activated within `radius` (default: 50 pixels)

**2. Cohesion (Flock Centering)**
- Steer toward the average position of local flockmates
- Considers neighbors within `perception_radius` (default: 150 pixels)

**3. Alignment (Velocity Matching)**
- Steer toward the average velocity of local flockmates
- Synchronizes movement direction with nearby boids

### LLM Boids

Uses the same three rules but makes decisions via LLM agents:
- **SeparationAgent**: AI determines separation vectors
- **CohesionAgent**: AI calculates cohesion forces
- **AlignmentAgent**: AI computes alignment adjustments

Each agent receives contextual prompts with:
- Current position and velocity
- Nearby boids' positions and velocities
- Relevant radii (separation/perception)
- Velocity damping information

## 📈 Expected Outputs

### Data Files

Generated in `results/` directory:

- `classic_boids_multi_run_YYYYMMDD_HHMMSS.json`: Classic Boids results
- `llm_boids_multi_run_YYYYMMDD_HHMMSS.json`: LLM Boids results
- Individual trial data with performance metrics
- System resource usage logs

### Visualizations

- Real-time flocking animation (Pygame window)
- Swarm cohesion plots over time
- Performance comparison charts
- Resource usage graphs
- Statistical distribution plots
- Agent-level performance heatmaps

### Console Output Example

```
🚀 Running Classic Boids Multi-Run Analysis
============================================================
🔄 Trial 1/30 (Seed: 12345)
⏱️  Trial completed in 2.34s
📊 Swarm Cohesion: 0.87 | Coherence: 0.92

...

📈 Multi-Run Analysis Complete
============================================================
Average Execution Time: 2.45s ± 0.23s
Average Swarm Cohesion: 0.85 ± 0.08
Average Coherence: 0.89 ± 0.06
```

## 🔧 Configuration Parameters

### Simulation Settings

- `width`: Simulation area width (default: 600)
- `height`: Simulation area height (default: 400)
- `num_points`: Number of boids (default: 3)
- `simulation_steps`: Iterations per trial (default: 5)

### Behavior Parameters

- `radius`: Separation radius in pixels (default: 50)
- `perception_radius`: Cohesion/alignment radius (default: 150)
- `velocity_damping`: Speed damping factor (default: 0.95)

### Initial Conditions

- `position_spread`: Initial position clustering (0.0-1.0, default: 0.3)
- `velocity_spread`: Initial velocity variance (default: 0.5)
- `random_seeds`: List of seeds for reproducible multi-run experiments

### LLM Prompts

Each behavior has a customizable prompt template:
- `prompts.separation`: Separation force calculation prompt
- `prompts.cohesion`: Cohesion force calculation prompt
- `prompts.alignment`: Alignment force calculation prompt

## 🤖 LLM Integration

The LLM-based version uses the `swarm` library to create specialized agents:

```python
from swarm import Swarm, Agent

# Example: Separation Agent
separation_agent = Agent(
    name="SeparationAgent",
    instructions="You are a boid at position {position}..."
)

response = client.run(agent=separation_agent, messages=[])
```

### Agent Performance Tracking

The `AgentPerformanceTracker` class monitors:
- Force vectors produced by each agent
- Expected vs actual ranges
- Decision consistency
- Computation time per agent call

## 📊 Results Interpretation

### Good Flocking Indicators

- ✅ **High Cohesion** (> 0.80): Flock stays together
- ✅ **High Coherence** (> 0.85): Synchronized movement
- ✅ **Balanced Forces**: No single rule dominates
- ✅ **Low Variance**: Consistent across trials
- ✅ **Smooth Trajectories**: No erratic movements

### Performance Comparison

**Classic Boids Advantages:**
- Faster execution (no LLM API calls)
- Lower resource usage
- Predictable, deterministic behavior
- No external dependencies

**LLM Boids Advantages:**
- Potentially more adaptive behavior
- Can incorporate complex reasoning
- Natural language programmable
- Easier to modify behavior via prompts

## 🛠️ Troubleshooting

### Common Issues

**Pygame display issues:**
```
pygame.error: No available video device
```
**Solution:** Use headless mode: `python boids.py --headless`

**GPU monitoring not available:**
```
⚠️  GPUtil not available - GPU monitoring disabled
```
**Solution:** Normal if no GPU is present. Simulation continues without GPU tracking.

**LLM API errors:**
```
Error: Swarm API connection failed
```
**Solution:** Check internet connection and API credentials for the swarm library.

**Import errors:**
```
ModuleNotFoundError: No module named 'swarm'
```
**Solution:** Install dependencies:
```powershell
pip install swarm pygame numpy matplotlib psutil GPUtil
```

## 📚 Research Applications

This framework is suitable for:

- Comparing traditional vs LLM-based swarm algorithms
- Studying emergent flocking behavior
- Evaluating LLM decision-making in real-time simulations
- Teaching swarm intelligence and collective behavior
- Benchmarking algorithm performance under various conditions
- Exploring prompt engineering for multi-agent systems

## 🔬 Analysis Tools

### Multi-Run Analyzers

- **`multi_run_boids.py`**: Runs 30 classic boids simulations with different seeds
- **`multi_run_boids_llm.py`**: Runs 30 LLM boids simulations

### Comparison Tools

- **`boids_multi_run_comparison.py.py`**: Primary statistical comparison of classic vs LLM
  - Performance metrics comparison
  - Resource usage comparison
  - Visual plots and charts
  
- **`boids_multi_run_comparison_2.py`**: Alternative comparison analysis
  - Additional statistical perspectives
  - Different visualization approaches

## 📝 Reproducibility

### Random Seeds

The framework uses predefined random seeds from `config.json` to ensure:
- Identical initial conditions across experiments
- Fair comparison between classic and LLM approaches
- Reproducible statistical analysis

### Seed Usage

```python
# In multi_run_boids.py
for seed in self.seeds:
    boids_sim = ClassicBoids(config_file=self.config_file, 
                             seed=seed, 
                             headless=True)
    boids_sim.run()
```

Default seeds: `[12345, 67890, 11111, 22222, ...]` (30 total)

## 🔍 Future Enhancements

- [ ] 3D Boids implementation
- [ ] Obstacle avoidance scenarios
- [ ] Predator-prey dynamics
- [ ] Multiple swarm interactions
- [ ] Real-time LLM prompt optimization
- [ ] Hybrid classic-LLM approaches
- [ ] Web-based visualization dashboard
- [ ] Distributed multi-agent scenarios

## 📄 License

This project is open source and available for research and educational purposes.

## 👥 Contributing

Contributions are welcome! Areas for improvement:
- Additional flocking behaviors
- New performance metrics
- Optimization techniques
- Better visualization methods
- Documentation improvements
- Bug fixes and error handling

## 🙏 Acknowledgments

Built using:
- **Pygame** for real-time visualization
- **NumPy** for numerical computations
- **Matplotlib** for analysis plots
- **Swarm** library for LLM integration
- **PSUtil** and **GPUtil** for resource monitoring

Based on Craig Reynolds' seminal work on Boids (1987).

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Submit a pull request

---

**Note**: LLM performance depends on the model used, API response times, and prompt engineering. Classic boids provide a deterministic baseline for comparison. Results may vary based on hardware and LLM configuration.

## 🎓 Citation

If you use this framework in your research, please cite:

```
Reynolds, C. W. (1987). Flocks, herds and schools: A distributed behavioral model.
SIGGRAPH '87: Proceedings of the 14th annual conference on Computer graphics 
and interactive techniques.
```
