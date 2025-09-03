import json
import time
import math
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class AgentPerformanceMetrics:
    """Performance metrics for individual agents"""
    agent_name: str
    response_time: float = 0.0
    success_rate: float = 0.0
    output_quality_score: float = 0.0
    consistency_score: float = 0.0
    effectiveness_score: float = 0.0
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_response_time: float = 0.0
    last_updated: str = ""

@dataclass
class SwarmPerformanceMetrics:
    """Overall swarm performance metrics"""
    cohesion_metric: float = 0.0
    separation_metric: float = 0.0
    alignment_metric: float = 0.0
    overall_fitness: float = 0.0
    energy_efficiency: float = 0.0
    convergence_rate: float = 0.0

class AgentPerformanceTracker:
    def __init__(self, config_file: str = "config/config.json"):
        self.config_file = config_file
        self.agent_metrics: Dict[str, AgentPerformanceMetrics] = {}
        self.swarm_metrics = SwarmPerformanceMetrics()
        self.performance_history: List[Dict] = []
        self.response_times: Dict[str, List[float]] = {}
        self.output_history: Dict[str, List[Any]] = {}
        self.expected_ranges: Dict[str, Tuple[float, float]] = {
            "SeparationAgent": (-5.0, 5.0),  # Expected dx, dy range
            "CohesionAgent": (-3.0, 3.0),
            "AlignmentAgent": (-4.0, 4.0)
        }
        
    def initialize_agent_metrics(self, agent_name: str):
        """Initialize metrics for a new agent"""
        if agent_name not in self.agent_metrics:
            self.agent_metrics[agent_name] = AgentPerformanceMetrics(agent_name=agent_name)
            self.response_times[agent_name] = []
            self.output_history[agent_name] = []
    
    def track_agent_call(self, agent_name: str, start_time: float, end_time: float, 
                        output: Tuple[float, float], expected_behavior: str = None):
        """Track a single agent call performance"""
        self.initialize_agent_metrics(agent_name)
        
        response_time = end_time - start_time
        metrics = self.agent_metrics[agent_name]
        
        # Update basic metrics
        metrics.total_calls += 1
        metrics.response_time = response_time
        self.response_times[agent_name].append(response_time)
        metrics.average_response_time = np.mean(self.response_times[agent_name])
        
        # Evaluate output quality
        quality_score = self._evaluate_output_quality(agent_name, output)
        metrics.output_quality_score = quality_score
        
        # Track success/failure
        if self._is_valid_output(output):
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
            
        metrics.success_rate = metrics.successful_calls / metrics.total_calls
        
        # Store output history for consistency analysis
        self.output_history[agent_name].append(output)
        metrics.consistency_score = self._calculate_consistency_score(agent_name)
        
        # Update timestamp
        metrics.last_updated = datetime.now().isoformat()
        
    def _is_valid_output(self, output: Tuple[float, float]) -> bool:
        """Check if agent output is valid"""
        try:
            dx, dy = output
            return (isinstance(dx, (int, float)) and isinstance(dy, (int, float)) and
                   not math.isnan(dx) and not math.isnan(dy) and
                   abs(dx) < 100 and abs(dy) < 100)  # Reasonable bounds
        except:
            return False
    
    def _evaluate_output_quality(self, agent_name: str, output: Tuple[float, float]) -> float:
        """Evaluate the quality of agent output based on expected ranges"""
        if not self._is_valid_output(output):
            return 0.0
            
        dx, dy = output
        expected_min, expected_max = self.expected_ranges.get(agent_name, (-10.0, 10.0))
        
        # Score based on whether output is within expected range
        if expected_min <= dx <= expected_max and expected_min <= dy <= expected_max:
            return 1.0
        else:
            # Penalize based on how far outside the range
            dx_penalty = max(0, abs(dx) - expected_max) / expected_max if dx > expected_max else 0
            dy_penalty = max(0, abs(dy) - expected_max) / expected_max if dy > expected_max else 0
            return max(0.0, 1.0 - (dx_penalty + dy_penalty) / 2)
    
    def _calculate_consistency_score(self, agent_name: str) -> float:
        """Calculate how consistent an agent's outputs are"""
        history = self.output_history[agent_name]
        if len(history) < 2:
            return 1.0
            
        # Calculate variance in outputs (lower variance = higher consistency)
        recent_outputs = history[-10:]  # Last 10 outputs
        dx_values = [out[0] for out in recent_outputs if self._is_valid_output(out)]
        dy_values = [out[1] for out in recent_outputs if self._is_valid_output(out)]
        
        if len(dx_values) < 2:
            return 0.5
            
        dx_variance = np.var(dx_values)
        dy_variance = np.var(dy_values)
        
        # Convert variance to consistency score (0-1 scale)
        total_variance = dx_variance + dy_variance
        consistency = 1.0 / (1.0 + total_variance)  # Higher variance = lower consistency
        
        return min(1.0, consistency)
    
    def evaluate_swarm_performance(self, coordinates: List[List[float]], 
                                 velocities: List[List[float]], 
                                 radius: float, perception_radius: float) -> SwarmPerformanceMetrics:
        """Evaluate overall swarm performance"""
        if len(coordinates) < 2:
            return self.swarm_metrics
            
        # Calculate cohesion metric (how close boids are to each other)
        cohesion_score = self._calculate_cohesion_metric(coordinates, perception_radius)
        
        # Calculate separation metric (how well boids avoid crowding)
        separation_score = self._calculate_separation_metric(coordinates, radius)
        
        # Calculate alignment metric (how aligned velocities are)
        alignment_score = self._calculate_alignment_metric(velocities, coordinates, perception_radius)
        
        # Calculate overall fitness
        overall_fitness = (cohesion_score + separation_score + alignment_score) / 3.0
        
        # Update swarm metrics
        self.swarm_metrics.cohesion_metric = cohesion_score
        self.swarm_metrics.separation_metric = separation_score
        self.swarm_metrics.alignment_metric = alignment_score
        self.swarm_metrics.overall_fitness = overall_fitness
        
        return self.swarm_metrics
    
    def _calculate_cohesion_metric(self, coordinates: List[List[float]], perception_radius: float) -> float:
        """Calculate how well boids group together"""
        if len(coordinates) < 2:
            return 1.0
            
        total_distance = 0.0
        count = 0
        
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + 
                                   (coordinates[i][1] - coordinates[j][1])**2)
                if distance <= perception_radius:
                    total_distance += distance
                    count += 1
        
        if count == 0:
            return 0.0
            
        avg_distance = total_distance / count
        # Normalize: closer distances = better cohesion
        cohesion_score = max(0.0, 1.0 - (avg_distance / perception_radius))
        return cohesion_score
    
    def _calculate_separation_metric(self, coordinates: List[List[float]], radius: float) -> float:
        """Calculate how well boids maintain separation"""
        if len(coordinates) < 2:
            return 1.0
            
        violations = 0
        total_pairs = 0
        
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                distance = math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + 
                                   (coordinates[i][1] - coordinates[j][1])**2)
                total_pairs += 1
                if distance < radius:
                    violations += 1
        
        if total_pairs == 0:
            return 1.0
            
        separation_score = 1.0 - (violations / total_pairs)
        return max(0.0, separation_score)
    
    def _calculate_alignment_metric(self, velocities: List[List[float]], 
                                  coordinates: List[List[float]], perception_radius: float) -> float:
        """Calculate how well boids align their velocities"""
        if len(velocities) < 2:
            return 1.0
            
        alignment_scores = []
        
        for i in range(len(velocities)):
            neighbors = []
            for j in range(len(coordinates)):
                if i != j:
                    distance = math.sqrt((coordinates[i][0] - coordinates[j][0])**2 + 
                                       (coordinates[i][1] - coordinates[j][1])**2)
                    if distance <= perception_radius:
                        neighbors.append(j)
            
            if not neighbors:
                continue
                
            # Calculate average neighbor velocity
            avg_vel = [0.0, 0.0]
            for neighbor in neighbors:
                avg_vel[0] += velocities[neighbor][0]
                avg_vel[1] += velocities[neighbor][1]
            avg_vel[0] /= len(neighbors)
            avg_vel[1] /= len(neighbors)
            
            # Calculate alignment with average velocity
            current_vel = velocities[i]
            dot_product = (current_vel[0] * avg_vel[0] + current_vel[1] * avg_vel[1])
            
            current_mag = math.sqrt(current_vel[0]**2 + current_vel[1]**2)
            avg_mag = math.sqrt(avg_vel[0]**2 + avg_vel[1]**2)
            
            if current_mag > 0 and avg_mag > 0:
                cosine_similarity = dot_product / (current_mag * avg_mag)
                alignment_scores.append(max(0.0, cosine_similarity))
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def get_performance_report(self) -> Dict:
        """Generate a comprehensive performance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "agent_performance": {},
            "swarm_performance": asdict(self.swarm_metrics),
            "summary": {}
        }
        
        # Agent performance details
        for agent_name, metrics in self.agent_metrics.items():
            report["agent_performance"][agent_name] = asdict(metrics)
        
        # Summary statistics
        if self.agent_metrics:
            avg_success_rate = np.mean([m.success_rate for m in self.agent_metrics.values()])
            avg_response_time = np.mean([m.average_response_time for m in self.agent_metrics.values()])
            avg_quality = np.mean([m.output_quality_score for m in self.agent_metrics.values()])
            
            report["summary"] = {
                "average_success_rate": avg_success_rate,
                "average_response_time": avg_response_time,
                "average_quality_score": avg_quality,
                "total_agents": len(self.agent_metrics),
                "overall_system_health": self._calculate_system_health()
            }
        
        return report
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health status"""
        if not self.agent_metrics:
            return "UNKNOWN"
            
        avg_success = np.mean([m.success_rate for m in self.agent_metrics.values()])
        avg_quality = np.mean([m.output_quality_score for m in self.agent_metrics.values()])
        
        if avg_success > 0.9 and avg_quality > 0.8:
            return "EXCELLENT"
        elif avg_success > 0.7 and avg_quality > 0.6:
            return "GOOD"
        elif avg_success > 0.5 and avg_quality > 0.4:
            return "FAIR"
        else:
            return "POOR"
    
    def save_performance_data(self, filename: str = "performance_data.json"):
        """Save performance data to file"""
        report = self.get_performance_report()
        self.performance_history.append(report)
        
        with open(filename, 'w') as f:
            json.dump({
                "current_report": report,
                "history": self.performance_history
            }, f, indent=2)
    
    def load_performance_data(self, filename: str = "performance_data.json"):
        """Load performance data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.performance_history = data.get("history", [])
                return data.get("current_report", {})
        except FileNotFoundError:
            return {}
    
    def print_performance_summary(self):
        """Print a readable performance summary"""
        print("\n" + "="*60)
        print("           AGENT PERFORMANCE SUMMARY")
        print("="*60)
        
        for agent_name, metrics in self.agent_metrics.items():
            print(f"\n🤖 {agent_name}:")
            print(f"   Success Rate: {metrics.success_rate:.2%}")
            print(f"   Avg Response Time: {metrics.average_response_time:.3f}s")
            print(f"   Output Quality: {metrics.output_quality_score:.2f}/1.0")
            print(f"   Consistency: {metrics.consistency_score:.2f}/1.0")
            print(f"   Total Calls: {metrics.total_calls}")
            
            # Performance status
            if metrics.success_rate > 0.8 and metrics.output_quality_score > 0.7:
                status = "✅ PERFORMING WELL"
            elif metrics.success_rate > 0.6 and metrics.output_quality_score > 0.5:
                status = "⚠️  NEEDS ATTENTION"
            else:
                status = "❌ POOR PERFORMANCE"
            print(f"   Status: {status}")
        
        print(f"\n🌐 SWARM PERFORMANCE:")
        print(f"   Cohesion: {self.swarm_metrics.cohesion_metric:.2f}/1.0")
        print(f"   Separation: {self.swarm_metrics.separation_metric:.2f}/1.0")
        print(f"   Alignment: {self.swarm_metrics.alignment_metric:.2f}/1.0")
        print(f"   Overall Fitness: {self.swarm_metrics.overall_fitness:.2f}/1.0")
        print(f"   System Health: {self._calculate_system_health()}")
        print("="*60)
