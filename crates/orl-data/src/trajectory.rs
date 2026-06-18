use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationTechnique {
    pub technique: String,
    pub relevance_score: f64,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct BaselineData {
    pub init_code: String,
    pub profile: String,
    pub optimization_plan: Vec<OptimizationTechnique>,
    pub state_response: String,
}

#[derive(Debug, Clone)]
pub struct StepData {
    pub trajectory_id: usize,
    pub step_id: usize,
    pub technique: String,
    pub prompt_text: String,
    pub response_text: String,
    pub cuda_code: String,
    pub reward: f64,
    pub baseline_cycles: u64,
    pub optimized_cycles: u64,
    pub improvement_pct: f64,
    pub compiled: bool,
    pub executed: bool,
}

#[derive(Debug, Clone)]
pub struct Trajectory {
    pub id: usize,
    pub hash: String,
    pub steps: Vec<StepData>,
}

impl Trajectory {
    pub fn best_reward(&self) -> f64 {
        self.steps
            .iter()
            .map(|s| s.reward)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    pub fn mean_reward(&self) -> f64 {
        if self.steps.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.steps.iter().map(|s| s.reward).sum();
        sum / self.steps.len() as f64
    }
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub problem_name: String,
    pub model_name: String,
    pub level: String,
    pub baseline: BaselineData,
    pub trajectories: Vec<Trajectory>,
    pub all_rewards: Vec<f64>,
}

impl Experiment {
    pub fn all_steps(&self) -> impl Iterator<Item = &StepData> {
        self.trajectories.iter().flat_map(|t| t.steps.iter())
    }

    pub fn total_steps(&self) -> usize {
        self.trajectories.iter().map(|t| t.steps.len()).sum()
    }

    pub fn unique_techniques(&self) -> Vec<String> {
        let mut techniques: Vec<String> = self
            .all_steps()
            .map(|s| s.technique.clone())
            .collect();
        techniques.sort();
        techniques.dedup();
        techniques
    }
}

#[derive(Debug, Clone)]
pub struct Dataset {
    pub experiments: Vec<Experiment>,
}

impl Dataset {
    pub fn all_steps(&self) -> impl Iterator<Item = &StepData> {
        self.experiments.iter().flat_map(|e| e.all_steps())
    }

    pub fn total_steps(&self) -> usize {
        self.experiments.iter().map(|e| e.total_steps()).sum()
    }

    pub fn total_trajectories(&self) -> usize {
        self.experiments.iter().map(|e| e.trajectories.len()).sum()
    }

    pub fn reward_stats(&self) -> (f64, f64, f64, f64) {
        let rewards: Vec<f64> = self.all_steps().map(|s| s.reward).collect();
        if rewards.is_empty() {
            return (0.0, 0.0, 0.0, 0.0);
        }
        let min = rewards.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let variance = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rewards.len() as f64;
        (min, max, mean, variance.sqrt())
    }
}
