use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct SchedulerConfig {
    #[serde(rename = "type")]
    pub scheduler_type: String,
    pub min_lr: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            scheduler_type: "cosine".to_string(),
            min_lr: 1e-6,
        }
    }
}

pub struct LrScheduler {
    config: SchedulerConfig,
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    current_step: usize,
}

impl LrScheduler {
    pub fn new(config: SchedulerConfig, base_lr: f64, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            config,
            base_lr,
            warmup_steps,
            total_steps,
            current_step: 0,
        }
    }

    pub fn step(&mut self) -> f64 {
        let lr = self.get_lr();
        self.current_step += 1;
        lr
    }

    pub fn get_lr(&self) -> f64 {
        if self.current_step < self.warmup_steps {
            // Linear warmup
            let progress = self.current_step as f64 / self.warmup_steps.max(1) as f64;
            self.config.min_lr + (self.base_lr - self.config.min_lr) * progress
        } else {
            match self.config.scheduler_type.as_str() {
                "cosine" => self.cosine_lr(),
                "linear" => self.linear_lr(),
                _ => self.base_lr,
            }
        }
    }

    fn cosine_lr(&self) -> f64 {
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let current = self.current_step.saturating_sub(self.warmup_steps);
        let progress = current as f64 / decay_steps as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        self.config.min_lr + (self.base_lr - self.config.min_lr) * cosine
    }

    fn linear_lr(&self) -> f64 {
        let decay_steps = self.total_steps.saturating_sub(self.warmup_steps).max(1);
        let current = self.current_step.saturating_sub(self.warmup_steps);
        let progress = current as f64 / decay_steps as f64;
        let linear = 1.0 - progress;
        self.config.min_lr + (self.base_lr - self.config.min_lr) * linear
    }

    pub fn current_step(&self) -> usize {
        self.current_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warmup() {
        let config = SchedulerConfig {
            scheduler_type: "cosine".to_string(),
            min_lr: 0.0,
        };
        let mut sched = LrScheduler::new(config, 1.0, 10, 100);

        let lr0 = sched.step();
        assert!(lr0 < 0.2, "first step lr should be small: {}", lr0);

        for _ in 1..10 {
            sched.step();
        }
        let lr10 = sched.get_lr();
        assert!((lr10 - 1.0).abs() < 0.01, "after warmup lr should be ~1.0: {}", lr10);
    }

    #[test]
    fn test_cosine_decay() {
        let config = SchedulerConfig {
            scheduler_type: "cosine".to_string(),
            min_lr: 0.0,
        };
        let mut sched = LrScheduler::new(config, 1.0, 0, 100);

        let lr_start = sched.step();
        assert!((lr_start - 1.0).abs() < 0.05);

        for _ in 1..50 {
            sched.step();
        }
        let lr_mid = sched.get_lr();
        assert!((lr_mid - 0.5).abs() < 0.1, "mid lr should be ~0.5: {}", lr_mid);
    }
}
