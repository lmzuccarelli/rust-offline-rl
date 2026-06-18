use std::fs;
use std::path::{Path, PathBuf};

use crate::trajectory::*;

#[derive(Debug, thiserror::Error)]
pub enum IngestError {
    #[error("IO error at {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("parse error in {context}: {message}")]
    Parse { context: String, message: String },
    #[error("no experiments found in {0}")]
    NoData(String),
}

fn read_file(path: &Path) -> Result<String, IngestError> {
    fs::read_to_string(path).map_err(|e| IngestError::Io {
        path: path.display().to_string(),
        source: e,
    })
}

fn read_file_optional(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

fn parse_stats(content: &str) -> Result<(u64, u64, f64, f64), IngestError> {
    let mut baseline_cycles = 0u64;
    let mut elapsed_cycles = 0u64;
    let mut improvement = 0.0f64;
    let mut reward = 0.0f64;

    for line in content.lines() {
        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key = parts[0].trim();
        let val = parts[1].trim().trim_end_matches('%');

        if key.starts_with("baseline") {
            baseline_cycles = val.parse().unwrap_or(0);
        } else if key.starts_with("elapsed") {
            elapsed_cycles = val.parse().unwrap_or(0);
        } else if key.starts_with("improvement") {
            improvement = val.parse().unwrap_or(0.0);
        } else if key.starts_with("reward") {
            reward = val.parse().unwrap_or(0.0);
        }
    }

    Ok((baseline_cycles, elapsed_cycles, improvement, reward))
}

fn parse_optimization_plan(content: &str) -> Result<Vec<OptimizationTechnique>, IngestError> {
    let cleaned = content.trim();
    let json_str = if cleaned.starts_with("```") {
        let start = cleaned.find('[').unwrap_or(0);
        let end = cleaned.rfind(']').map(|i| i + 1).unwrap_or(cleaned.len());
        &cleaned[start..end]
    } else {
        cleaned
    };

    serde_json::from_str(json_str).map_err(|e| IngestError::Parse {
        context: "optimization-plan.json".to_string(),
        message: e.to_string(),
    })
}

fn parse_rewards_csv(content: &str) -> Vec<f64> {
    content
        .trim()
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect()
}

fn discover_trajectories(experiment_dir: &Path) -> Result<Vec<(usize, String, PathBuf)>, IngestError> {
    let mut trajectories = Vec::new();

    let entries = fs::read_dir(experiment_dir).map_err(|e| IngestError::Io {
        path: experiment_dir.display().to_string(),
        source: e,
    })?;

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("trajectory_") || !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        // trajectory_N_HASH
        let parts: Vec<&str> = name.splitn(3, '_').collect();
        if parts.len() >= 3 {
            if let Ok(id) = parts[1].parse::<usize>() {
                trajectories.push((id, parts[2].to_string(), entry.path()));
            }
        }
    }

    trajectories.sort_by_key(|(id, _, _)| *id);
    Ok(trajectories)
}

fn discover_steps(trajectory_dir: &Path) -> Result<Vec<(usize, PathBuf)>, IngestError> {
    let mut steps = Vec::new();

    let entries = fs::read_dir(trajectory_dir).map_err(|e| IngestError::Io {
        path: trajectory_dir.display().to_string(),
        source: e,
    })?;

    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !name.starts_with("step_") || !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            continue;
        }
        if let Some(id_str) = name.strip_prefix("step_") {
            if let Ok(id) = id_str.parse::<usize>() {
                steps.push((id, entry.path()));
            }
        }
    }

    steps.sort_by_key(|(id, _)| *id);
    Ok(steps)
}

fn find_prompt_file(step_dir: &Path) -> Option<(String, PathBuf)> {
    let entries = fs::read_dir(step_dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name.ends_with(".prompt") {
            let technique = name.trim_end_matches(".prompt").to_string();
            return Some((technique, entry.path()));
        }
    }
    None
}

fn find_response_file(step_dir: &Path, technique: &str) -> Option<PathBuf> {
    let response_name = format!("{}_llm_response.txt", technique);
    let path = step_dir.join(&response_name);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

fn parse_step(
    step_dir: &Path,
    trajectory_id: usize,
    step_id: usize,
) -> Result<Option<StepData>, IngestError> {
    let (technique, prompt_path) = match find_prompt_file(step_dir) {
        Some(v) => v,
        None => {
            tracing::warn!("no .prompt file in {}", step_dir.display());
            return Ok(None);
        }
    };

    let response_path = match find_response_file(step_dir, &technique) {
        Some(v) => v,
        None => {
            tracing::warn!("no response file for {} in {}", technique, step_dir.display());
            return Ok(None);
        }
    };

    let stats_path = step_dir.join("stats.txt");
    if !stats_path.exists() {
        tracing::warn!("no stats.txt in {}", step_dir.display());
        return Ok(None);
    }

    let prompt_text = read_file(&prompt_path)?;
    let response_text = read_file(&response_path)?;
    let stats_content = read_file(&stats_path)?;
    let (baseline_cycles, optimized_cycles, improvement_pct, reward) = parse_stats(&stats_content)?;

    let cuda_path = step_dir.join(format!("{}.cu", technique));
    let cuda_code = read_file_optional(&cuda_path).unwrap_or_default();

    let compile_path = step_dir.join("compile.txt");
    let compiled = read_file_optional(&compile_path)
        .map(|c| !c.to_lowercase().contains("error"))
        .unwrap_or(false);

    let execute_path = step_dir.join("execute.txt");
    let executed = read_file_optional(&execute_path)
        .map(|c| c.trim().to_lowercase().contains("passed"))
        .unwrap_or(false);

    Ok(Some(StepData {
        trajectory_id,
        step_id,
        technique,
        prompt_text,
        response_text,
        cuda_code,
        reward,
        baseline_cycles,
        optimized_cycles,
        improvement_pct,
        compiled,
        executed,
    }))
}

fn parse_baseline(baseline_dir: &Path) -> Result<BaselineData, IngestError> {
    let init_code = read_file(&baseline_dir.join("init.cu"))
        .unwrap_or_else(|_| String::new());
    let profile = read_file_optional(&baseline_dir.join("profile.txt"))
        .unwrap_or_default();
    let state_response = read_file_optional(&baseline_dir.join("llm_state_response.txt"))
        .unwrap_or_default();

    let plan_content = read_file_optional(&baseline_dir.join("optimization-plan.json"))
        .unwrap_or_else(|| "[]".to_string());
    let optimization_plan = parse_optimization_plan(&plan_content).unwrap_or_default();

    Ok(BaselineData {
        init_code,
        profile,
        optimization_plan,
        state_response,
    })
}

fn ingest_experiment(experiment_dir: &Path) -> Result<Option<Experiment>, IngestError> {
    let baseline_dir = experiment_dir.join("baseline");
    if !baseline_dir.exists() {
        tracing::warn!("no baseline dir in {}", experiment_dir.display());
        return Ok(None);
    }

    let baseline = parse_baseline(&baseline_dir)?;

    let rewards_path = experiment_dir.join("rewards.csv");
    let all_rewards = if rewards_path.exists() {
        let content = read_file(&rewards_path)?;
        parse_rewards_csv(&content)
    } else {
        Vec::new()
    };

    let raw_trajectories = discover_trajectories(experiment_dir)?;
    let mut trajectories = Vec::new();

    for (traj_id, hash, traj_dir) in &raw_trajectories {
        let step_dirs = discover_steps(traj_dir)?;
        let mut steps = Vec::new();

        for (step_id, step_dir) in &step_dirs {
            match parse_step(step_dir, *traj_id, *step_id)? {
                Some(step) => steps.push(step),
                None => continue,
            }
        }

        if !steps.is_empty() {
            trajectories.push(Trajectory {
                id: *traj_id,
                hash: hash.clone(),
                steps,
            });
        }
    }

    // Extract names from path: .../model_name/levelN/problem_name/rl-ncu/
    let components: Vec<&str> = experiment_dir
        .components()
        .filter_map(|c| c.as_os_str().to_str())
        .collect();

    let problem_name = components.iter().rev().nth(1).unwrap_or(&"unknown").to_string();
    let level = components.iter().rev().nth(2).unwrap_or(&"unknown").to_string();
    let model_name = components.iter().rev().nth(3).unwrap_or(&"unknown").to_string();

    Ok(Some(Experiment {
        problem_name,
        model_name,
        level,
        baseline,
        trajectories,
        all_rewards,
    }))
}

pub fn ingest_logs(logs_dir: &Path) -> Result<Dataset, IngestError> {
    let mut experiments = Vec::new();

    // Walk: logs/{model}/{level}/{problem}/rl-ncu/
    let walk = walkdir(logs_dir)?;
    for experiment_dir in walk {
        match ingest_experiment(&experiment_dir) {
            Ok(Some(exp)) => {
                tracing::info!(
                    "ingested experiment: {} ({} trajectories, {} steps)",
                    exp.problem_name,
                    exp.trajectories.len(),
                    exp.total_steps()
                );
                experiments.push(exp);
            }
            Ok(None) => {}
            Err(e) => tracing::warn!("failed to ingest {}: {}", experiment_dir.display(), e),
        }
    }

    if experiments.is_empty() {
        return Err(IngestError::NoData(logs_dir.display().to_string()));
    }

    Ok(Dataset { experiments })
}

fn walkdir(logs_dir: &Path) -> Result<Vec<PathBuf>, IngestError> {
    let mut rl_ncu_dirs = Vec::new();
    walk_recursive(logs_dir, 0, &mut rl_ncu_dirs)?;
    Ok(rl_ncu_dirs)
}

fn walk_recursive(dir: &Path, depth: usize, results: &mut Vec<PathBuf>) -> Result<(), IngestError> {
    if depth > 6 {
        return Ok(());
    }

    let dir_name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("");

    if dir_name == "rl-ncu" && dir.join("baseline").exists() {
        results.push(dir.to_path_buf());
        return Ok(());
    }

    let entries = fs::read_dir(dir).map_err(|e| IngestError::Io {
        path: dir.display().to_string(),
        source: e,
    })?;

    for entry in entries.flatten() {
        if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
            walk_recursive(&entry.path(), depth + 1, results)?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_stats() {
        let content = "baseline elapsed cycles : 38730172\n\
                        elapsed cycles          : 24049136\n\
                        improvement             : 37.91%\n\
                        reward                  : 0.3791";
        let (baseline, elapsed, improvement, reward) = parse_stats(content).unwrap();
        assert_eq!(baseline, 38730172);
        assert_eq!(elapsed, 24049136);
        assert!((improvement - 37.91).abs() < 0.01);
        assert!((reward - 0.3791).abs() < 0.001);
    }

    #[test]
    fn test_parse_rewards_csv() {
        let csv = "0.379,0.207,0.295,0.817";
        let rewards = parse_rewards_csv(csv);
        assert_eq!(rewards.len(), 4);
        assert!((rewards[0] - 0.379).abs() < 0.001);
    }

    #[test]
    fn test_parse_optimization_plan() {
        let json = r#"[{"technique":"register_blocking","relevance_score":0.95,"description":"test"}]"#;
        let plan = parse_optimization_plan(json).unwrap();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].technique, "register_blocking");
    }

    #[test]
    fn test_parse_optimization_plan_with_code_fence() {
        let json = "```json\n[{\"technique\":\"test\",\"relevance_score\":0.5,\"description\":\"d\"}]\n```";
        let plan = parse_optimization_plan(json).unwrap();
        assert_eq!(plan.len(), 1);
    }
}
