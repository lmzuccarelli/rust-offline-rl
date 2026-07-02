use std::collections::HashMap;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use rand::rngs::StdRng;

use orl_data::replay_buffer::ReplayBuffer;
use orl_model::causal_lm::TrainableCausalLM;

pub type Metrics = HashMap<String, f64>;

pub trait OfflineRLAlgorithm {
    fn name(&self) -> &str;

    fn compute_loss(
        &mut self,
        model: &mut TrainableCausalLM,
        rng: &mut StdRng,
        buffer: &ReplayBuffer,
        device: &Device,
    ) -> candle_core::Result<(Tensor, Metrics)>;

    fn post_step(&mut self) -> candle_core::Result<()> {
        Ok(())
    }

    fn auxiliary_varmap(&self) -> Option<&VarMap> {
        None
    }
}
