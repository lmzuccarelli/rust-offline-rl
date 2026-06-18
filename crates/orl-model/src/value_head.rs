use candle_core::{DType, Device, Tensor};
use candle_nn::{linear_no_bias, VarBuilder, VarMap};

#[derive(Debug, thiserror::Error)]
pub enum ValueHeadError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

pub struct ValueHeads {
    q_head_w1: candle_nn::Linear,
    q_head_w2: candle_nn::Linear,
    v_head: candle_nn::Linear,

    target_q_head_w1: candle_nn::Linear,
    target_q_head_w2: candle_nn::Linear,
    target_v_head: candle_nn::Linear,

}

impl ValueHeads {
    pub fn new(
        hidden_size: usize,
        vocab_size: usize,
        varmap: &VarMap,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, ValueHeadError> {
        let vb = VarBuilder::from_varmap(varmap, dtype, device);

        let q_head_w1 = linear_no_bias(hidden_size, hidden_size, vb.pp("q_head.0"))?;
        let q_head_w2 = linear_no_bias(hidden_size, vocab_size, vb.pp("q_head.1"))?;
        let v_head = linear_no_bias(hidden_size, 1, vb.pp("v_head"))?;

        // Target networks are copies, initialized to same weights
        let target_vb = VarBuilder::from_varmap(varmap, dtype, device);
        let target_q_head_w1 = linear_no_bias(hidden_size, hidden_size, target_vb.pp("target_q_head.0"))?;
        let target_q_head_w2 = linear_no_bias(hidden_size, vocab_size, target_vb.pp("target_q_head.1"))?;
        let target_v_head = linear_no_bias(hidden_size, 1, target_vb.pp("target_v_head"))?;

        Ok(Self {
            q_head_w1,
            q_head_w2,
            v_head,
            target_q_head_w1,
            target_q_head_w2,
            target_v_head,
        })
    }

    pub fn q_values(&self, hidden_states: &Tensor) -> Result<Tensor, candle_core::Error> {
        let h = hidden_states.apply(&self.q_head_w1)?;
        let h = h.gelu()?;
        h.apply(&self.q_head_w2)
    }

    pub fn v_values(&self, hidden_states: &Tensor) -> Result<Tensor, candle_core::Error> {
        let v = hidden_states.apply(&self.v_head)?;
        v.squeeze(candle_core::D::Minus1)
    }

    pub fn target_q_values(&self, hidden_states: &Tensor) -> Result<Tensor, candle_core::Error> {
        let h = hidden_states.apply(&self.target_q_head_w1)?;
        let h = h.gelu()?;
        h.apply(&self.target_q_head_w2)
    }

    pub fn target_v_values(&self, hidden_states: &Tensor) -> Result<Tensor, candle_core::Error> {
        let v = hidden_states.apply(&self.target_v_head)?;
        v.squeeze(candle_core::D::Minus1)
    }

    pub fn update_targets(&self, tau: f64, varmap: &VarMap) -> Result<(), candle_core::Error> {
        let data = varmap.data().lock().unwrap();

        let pairs = [
            ("q_head.0", "target_q_head.0"),
            ("q_head.1", "target_q_head.1"),
            ("v_head", "target_v_head"),
        ];

        for (src_prefix, tgt_prefix) in &pairs {
            for (name, var) in data.iter() {
                if name.starts_with(tgt_prefix) {
                    let src_name = name.replace(tgt_prefix, src_prefix);
                    if let Some(src_var) = data.get(&src_name) {
                        // target = tau * current + (1 - tau) * target
                        let new_val = (src_var.as_tensor() * tau)?
                            .add(&(var.as_tensor() * (1.0 - tau))?)?;
                        var.set(&new_val)?;
                    }
                }
            }
        }

        Ok(())
    }
}
