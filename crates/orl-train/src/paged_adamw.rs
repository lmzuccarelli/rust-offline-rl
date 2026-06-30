use candle_core::{Device, Result, Tensor, Var};

#[derive(Clone, Debug)]
pub struct ParamsPagedAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for ParamsPagedAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

struct VarState {
    var: Var,
    first_moment: Tensor,
    second_moment: Tensor,
    device: Device,
}

/// AdamW optimizer that pages first/second moment states to CPU memory,
/// transferring them to the compute device only during the update step.
/// This trades PCIe bandwidth for GPU memory — critical when fine-tuning
/// large models on limited VRAM.
pub struct PagedAdamW {
    vars: Vec<VarState>,
    step_t: usize,
    params: ParamsPagedAdamW,
}

impl PagedAdamW {
    pub fn new(vars: Vec<Var>, params: ParamsPagedAdamW) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device().clone();
                let first_moment = Tensor::zeros(shape, dtype, &Device::Cpu)?;
                let second_moment = Tensor::zeros(shape, dtype, &Device::Cpu)?;
                Ok(VarState {
                    var,
                    first_moment,
                    second_moment,
                    device,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            params,
            step_t: 0,
        })
    }

    pub fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    pub fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    pub fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));

        for state in self.vars.iter_mut() {
            if let Some(g) = grads.get(&state.var) {
                let dev = &state.device;

                // Page in: move moments from CPU to compute device
                let m = state.first_moment.to_device(dev)?;
                let v = state.second_moment.to_device(dev)?;

                let next_m = ((&m * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((&v * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (state.var.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                state.var.set(&next_theta)?;

                // Page out: move moments back to CPU
                state.first_moment = next_m.to_device(&Device::Cpu)?;
                state.second_moment = next_v.to_device(&Device::Cpu)?;
            }
        }
        Ok(())
    }

    pub fn params(&self) -> &ParamsPagedAdamW {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_paged_adamw_step() -> Result<()> {
        let device = Device::Cpu;
        let x = Var::new(&[3f32, 1.0], &device)?;
        let params = ParamsPagedAdamW {
            lr: 0.1,
            ..Default::default()
        };
        let mut optimizer = PagedAdamW::new(vec![x.clone()], params)?;

        let loss = x.as_tensor().sqr()?.sum_all()?;
        optimizer.backward_step(&loss)?;

        let updated = x.as_tensor().to_vec1::<f32>()?;
        assert!(updated[0] < 3.0, "x[0] should decrease: {}", updated[0]);
        assert!(updated[1] < 1.0, "x[1] should decrease: {}", updated[1]);
        Ok(())
    }

    #[test]
    fn test_paged_adamw_moments_on_cpu() -> Result<()> {
        let device = Device::Cpu;
        let x = Var::new(&[1f32, 2.0], &device)?;
        let params = ParamsPagedAdamW::default();
        let optimizer = PagedAdamW::new(vec![x], params)?;

        for state in &optimizer.vars {
            assert!(
                matches!(state.first_moment.device(), Device::Cpu),
                "first moment should be on CPU"
            );
            assert!(
                matches!(state.second_moment.device(), Device::Cpu),
                "second moment should be on CPU"
            );
        }
        Ok(())
    }

    #[test]
    fn test_set_learning_rate() {
        let params = ParamsPagedAdamW::default();
        let mut optimizer = PagedAdamW::new(vec![], params).unwrap();
        assert!((optimizer.learning_rate() - 0.001).abs() < 1e-10);
        optimizer.set_learning_rate(0.01);
        assert!((optimizer.learning_rate() - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_convergence() -> Result<()> {
        let device = Device::Cpu;
        let x = Var::new(&[5.0f32], &device)?;
        let params = ParamsPagedAdamW {
            lr: 0.1,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = PagedAdamW::new(vec![x.clone()], params)?;

        for _ in 0..100 {
            let loss = x.as_tensor().sqr()?.sum_all()?;
            optimizer.backward_step(&loss)?;
        }

        let val = x.as_tensor().to_vec1::<f32>()?[0];
        assert!(
            val.abs() < 0.1,
            "should converge near 0 after 100 steps: {}",
            val
        );
        Ok(())
    }

    #[test]
    fn test_weight_decay() -> Result<()> {
        let device = Device::Cpu;

        let x_wd = Var::new(&[2.0f32], &device)?;
        let params_wd = ParamsPagedAdamW {
            lr: 0.01,
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut opt_wd = PagedAdamW::new(vec![x_wd.clone()], params_wd)?;

        let x_no = Var::new(&[2.0f32], &device)?;
        let params_no = ParamsPagedAdamW {
            lr: 0.01,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut opt_no = PagedAdamW::new(vec![x_no.clone()], params_no)?;

        for _ in 0..10 {
            let loss_wd = x_wd.as_tensor().sqr()?.sum_all()?;
            opt_wd.backward_step(&loss_wd)?;
            let loss_no = x_no.as_tensor().sqr()?.sum_all()?;
            opt_no.backward_step(&loss_no)?;
        }

        let val_wd = x_wd.as_tensor().to_vec1::<f32>()?[0];
        let val_no = x_no.as_tensor().to_vec1::<f32>()?[0];
        assert!(
            val_wd.abs() < val_no.abs(),
            "weight decay should shrink params faster: wd={}, no={}",
            val_wd,
            val_no
        );
        Ok(())
    }
}
