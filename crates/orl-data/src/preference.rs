#[derive(Debug, Clone)]
pub struct PreferencePair {
    pub chosen_idx: usize,
    pub rejected_idx: usize,
    pub chosen_prompt_ids: Vec<u32>,
    pub chosen_response_ids: Vec<u32>,
    pub rejected_prompt_ids: Vec<u32>,
    pub rejected_response_ids: Vec<u32>,
    pub reward_chosen: f64,
    pub reward_rejected: f64,
}

impl PreferencePair {
    pub fn reward_margin(&self) -> f64 {
        self.reward_chosen - self.reward_rejected
    }
}
