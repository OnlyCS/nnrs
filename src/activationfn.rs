use serde::{Deserialize, Serialize};

/// Activation function for a neuron.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
pub enum ActivationFn {
    /// |x| x.max(0.0)
    ReLU,

    /// |x| 1.0 / (1.0 + (-x).exp())
    Sigmoid,

    /// |x| x.tanh()
    Tanh,

    /// |x| x, aka do nothing
    Linear,
}

impl ActivationFn {
    pub(crate) fn run(&self, x: f64) -> f64 {
        match self {
            ActivationFn::ReLU => x.max(0.0),
            ActivationFn::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            ActivationFn::Tanh => x.tanh(),
            ActivationFn::Linear => x,
        }
    }
}
