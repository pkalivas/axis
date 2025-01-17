#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Activation {
    Sigmoid,
    ReLU,
    LeakyReLU,
    Tanh,
    Softmax,
    Linear,
}

impl Activation {
    pub fn activate(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU => x.max(0.01 * x),
            Activation::Tanh => x.tanh(),
            Activation::Softmax => x.exp(),
            Activation::Linear => x,
        }
    }

    pub fn deactivate(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => x * (1.0 - x),
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            Activation::Tanh => 1.0 - x.powi(2),
            Activation::Softmax => x * (1.0 - x),
            Activation::Linear => 1.0,
        }
    }
}
