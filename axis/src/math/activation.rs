const MAX: f32 = 1e10;
const MIN: f32 = -1e10;

fn clamp(x: f32) -> f32 {
    if x.is_nan() {
        return 0.0;
    }

    x.clamp(MIN, MAX)
}

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
            Activation::Sigmoid => clamp(1.0 / (1.0 + (-x).exp())),
            Activation::ReLU => clamp(x.max(0.0)),
            Activation::LeakyReLU => clamp(x.max(0.01 * x)),
            Activation::Tanh => clamp(x.tanh()),
            Activation::Softmax => {
                let exp_x = x.exp();
                clamp(exp_x / (1.0 + exp_x))
            }
            Activation::Linear => clamp(x),
        }
    }

    pub fn deactivate(&self, x: f32) -> f32 {
        match self {
            Activation::Sigmoid => clamp(x * (1.0 - x)),
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
            Activation::Tanh => clamp(1.0 - x.powi(2)),
            Activation::Softmax => {
                let exp_x = x.exp();
                clamp(exp_x * (1.0 - exp_x))
            }
            Activation::Linear => 1.0,
        }
    }
}
