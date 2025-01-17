pub enum Optimizer {
    SGD {
        learning_rate: f32,
    },

    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
}

impl Optimizer {
    pub fn update(&self, weights: &mut Vec<f32>, gradients: &Vec<f32>) {
        match self {
            Optimizer::SGD { learning_rate } => {
                for (weight, gradient) in weights.iter_mut().zip(gradients.iter()) {
                    *weight -= learning_rate * gradient;
                }
            }
            Optimizer::Adam { .. } => {
                panic!("Adam optimizer not implemented yet");
            }
        }
    }
}
