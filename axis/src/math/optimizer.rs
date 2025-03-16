use super::Matrix;

pub enum Optimizer {
    SGD(f32),
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
}

impl Optimizer {
    pub fn update(&self, weights: &mut Matrix<f32>, gradients: &Matrix<f32>) {
        match self {
            Optimizer::SGD(learning_rate) => {
                for i in 0..weights.rows() {
                    for j in 0..weights.cols() {
                        weights[(i, j)] -= learning_rate * gradients[(i, j)];
                    }
                }
            }
            Optimizer::Adam { .. } => {
                panic!("Adam optimizer not implemented yet");
            }
        }
    }
}
