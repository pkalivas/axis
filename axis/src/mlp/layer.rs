use crate::{Matrix, math::Optimizer};

pub trait Layer {
    fn feed_forward(&mut self, input: &Matrix<f32>) -> Matrix<f32>;
    fn backpropagate(
        &mut self,
        error: &Matrix<f32>,
        prev_input: &Matrix<f32>,
        prev_output: &Matrix<f32>,
    ) -> Matrix<f32>;
    fn update(&mut self, optimizer: &Optimizer);
}
