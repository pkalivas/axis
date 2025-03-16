pub mod dense;
pub mod perceptron;

use crate::{Matrix, math::Optimizer};

pub use dense::Dense;
pub use perceptron::MultiLayerPerceptron;

pub trait Layer {
    fn predict(&mut self, input: &Matrix<f32>) -> Matrix<f32>;
    fn backpropagate(&mut self, error: &Matrix<f32>) -> Matrix<f32>;
    fn feed_forward(&mut self, input: &Matrix<f32>) -> Matrix<f32>;
    fn update(&mut self, optimizer: &Optimizer);
}
