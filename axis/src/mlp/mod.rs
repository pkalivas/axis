pub mod dense;
pub mod perceptron;

pub use dense::Dense;
pub use perceptron::MultiLayerPerceptron;

use crate::Matrix;

pub trait Layer {
    fn predict(&mut self, input: Matrix<f32>) -> Matrix<f32>;
    fn backpropagate(&mut self, error: Matrix<f32>) -> Matrix<f32>;
    fn feed_forward(&mut self, input: Matrix<f32>) -> Matrix<f32>;
}
