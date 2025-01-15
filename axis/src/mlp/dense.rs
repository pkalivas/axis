use std::collections::VecDeque;

use crate::{math::Activation, Matrix};

#[derive(PartialEq, Clone)]
pub struct Dense {
    shape: (usize, usize),
    activation: Activation,
    inputs: VecDeque<Matrix<f32>>,
    outputs: VecDeque<Matrix<f32>>,
    weights: Matrix<f32>,
    biases: Matrix<f32>,
    weight_gradient: Matrix<f32>,
    bias_gradient: Matrix<f32>,
}

impl Dense {
    pub fn new(shape: (usize, usize), activation: Activation) -> Self {
        let weights = Matrix::random(shape);
        let biases = Matrix::random((1, shape.1));

        let weight_gradient = Matrix::from_shape(shape);
        let bias_gradient = Matrix::from_shape((1, shape.1));

        Dense {
            shape,
            activation,
            inputs: VecDeque::new(),
            outputs: VecDeque::new(),
            weights,
            biases,
            weight_gradient,
            bias_gradient,
        }
    }

    pub fn feed_forward(&mut self, input: Matrix<f32>) -> Matrix<f32> {
        self.inputs.push_back(input.clone());
        let output = self.predict(input);
        self.outputs.push_back(output.clone());

        output
    }

    pub fn predict(&mut self, input: Matrix<f32>) -> Matrix<f32> {
        let mut output = Matrix::from_shape((input.shape().0, self.shape.1));
        for i in 0..input.shape().0 {
            for j in 0..self.shape.1 {
                let mut sum = self.biases[(0, j)];
                for k in 0..input.shape().1 {
                    sum += input[(i, k)] * self.weights[(k, j)];
                }

                output[(i, j)] = self.activation.activate(sum);
            }
        }

        output
    }
}

#[cfg(test)]
mod test {
    use crate::domain::random_provider;

    use super::*;

    #[test]
    fn test_dense() {
        random_provider::set_seed(42);

        let mut dense = Dense::new((2, 2), Activation::ReLU);
        let input = Matrix::from(vec![vec![1.0, 2.0]]);
        let output = dense.feed_forward(input);

        assert_eq!(output.shape(), (1, 2));
    }
}
