use super::Layer;
use crate::{
    Matrix,
    math::{Activation, Optimizer},
};
use std::collections::VecDeque;

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
        let weight_shape = (shape.1, shape.0);
        let bias_shape = (1, shape.1);

        Dense {
            shape,
            activation,
            inputs: VecDeque::new(),
            outputs: VecDeque::new(),
            weights: Matrix::random(weight_shape, -1.0..1.0),
            biases: Matrix::random(bias_shape, -1.0..1.0),
            weight_gradient: Matrix::new(weight_shape),
            bias_gradient: Matrix::new(bias_shape),
        }
    }
}

impl Layer for Dense {
    fn feed_forward(&mut self, input: &Matrix<f32>) -> Matrix<f32> {
        self.inputs.push_back(input.clone());
        let output = self.predict(&input);
        self.outputs.push_back(output.clone());

        output
    }

    fn backpropagate(&mut self, error: &Matrix<f32>) -> Matrix<f32> {
        let prev_output = self.outputs.pop_back().unwrap();
        let prev_input = self.inputs.pop_back().unwrap();

        let mut output_error = Matrix::new(prev_input.shape());

        for i in 0..self.shape.1 {
            let current_gradient = self.activation.deactivate(prev_output[(0, i)]);
            let delta = current_gradient * error[(0, i)];

            self.bias_gradient[(0, i)] += delta;

            for j in 0..self.shape.0 {
                self.weight_gradient[(i, j)] += delta * prev_input[(0, j)];
                output_error[(0, j)] += self.weights[(i, j)] * error[(0, i)];
            }
        }

        output_error
    }

    fn predict(&mut self, input: &Matrix<f32>) -> Matrix<f32> {
        let mut output = Matrix::new((1, self.shape.1));
        for i in 0..self.shape.1 {
            let mut sum = self.biases[(0, i)];
            for j in 0..self.shape.0 {
                sum += input[(0, j)] * self.weights[(i, j)];
            }

            output[(0, i)] = self.activation.activate(sum);
        }

        output
    }

    fn update(&mut self, optimizer: &Optimizer) {
        optimizer.update(&mut self.weights, &self.weight_gradient);
        optimizer.update(&mut self.biases, &self.bias_gradient);
        self.weight_gradient.fill(0.0);
        self.bias_gradient.fill(0.0);
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
        let output = dense.feed_forward(&input);

        assert_eq!(output.shape(), (1, 2));
    }

    #[test]
    fn test_dense_backpropagation() {
        random_provider::set_seed(42);

        let mut dense = Dense::new((2, 2), Activation::ReLU);
        let input = Matrix::from(vec![vec![1.0, 2.0]]);
        let _ = dense.feed_forward(&input);
        let error = Matrix::from(vec![vec![0.5, 0.5]]);
        let output_error = dense.backpropagate(&error);

        assert_eq!(output_error.shape(), (1, 2));
    }
}
