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
        let weights = Matrix::random(shape, -1.0..1.0);
        let biases = Matrix::random((1, shape.1), -1.0..1.0);

        let weight_gradient = Matrix::new(shape);
        let bias_gradient = Matrix::new((1, shape.1));

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
}

impl Layer for Dense {
    fn feed_forward(&mut self, input: Matrix<f32>) -> Matrix<f32> {
        self.inputs.push_back(input.clone());
        let output = self.predict(input);
        self.outputs.push_back(output.clone());

        output
    }

    fn backpropagate(&mut self, error: Matrix<f32>) -> Matrix<f32> {
        let prev_output = self.outputs.pop_back().unwrap();
        let prev_input = self.inputs.pop_back().unwrap();

        let mut output_error = Matrix::new(prev_input.shape());

        for i in 0..self.weights.cols() {
            let current_grad = self.activation.deactivate(prev_output[(0, i)]);
            let delta = current_grad * error[(0, i)];

            self.bias_gradient[(0, i)] += delta;

            for j in 0..self.weights.rows() {
                self.weight_gradient[(j, i)] += delta * prev_input[(0, j)];
                output_error[(0, j)] += delta * self.weights[(j, i)];
            }
        }

        output_error
    }

    fn predict(&mut self, input: Matrix<f32>) -> Matrix<f32> {
        let mut output = Matrix::new((input.shape().0, self.shape.1));
        for i in 0..input.rows() {
            for j in 0..self.shape.1 {
                let mut sum = self.biases[(0, j)];
                for k in 0..input.cols() {
                    sum += input[(i, k)] * self.weights[(k, j)];
                }

                output[(i, j)] = self.activation.activate(sum);
            }
        }

        output
    }

    fn update(&mut self, optimizer: &Optimizer) {
        let mut weights = self.weights.as_mut();
        let mut biases = self.biases.as_mut();

        optimizer.update(&mut weights, &self.weight_gradient.as_ref());
        optimizer.update(&mut biases, &self.bias_gradient.as_ref());

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
        let output = dense.feed_forward(input);

        assert_eq!(output.shape(), (1, 2));
    }
}
