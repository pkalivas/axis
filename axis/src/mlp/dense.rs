use super::Layer;
use crate::{
    Matrix,
    math::{Activation, Optimizer},
};

#[derive(PartialEq, Clone, Debug)]
pub struct Dense {
    shape: (usize, usize),
    activation: Activation,
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
            weights: Matrix::random(weight_shape, -1.0..1.0),
            biases: Matrix::random(bias_shape, -1.0..1.0),
            weight_gradient: Matrix::new(weight_shape),
            bias_gradient: Matrix::new(bias_shape),
        }
    }
}

impl Layer for Dense {
    fn feed_forward(&mut self, input: &Matrix<f32>) -> Matrix<f32> {
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

    fn backpropagate(
        &mut self,
        error: &Matrix<f32>,
        prev_input: &Matrix<f32>,
        prev_output: &Matrix<f32>,
    ) -> Matrix<f32> {
        let mut output_error = Matrix::new(prev_input.shape());

        for i in 0..self.shape.1 {
            let delta = self.activation.deactivate(prev_output[(0, i)]) * error[(0, i)];

            self.bias_gradient[(0, i)] += delta;

            for j in 0..self.shape.0 {
                self.weight_gradient[(i, j)] += delta * prev_input[(0, j)];
                output_error[(0, j)] += self.weights[(i, j)] * error[(0, i)];
            }
        }

        output_error
    }

    fn update(&mut self, optimizer: &Optimizer) {
        optimizer.update(&mut self.weights, &mut self.weight_gradient);
        optimizer.update(&mut self.biases, &mut self.bias_gradient);
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::domain::random_provider;

    #[test]
    fn test_dense() {
        random_provider::set_seed(42);

        let mut dense = Dense::new((2, 2), Activation::ReLU);
        let input = Matrix::from(vec![vec![1.0, 2.0]]);
        let output = dense.feed_forward(&input);

        assert_eq!(output.shape(), (1, 2));
    }
}
