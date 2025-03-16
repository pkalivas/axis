use super::Layer;
use crate::{
    Matrix,
    math::{Loss, Optimizer},
};

pub struct MultiLayerPerceptron {
    layers: Vec<Box<dyn Layer>>,
    loss: Loss,
}

impl MultiLayerPerceptron {
    pub fn new() -> Self {
        MultiLayerPerceptron {
            layers: Vec::new(),
            loss: Loss::MSE,
        }
    }

    pub fn layer<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn predict(&mut self, input: Matrix<f32>) -> Matrix<f32> {
        let mut output = input;

        for layer in self.layers.iter_mut() {
            output = layer.predict(&output);
        }

        output
    }

    pub fn train(&mut self, input: &[Matrix<f32>], target: &[Matrix<f32>], optimizer: &Optimizer) {
        for (input, target) in input.iter().zip(target.iter()) {
            let mut current_output = input;

            let mut layer_outputs = Vec::new();
            for layer in self.layers.iter_mut() {
                let output = layer.feed_forward(current_output);
                layer_outputs.push(output);
                current_output = layer_outputs.last().unwrap();
            }

            let mut error = Matrix::from(self.loss.apply(target.as_ref(), current_output.as_ref()));

            for layer in self.layers.iter_mut().rev() {
                error = layer.backpropagate(&error);
                layer.update(&optimizer);
            }
        }
    }
}
