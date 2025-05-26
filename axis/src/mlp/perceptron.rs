use super::Layer;
use crate::{
    Matrix,
    math::{Loss, Optimizer},
};

pub struct MultiLayerPerceptron {
    layers: Vec<Box<dyn Layer>>,
}

impl MultiLayerPerceptron {
    pub fn new() -> Self {
        MultiLayerPerceptron { layers: Vec::new() }
    }

    pub fn layer<L: Layer + 'static>(mut self, layer: L) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn predict(&mut self, input: &Matrix<f32>) -> Matrix<f32> {
        let mut layer_outputs = Vec::new();
        let mut current_output = input;
        for layer in self.layers.iter_mut() {
            let output = layer.feed_forward(current_output);
            layer_outputs.push(output);
            current_output = layer_outputs.last().unwrap();
        }

        current_output.clone()
    }

    pub fn fit(
        &mut self,
        input: &[Matrix<f32>],
        target: &[Matrix<f32>],
        optimizer: &Optimizer,
        loss: &Loss,
    ) {
        for (input, target) in input.iter().zip(target.iter()) {
            let mut current_output = input;

            let mut layer_outputs = Vec::new();
            for layer in self.layers.iter_mut() {
                let output = layer.feed_forward(current_output);
                layer_outputs.push(output);
                current_output = layer_outputs.last().unwrap();
            }

            let mut error = Matrix::from(loss.apply(target, current_output));

            let layer_count = self.layers.len();
            for (idx, layer) in self.layers.iter_mut().rev().enumerate() {
                let prev_output = &layer_outputs[layer_count - idx - 1];
                let prev_input = if idx == 0 {
                    &layer_outputs[idx + 1]
                } else if idx == layer_count - 1 {
                    input
                } else {
                    &layer_outputs[idx]
                };

                error = layer.backpropagate(&error, prev_input, prev_output);
            }
        }

        for layer in self.layers.iter_mut() {
            layer.update(optimizer);
        }
    }
}
