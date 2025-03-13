use crate::math::Loss;

use super::Layer;

#[allow(dead_code)]
pub struct MultiLayerPerceptron {
    layers: Vec<Box<dyn Layer>>,
    loss: Loss,
}

impl MultiLayerPerceptron {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        MultiLayerPerceptron {
            layers,
            loss: Loss::MSE,
        }
    }
}
