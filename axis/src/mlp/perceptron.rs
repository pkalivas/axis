use super::Layer;

#[allow(dead_code)]
pub struct MultiLayerPerceptron {
    layers: Vec<Box<dyn Layer>>,
}

impl MultiLayerPerceptron {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        MultiLayerPerceptron { layers }
    }
}
