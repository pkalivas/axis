#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Loss {
    MSE,
    CrossEntropy,
    BinaryCrossEntropy,
    Difference,
    Hinge,
    Huber,
}

impl Loss {
    pub fn apply<T: AsRef<[f32]>>(&self, y_true: &T, y_pred: &T) -> Vec<f32> {
        match self {
            Loss::MSE => y_true
                .as_ref()
                .iter()
                .zip(y_pred.as_ref().iter())
                .map(|(y_t, y_p)| 2.0 * (y_p - y_t)) // d/dy (y - t)^2 = 2(y - t)
                .collect(),
            Loss::CrossEntropy => y_true
                .as_ref()
                .iter()
                .zip(y_pred.as_ref().iter())
                .map(|(y_t, y_p)| -y_t * y_p.ln() - (1.0 - y_t) * (1.0 - y_p).ln())
                .collect(),
            Loss::BinaryCrossEntropy => y_true
                .as_ref()
                .iter()
                .zip(y_pred.as_ref().iter())
                .map(|(y_t, y_p)| -y_t * y_p.ln() - (1.0 - y_t) * (1.0 - y_p).ln())
                .collect(),
            Loss::Difference => y_true
                .as_ref()
                .iter()
                .zip(y_pred.as_ref().iter())
                .map(|(y_t, y_p)| (y_t - y_p).abs())
                .collect(),
            Loss::Hinge => y_true
                .as_ref()
                .iter()
                .zip(y_pred.as_ref().iter())
                .map(|(y_t, y_p)| (1.0 - y_t * y_p).max(0.0))
                .collect(),
            Loss::Huber => {
                let delta = 1.0;
                y_true
                    .as_ref()
                    .iter()
                    .zip(y_pred.as_ref().iter())
                    .map(|(y_t, y_p)| {
                        let error = (y_t - y_p).abs();
                        if error <= delta {
                            0.5 * error.powi(2)
                        } else {
                            delta * (error - 0.5 * delta)
                        }
                    })
                    .collect()
            }
        }
    }
}
