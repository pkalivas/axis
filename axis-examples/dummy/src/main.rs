use axis::{
    Matrix,
    domain::random_provider,
    math::{Activation, Loss, Optimizer},
    mlp::{Dense, Layer},
};

fn main() {
    random_provider::set_seed(402);
    let xor = xor()
        .into_iter()
        .map(|vals| {
            let (input, output) = vals;
            (Matrix::from(input), Matrix::from(output))
        })
        .collect::<Vec<(Matrix<f32>, Matrix<f32>)>>();

    let mut layer_one = Dense::new((2, 16), Activation::Sigmoid);
    let mut layer_two = Dense::new((16, 16), Activation::Sigmoid);
    let mut layer_three = Dense::new((16, 16), Activation::Sigmoid);
    let mut layer_four = Dense::new((16, 1), Activation::Sigmoid);

    let loss = Loss::MSE;
    let optimizer = Optimizer::SGD {
        learning_rate: 0.01,
    };

    for _ in 0..1500 {
        for (input, output) in xor.iter() {
            let input = input.clone();
            let output = output.clone();

            let layer_one_output = layer_one.feed_forward(input);
            let layer_two_output = layer_two.feed_forward(layer_one_output);
            let layer_three_output = layer_three.feed_forward(layer_two_output);
            let layer_four_output = layer_four.feed_forward(layer_three_output);

            let error = loss.apply(output.as_ref(), layer_four_output.as_ref());

            let layer_four_error = layer_four.backpropagate(Matrix::from(error));
            let layer_three_error = layer_three.backpropagate(layer_four_error.clone());
            let layer_two_error = layer_two.backpropagate(layer_three_error.clone());
            let _ = layer_one.backpropagate(layer_two_error.clone());

            layer_four.update(&optimizer);
            layer_three.update(&optimizer);
            layer_two.update(&optimizer);
            layer_one.update(&optimizer);
        }
    }

    for (input, output) in xor.iter() {
        let input = input.clone();
        let layer_one_output = layer_one.predict(input);
        let layer_two_output = layer_two.predict(layer_one_output);
        let layer_three_output = layer_three.predict(layer_two_output);
        let layer_four_output = layer_four.predict(layer_three_output);

        println!("Output: {:?}", layer_four_output);
        println!("Expected: {:?}", output);
        println!(
            "Loss: {:?}",
            loss.apply(output.as_ref(), layer_four_output.as_ref())
        );
        println!("----------------------------------");
        println!();
    }
}

fn xor() -> Vec<(Vec<f32>, Vec<f32>)> {
    vec![
        (vec![0.0, 0.0], vec![0.0]),
        (vec![0.0, 1.0], vec![1.0]),
        (vec![1.0, 0.0], vec![1.0]),
        (vec![1.0, 1.0], vec![0.0]),
    ]
}
