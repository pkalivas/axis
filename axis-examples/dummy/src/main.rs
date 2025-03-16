use axis::{
    Matrix,
    domain::random_provider,
    math::{Activation, Loss, Optimizer},
    mlp::{Dense, MultiLayerPerceptron},
};

fn main() {
    random_provider::set_seed(4992);
    let xor = xor()
        .into_iter()
        .map(|vals| {
            let (input, output) = vals;
            (Matrix::from(input), Matrix::from(output))
        })
        .collect::<Vec<(Matrix<f32>, Matrix<f32>)>>();

    let features = xor
        .iter()
        .map(|(input, _)| input.clone())
        .collect::<Vec<Matrix<f32>>>();
    let targets = xor
        .iter()
        .map(|(_, output)| output.clone())
        .collect::<Vec<Matrix<f32>>>();

    let mut mlp = MultiLayerPerceptron::new()
        .layer(Dense::new((2, 16), Activation::Sigmoid))
        .layer(Dense::new((16, 16), Activation::ReLU))
        .layer(Dense::new((16, 1), Activation::Sigmoid));

    let loss = Loss::MSE;
    let optimizer = Optimizer::SGD(0.01);

    let start_time = std::time::Instant::now();
    for _ in 0..1500 {
        mlp.fit(&features, &targets, &optimizer);
    }

    let elapsed_time = start_time.elapsed();
    println!("Time taken: {:?}", elapsed_time);

    for (input, output) in xor.iter() {
        let prediction = mlp.predict(input.clone());

        println!("Output: {:?}", prediction);
        println!("Expected: {:?}", output);
        println!(
            "Loss: {:?}",
            loss.apply(output.as_ref(), prediction.as_ref())
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
