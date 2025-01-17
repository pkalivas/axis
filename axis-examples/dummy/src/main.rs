use axis::Matrix;

fn main() {
    let matrix = Matrix::arange(0.0, 10.0, 1.0).reshape((2, 5));
    let other = Matrix::arange(0.0, 10.0, 1.0).reshape((2, 5));

    let result = matrix + other;
    println!("{:?}", result);
}
