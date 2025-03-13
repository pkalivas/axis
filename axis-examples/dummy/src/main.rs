use axis::Matrix;

fn main() {
    let matrix = Matrix::arange(1..10, 1).reshape((3, 3)).transpose();

    println!("{:?}", matrix);
}
