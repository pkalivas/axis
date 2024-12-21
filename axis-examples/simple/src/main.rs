use axis::*;

fn main() {
    let mut frame = DataFrame::new();

    frame.set("A", vec![1, 2, 3, 4, 5]);
    frame.set("B", vec!["a", "b", "c", "d", "e"]);
    frame.set("C", vec![1.1, 2.2, 3.3, 4.4, 5.5]);
    frame.set("D", vec![true, false, true]);

    for series in frame.iter() {
        println!("Column: {}", series.name);
        for value in series.iter() {
            println!("{:?}", value);
        }
    }
}
