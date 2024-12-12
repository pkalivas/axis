use axis::*;

fn main() {
    let mut frame = DataFrame::new();
    let column = Series::new("column1");
    frame.add_series(column);
    assert_eq!(frame["column1"].len(), 0);

    let mut frame = DataFrame::new();
    let column = Series::new("add");
    frame.add_series(column);
    assert_eq!(frame.columns.len(), 1);

    let frame = DataFrame::new();
    assert_eq!(frame.columns.len(), 0);
}
