use std::ops::{Index, IndexMut};

use crate::core::series::Series;

pub struct DataFrame {
    pub columns: Vec<Series>,
}

impl DataFrame {
    pub fn new() -> Self {
        DataFrame {
            columns: Vec::new(),
        }
    }

    pub fn add_series(&mut self, series: Series) {
        self.columns.push(series);
    }
}

impl Index<&'static str> for DataFrame {
    type Output = Series;

    fn index(&self, name: &'static str) -> &Self::Output {
        if !self.columns.iter().any(|column| column.name == name) {
            panic!("Column not found: {}", name);
        }

        self.columns
            .iter()
            .find(|column| column.name == name)
            .unwrap()
    }
}

impl IndexMut<&'static str> for DataFrame {
    fn index_mut(&mut self, name: &'static str) -> &mut Self::Output {
        if !self.columns.iter().any(|column| column.name == name) {
            panic!("Column not found: {}", name);
        }

        self.columns
            .iter_mut()
            .find(|column| column.name == name)
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_frame() {
        let frame = DataFrame::new();
        assert_eq!(frame.columns.len(), 0);
    }

    #[test]
    fn test_add_column() {
        let mut frame = DataFrame::new();
        let column = Series::new("add");
        frame.add_series(column);
        assert_eq!(frame.columns.len(), 1);
    }

    #[test]
    fn test_index() {
        let mut frame = DataFrame::new();
        let column = Series::new("column1");
        frame.add_series(column);
        assert_eq!(frame["column1"].len(), 0);
    }

    #[test]
    fn test_index_mut() {
        let mut frame = DataFrame::new();
        let column = Series::new("column1");
        frame.add_series(column);
        frame["column1"].push(42);
        assert_eq!(frame["column1"].len(), 1);
    }
}
