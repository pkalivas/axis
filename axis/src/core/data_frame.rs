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

    pub fn push<T: Into<Series>>(&mut self, series: T) {
        self.columns.push(series.into());
    }

    pub fn set<T: Into<Series>>(&mut self, name: &'static str, series: T) {
        let mut new_series = series.into();
        new_series.name = name;

        if let Some(column) = self.columns.iter_mut().find(|column| column.name == name) {
            *column = new_series;
        } else {
            self.push(new_series);
        }
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<Series> {
        self.columns.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Series> {
        self.columns.iter_mut()
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
            .expect("Column not found")
    }
}

impl IndexMut<&'static str> for DataFrame {
    fn index_mut(&mut self, name: &'static str) -> &mut Self::Output {
        self.columns
            .iter_mut()
            .find(|column| column.name == name)
            .expect("Column not found")
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
        frame.push(column);
        assert_eq!(frame.columns.len(), 1);
    }

    #[test]
    fn test_index() {
        let mut frame = DataFrame::new();
        let column = Series::new("column1");
        frame.push(column);
        assert_eq!(frame["column1"].len(), 0);
    }

    #[test]
    fn test_index_mut() {
        let mut frame = DataFrame::new();
        let column = Series::new("column1");
        frame.push(column);
        frame["column1"].push(42);
        assert_eq!(frame["column1"].len(), 1);
    }
}
