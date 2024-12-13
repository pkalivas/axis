use std::{
    collections::BTreeMap,
    ops::{Index, IndexMut},
};

use crate::core::series::Series;

pub struct DataFrame {
    pub series: BTreeMap<&'static str, Series>,
    column_order: Vec<&'static str>,
}

impl DataFrame {
    pub fn new() -> Self {
        DataFrame {
            series: BTreeMap::new(),
            column_order: Vec::new(),
        }
    }

    pub fn push<T: Into<Series>>(&mut self, series: T) {
        let new_series = series.into();
        self.column_order.push(new_series.name);
        self.series.insert(new_series.name, new_series);
    }

    pub fn set<T: Into<Series>>(&mut self, name: &'static str, series: T) {
        let mut new_series = series.into();
        new_series.name = name;

        if !self.series.contains_key(name) {
            self.column_order.push(name);
        }

        self.series.insert(name, new_series);
    }

    pub fn len(&self) -> usize {
        self.series.len()
    }

    pub fn is_empty(&self) -> bool {
        self.series.is_empty()
    }
}

impl Index<&'static str> for DataFrame {
    type Output = Series;

    fn index(&self, name: &'static str) -> &Self::Output {
        self.series.get(name).expect("Column not found")
    }
}

impl IndexMut<&'static str> for DataFrame {
    fn index_mut(&mut self, name: &'static str) -> &mut Self::Output {
        self.series.get_mut(name).expect("Column not found")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_frame() {
        let frame = DataFrame::new();
        assert_eq!(frame.len(), 0);
    }

    #[test]
    fn test_add_column() {
        let mut frame = DataFrame::new();
        let column = Series::new("add");
        frame.push(column);
        assert_eq!(frame.len(), 1);
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
