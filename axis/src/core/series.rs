use std::ops::{Index, IndexMut};

use super::Scaler;

pub struct Series {
    pub name: &'static str,
    pub values: Vec<Scaler>,
}

impl Series {
    pub fn new(name: &'static str) -> Self {
        Series {
            name,
            values: Vec::new(),
        }
    }

    pub fn push<T: Into<Scaler>>(&mut self, value: T) {
        let new_value = value.into();

        if self.values.is_empty() {
            self.values.push(new_value);
        } else if std::mem::discriminant(&self.values[0]) == std::mem::discriminant(&new_value) {
            self.values.push(new_value);
        } else {
            panic!("Invalid value pushed to column: {}", self.name);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn iter(&self) -> std::slice::Iter<Scaler> {
        self.values.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<Scaler> {
        self.values.iter_mut()
    }
}

impl Index<usize> for Series {
    type Output = Scaler;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl IndexMut<usize> for Series {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<T> From<Vec<T>> for Series
where
    T: Into<Scaler>,
{
    fn from(values: Vec<T>) -> Self {
        let mut column = Series::new("default");
        for value in values {
            column.push(value.into());
        }
        column
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_column() {
        let column = Series::new("TEST");
        assert_eq!(column.values.len(), 0);
    }

    #[test]
    fn test_push() {
        let mut column = Series::new("TEST");
        column.push(Scaler::I32(42));
        column.push(45);
        assert_eq!(column.values.len(), 2);

        for (index, value) in column.values.iter().enumerate() {
            match index {
                0 => assert_eq!(*value, Scaler::I32(42)),
                1 => assert_eq!(*value, Scaler::I32(45)),
                _ => panic!("Invalid index: {}", index),
            }
        }
    }

    #[test]
    fn test_is_empty() {
        let column = Series::new("empty");
        assert_eq!(column.is_empty(), true);
    }

    #[test]
    fn test_from_vec() {
        let column: Series = vec![1, 2, 3].into();
        assert_eq!(column.len(), 3);
    }
}
