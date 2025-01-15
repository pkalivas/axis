use std::ops::{Index, IndexMut};

use rand::{distributions::Standard, prelude::Distribution};

use crate::domain::random_provider;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T> {
    data: Vec<T>,
    shape: (usize, usize),
}

impl<T> Matrix<T> {
    pub fn from_shape(shape: (usize, usize)) -> Self
    where
        T: Default + Clone,
    {
        Matrix {
            data: vec![Default::default(); shape.0 * shape.1],
            shape,
        }
    }

    pub fn from_vec(shape: (usize, usize), data: Vec<T>) -> Self {
        if data.len() != shape.0 * shape.1 {
            panic!("Data length does not match shape");
        }

        Matrix { data, shape }
    }

    pub fn random(shape: (usize, usize)) -> Self
    where
        T: rand::distributions::uniform::SampleUniform,
        Standard: Distribution<T>,
    {
        let data = (0..shape.0 * shape.1)
            .map(|_| random_provider::random::<T>())
            .collect::<Vec<T>>();

        Matrix { data, shape }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut()
    }
}

impl<T> Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        if index.0 >= self.shape.0 || index.1 >= self.shape.1 {
            panic!("Index out of bounds");
        }

        &self.data[index.0 * self.shape.1 + index.1]
    }
}

impl<T> IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        if index.0 >= self.shape.0 || index.1 >= self.shape.1 {
            panic!("Index out of bounds");
        }

        &mut self.data[index.0 * self.shape.1 + index.1]
    }
}

impl<T> From<Vec<Vec<T>>> for Matrix<T>
where
    T: Clone,
{
    fn from(data: Vec<Vec<T>>) -> Self {
        let shape = (data.len(), data[0].len());
        let data = data.into_iter().flatten().collect();

        Matrix { data, shape }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_matrix() {
        let mut matrix = Matrix::from_shape((2, 2));

        matrix[(0, 0)] = 1;
        matrix[(0, 1)] = 2;
        matrix[(1, 0)] = 3;
        matrix[(1, 1)] = 4;

        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
    }
}
