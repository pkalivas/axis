use std::fmt::Debug;
use std::ops::{Add, Div, Index, IndexMut, Mul, Range, Sub};

use rand::{
    distributions::{uniform::SampleUniform, Standard},
    prelude::Distribution,
};

use crate::domain::random_provider;

#[derive(Clone, PartialEq)]
pub struct Matrix<T> {
    data: Vec<T>,
    shape: (usize, usize),
}

impl<T> Matrix<T> {
    pub fn new(shape: (usize, usize)) -> Self
    where
        T: Default + Clone,
    {
        Matrix {
            data: vec![Default::default(); shape.0 * shape.1],
            shape,
        }
    }

    pub fn zeros(shape: (usize, usize)) -> Self
    where
        T: Default + Clone,
    {
        Matrix {
            data: vec![Default::default(); shape.0 * shape.1],
            shape,
        }
    }

    pub fn ones(shape: (usize, usize)) -> Self
    where
        T: Default + Clone + From<u8>,
    {
        Matrix {
            data: vec![T::from(1); shape.0 * shape.1],
            shape,
        }
    }

    pub fn arange(start: T, end: T, step: T) -> Self
    where
        T: Default + Clone + Add<Output = T> + PartialOrd + From<u8>,
    {
        let mut data = Vec::new();
        let mut current = start.clone();
        while current < end {
            data.push(current.clone());
            current = current + step.clone();
        }

        let shape = (1, data.len());
        Matrix { data, shape }
    }

    pub fn random(shape: (usize, usize), range: Range<T>) -> Self
    where
        T: SampleUniform + Clone + PartialOrd,
        Standard: Distribution<T>,
    {
        let data = (0..shape.0 * shape.1)
            .map(|_| random_provider::gen_range(range.clone()))
            .collect::<Vec<T>>();

        Matrix { data, shape }
    }

    pub fn reshape(self, shape: (usize, usize)) -> Self
    where
        T: Default + Clone,
    {
        if self.data.len() != shape.0 * shape.1 {
            panic!("Matrix dimensions do not match");
        }

        Matrix {
            data: self.data,
            shape,
        }
    }

    pub fn transpose(&self) -> Self
    where
        T: Default + Clone,
    {
        let mut data = Vec::with_capacity(self.data.len());
        for j in 0..self.shape.1 {
            for i in 0..self.shape.0 {
                data.push(self[(i, j)].clone());
            }
        }

        Matrix {
            data,
            shape: (self.shape.1, self.shape.0),
        }
    }

    pub fn flatten(self) -> Self
    where
        T: Default + Clone,
    {
        let len = self.data.len();
        Matrix {
            data: self.data,
            shape: (1, len),
        }
    }

    pub fn rows(&self) -> usize {
        self.shape.0
    }

    pub fn cols(&self) -> usize {
        self.shape.1
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

impl<T> Matrix<T>
where
    T: Default + Clone + Add<Output = T> + Mul<Output = T>,
{
    pub fn dot(&self, other: &Matrix<T>) -> Matrix<T> {
        if self.shape.1 != other.shape.0 {
            panic!("Matrix dimensions do not match");
        }

        let mut result = Matrix::new((self.shape.0, other.shape.1));
        for i in 0..self.shape.0 {
            for j in 0..other.shape.1 {
                let mut sum = T::default();
                for k in 0..self.shape.1 {
                    sum = sum + self[(i, k)].clone() * other[(k, j)].clone();
                }

                result[(i, j)] = sum;
            }
        }

        result
    }
}

impl<T> Add for Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            panic!("Matrix dimensions do not match");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Add<T> for Matrix<T>
where
    T: Add<Output = T> + Clone,
{
    type Output = Self;

    fn add(self, scaler: T) -> Self::Output {
        let data = self
            .data
            .iter()
            .map(|a| a.clone() + scaler.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Mul for Matrix<T>
where
    T: Add<Output = T> + Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            panic!("Matrix dimensions do not match");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() * b.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Mul<T> for Matrix<T>
where
    T: Mul<Output = T> + Clone,
{
    type Output = Self;

    fn mul(self, scaler: T) -> Self::Output {
        let data = self
            .data
            .iter()
            .map(|a| a.clone() * scaler.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Div for Matrix<T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            panic!("Matrix dimensions do not match");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() / b.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Div<T> for Matrix<T>
where
    T: Div<Output = T> + Clone,
{
    type Output = Self;

    fn div(self, scaler: T) -> Self::Output {
        let data = self
            .data
            .iter()
            .map(|a| a.clone() / scaler.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Sub for Matrix<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        if self.shape != other.shape {
            panic!("Matrix dimensions do not match");
        }

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a.clone() - b.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
    }
}

impl<T> Sub<T> for Matrix<T>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Self;

    fn sub(self, scaler: T) -> Self::Output {
        let data = self
            .data
            .iter()
            .map(|a| a.clone() - scaler.clone())
            .collect();

        Matrix {
            data,
            shape: self.shape,
        }
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

impl<T> From<Vec<T>> for Matrix<T>
where
    T: Clone,
{
    fn from(data: Vec<T>) -> Self {
        let shape = (1, data.len());

        Matrix { data, shape }
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

impl<T> Debug for Matrix<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = String::new();
        for i in 0..self.shape.0 {
            s.push_str("[");
            for j in 0..self.shape.1 {
                s.push_str(&format!("{:?}", self[(i, j)]));
                if j < self.shape.1 - 1 {
                    s.push_str(", ");
                }
            }
            s.push_str("]");
            if i < self.shape.0 - 1 {
                s.push_str(",\n");
            }
        }

        write!(f, "{}", s)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn test_matrix() {
        let mut matrix = Matrix::new((2, 2));

        matrix[(0, 0)] = 1;
        matrix[(0, 1)] = 2;
        matrix[(1, 0)] = 3;
        matrix[(1, 1)] = 4;

        assert_eq!(matrix[(0, 0)], 1);
        assert_eq!(matrix[(0, 1)], 2);
        assert_eq!(matrix[(1, 0)], 3);
        assert_eq!(matrix[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_shape() {
        let matrix = Matrix::<f32>::new((2, 2));
        assert_eq!(matrix.shape(), (2, 2));
    }

    #[test]
    fn test_matrix_arange() {
        let matrix = Matrix::arange(0, 5, 1);

        assert_eq!(matrix.shape(), (1, 5));
        assert_eq!(matrix[(0, 0)], 0);
        assert_eq!(matrix[(0, 1)], 1);
        assert_eq!(matrix[(0, 2)], 2);
        assert_eq!(matrix[(0, 3)], 3);
        assert_eq!(matrix[(0, 4)], 4);
    }

    #[test]
    fn test_matrix_add_matrix() {
        let matrix1 = Matrix::arange(0, 4, 1).reshape((2, 2));
        let matrix2 = Matrix::arange(4, 8, 1).reshape((2, 2));

        let result = matrix1 + matrix2;

        assert_eq!(result[(0, 0)], 4);
        assert_eq!(result[(0, 1)], 6);
        assert_eq!(result[(1, 0)], 8);
        assert_eq!(result[(1, 1)], 10);
    }

    #[test]
    fn test_matrix_add_scaler() {
        let result = Matrix::arange(0, 4, 1).reshape((2, 2)) + 1;

        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(0, 1)], 2);
        assert_eq!(result[(1, 0)], 3);
        assert_eq!(result[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_mul_matrix() {
        let matrix1 = Matrix::arange(1, 5, 1).reshape((2, 2));
        let matrix2 = Matrix::arange(5, 9, 1).reshape((2, 2));

        let result = matrix1 * matrix2;

        assert_eq!(result[(0, 0)], 5);
        assert_eq!(result[(0, 1)], 12);
        assert_eq!(result[(1, 0)], 21);
        assert_eq!(result[(1, 1)], 32);
    }

    #[test]
    fn test_matrix_mul_scaler() {
        let result = Matrix::arange(1, 5, 1).reshape((2, 2)) * 2;

        assert_eq!(result[(0, 0)], 2);
        assert_eq!(result[(0, 1)], 4);
        assert_eq!(result[(1, 0)], 6);
        assert_eq!(result[(1, 1)], 8);
    }

    #[test]
    fn test_matrix_div_matrix() {
        let matrix1 = Matrix::arange(1, 5, 1).reshape((2, 2));
        let matrix2 = Matrix::arange(1, 5, 1).reshape((2, 2));

        let result = matrix1 / matrix2;

        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(0, 1)], 1);
        assert_eq!(result[(1, 0)], 1);
        assert_eq!(result[(1, 1)], 1);
    }

    #[test]
    fn test_matrix_div_scaler() {
        let result = Matrix::arange(2, 10, 2).reshape((2, 2)) / 2;

        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(0, 1)], 2);
        assert_eq!(result[(1, 0)], 3);
        assert_eq!(result[(1, 1)], 4);
    }

    #[test]
    fn test_matrix_sub_matrix() {
        let matrix1 = Matrix::arange(1, 5, 1).reshape((2, 2));
        let matrix2 = Matrix::arange(5, 9, 1).reshape((2, 2));

        let result = matrix1 - matrix2;

        assert_eq!(result[(0, 0)], -4);
        assert_eq!(result[(0, 1)], -4);
        assert_eq!(result[(1, 0)], -4);
        assert_eq!(result[(1, 1)], -4);
    }

    #[test]
    fn test_matrix_sub_scaler() {
        let result = Matrix::arange(1, 5, 1).reshape((2, 2)) - 1;

        assert_eq!(result[(0, 0)], 0);
        assert_eq!(result[(0, 1)], 1);
        assert_eq!(result[(1, 0)], 2);
        assert_eq!(result[(1, 1)], 3);
    }

    #[test]
    fn test_matrix_dot() {
        let matrix1 = Matrix::arange(1, 5, 1).reshape((2, 2));
        let matrix2 = Matrix::arange(5, 9, 1).reshape((2, 2));

        let result = matrix1.dot(&matrix2);

        assert_eq!(result[(0, 0)], 19);
        assert_eq!(result[(0, 1)], 22);
        assert_eq!(result[(1, 0)], 43);
        assert_eq!(result[(1, 1)], 50);
    }

    #[test]
    fn test_matrix_transpose() {
        let matrix = Matrix::arange(1, 7, 1).reshape((2, 3));
        let result = matrix.transpose();

        assert_eq!(result.shape(), (3, 2));
        assert_eq!(result[(0, 0)], 1);
        assert_eq!(result[(0, 1)], 4);
        assert_eq!(result[(1, 0)], 2);
        assert_eq!(result[(1, 1)], 5);
        assert_eq!(result[(2, 0)], 3);
        assert_eq!(result[(2, 1)], 6);
    }
}
