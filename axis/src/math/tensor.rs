use std::ops::{Index, IndexMut};

use super::Shape;

pub struct Tensor<T> {
    pub data: Vec<T>,
    pub shape: Shape,
    pub strides: Vec<usize>,
}

impl<T> Tensor<T> {
    pub fn new(shape: impl Into<Shape>) -> Self
    where
        T: Default + Clone,
    {
        let shape = shape.into();

        let mut strides = vec![1; shape.rank()];
        for i in (0..shape.rank() - 1).rev() {
            strides[i] = strides[i + 1] * shape.dim(i + 1);
        }

        Tensor {
            data: vec![T::default(); shape.size()],
            shape,
            strides,
        }
    }
}

impl<T> Index<usize> for Tensor<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> IndexMut<usize> for Tensor<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> Index<(usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        let idx = i * self.strides[0] + j * self.strides[1];
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize)> for Tensor<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        let idx = i * self.strides[0] + j * self.strides[1];
        &mut self.data[idx]
    }
}

impl<T> Index<(usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, (i, j, k): (usize, usize, usize)) -> &Self::Output {
        let idx = i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize, usize)> for Tensor<T> {
    fn index_mut(&mut self, (i, j, k): (usize, usize, usize)) -> &mut Self::Output {
        let idx = i * self.strides[0] + j * self.strides[1] + k * self.strides[2];
        &mut self.data[idx]
    }
}

impl<T> Index<(usize, usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, (i, j, k, l): (usize, usize, usize, usize)) -> &Self::Output {
        let idx =
            i * self.strides[0] + j * self.strides[1] + k * self.strides[2] + l * self.strides[3];
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize, usize, usize)> for Tensor<T> {
    fn index_mut(&mut self, (i, j, k, l): (usize, usize, usize, usize)) -> &mut Self::Output {
        let idx =
            i * self.strides[0] + j * self.strides[1] + k * self.strides[2] + l * self.strides[3];
        &mut self.data[idx]
    }
}

impl<T> Index<(usize, usize, usize, usize, usize)> for Tensor<T> {
    type Output = T;

    fn index(&self, (i, j, k, l, m): (usize, usize, usize, usize, usize)) -> &Self::Output {
        let idx = i * self.strides[0]
            + j * self.strides[1]
            + k * self.strides[2]
            + l * self.strides[3]
            + m * self.strides[4];
        &self.data[idx]
    }
}

impl<T> IndexMut<(usize, usize, usize, usize, usize)> for Tensor<T> {
    fn index_mut(
        &mut self,
        (i, j, k, l, m): (usize, usize, usize, usize, usize),
    ) -> &mut Self::Output {
        let idx = i * self.strides[0]
            + j * self.strides[1]
            + k * self.strides[2]
            + l * self.strides[3]
            + m * self.strides[4];
        &mut self.data[idx]
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let tensor = Tensor::<f32>::new(vec![2, 3, 4]);
        assert_eq!(tensor.shape.dims, vec![2, 3, 4]);
        assert_eq!(tensor.data.len(), 24);
        assert_eq!(tensor.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_tensor_indexing() {
        let mut tensor = Tensor::<f32>::new((2, 3));
        tensor[(0, 0)] = 1.0;
        tensor[(1, 2)] = 2.0;

        assert_eq!(tensor[(0, 0)], 1.0);
        assert_eq!(tensor[(1, 2)], 2.0);

        tensor[(0, 1)] = 3.0;
        assert_eq!(tensor[(0, 1)], 3.0);

        let mut tensor_two = Tensor::<f32>::new((2, 3, 4));
        tensor_two[(0, 0, 0)] = 1.0;
        tensor_two[(1, 2, 3)] = 2.0;
        assert_eq!(tensor_two[(0, 0, 0)], 1.0);
        assert_eq!(tensor_two[(1, 2, 3)], 2.0);
        tensor_two[(0, 1, 2)] = 3.0;
        assert_eq!(tensor_two[(0, 1, 2)], 3.0);
    }
}
