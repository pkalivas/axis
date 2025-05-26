#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Shape {
    pub dims: Vec<usize>,
}

impl Shape {
    pub fn new(dims: Vec<usize>) -> Self {
        Shape { dims }
    }

    pub fn size(&self) -> usize {
        self.dims.iter().product()
    }

    pub fn dim(&self, index: usize) -> usize {
        self.dims[index]
    }

    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    pub fn is_scalar(&self) -> bool {
        self.dims.len() == 1 && self.dims[0] == 1
    }

    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }

    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }

    pub fn is_tensor(&self) -> bool {
        self.dims.len() > 2
    }

    pub fn is_square(&self) -> bool {
        self.dims.len() == 2 && self.dims[0] == self.dims[1]
    }
}

impl Into<Shape> for Vec<usize> {
    fn into(self) -> Shape {
        Shape::new(self)
    }
}

impl Into<Shape> for usize {
    fn into(self) -> Shape {
        Shape::new(vec![self])
    }
}

impl Into<Shape> for (usize, usize) {
    fn into(self) -> Shape {
        Shape::new(vec![self.0, self.1])
    }
}

impl Into<Shape> for (usize, usize, usize) {
    fn into(self) -> Shape {
        Shape::new(vec![self.0, self.1, self.2])
    }
}

impl Into<Shape> for (usize, usize, usize, usize) {
    fn into(self) -> Shape {
        Shape::new(vec![self.0, self.1, self.2, self.3])
    }
}

impl Into<Shape> for (usize, usize, usize, usize, usize) {
    fn into(self) -> Shape {
        Shape::new(vec![self.0, self.1, self.2, self.3, self.4])
    }
}
