use rand::prelude::*;

#[derive(Clone)]
pub struct Matrix {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Matrix {
    // "static method" / constructor — no `self`
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    // method — takes &self (read-only reference)
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i * self.cols + j]
    }

    // mutable method — takes &mut self
    pub fn set(&mut self, i: usize, j: usize, val: f32) {
        self.data[i * self.cols + j] = val;
    }

    pub fn full_set(&mut self, r: Matrix) {
        assert_eq!(self.rows, r.rows);
        assert_eq!(self.cols, r.cols);
        self.data = r.data;
    }

    pub fn dot(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "dimension mismatch");
        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..self.cols {
                    let val = self.get(i, k) * other.get(k, j);
                    result.set(i, j, result.get(i, j) + val);
                }
            }
        }
        result
    }

    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }
}

use std::fmt;

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                write!(f, "{:.2} ", self.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

use std::ops::{Add, Mul, Neg, Sub};

impl Add for &Matrix {
    type Output = Matrix;

    fn add(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

impl Sub for &Matrix {
    type Output = Matrix;
    fn sub(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Matrix * scalar
impl Mul<f32> for &Matrix {
    type Output = Matrix;
    fn mul(self, scalar: f32) -> Matrix {
        let data = self.data.iter().map(|x| x * scalar).collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Hadamard product (element-wise multiplication)
impl Mul<&Matrix> for &Matrix {
    type Output = Matrix;
    fn mul(self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

// Negate -Matrix
impl Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        let data = self.data.iter().map(|x| -x).collect();
        Matrix {
            data,
            rows: self.rows,
            cols: self.cols,
        }
    }
}

#[allow(dead_code)]
pub fn random_matrix(rows: usize, cols: usize) -> Matrix {
    let mut m = Matrix::new(rows, cols);
    let mut rng = rand::rng();
    for i in 0..rows {
        for j in 0..cols {
            let val: f32 = rng.random_range(-1.0..1.0);
            m.set(i, j, val);
        }
    }
    m
}
