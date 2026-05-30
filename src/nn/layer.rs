use rand::prelude::*;
use crate::nn::mat::Matrix;
pub struct Layer {
    pub w1: Matrix,
    pub b1: Matrix,

    pub w2: Matrix,
    pub b2: Matrix,     

}

impl Layer {
    pub fn new(rows1: usize, cols1: usize, rows2: usize, cols2: usize) -> Self {
        let mut l = Layer {
            w1: Matrix::new(rows1, cols1),
            b1: Matrix::new(1, cols1),
            w2: Matrix::new(rows2, cols2),
            b2: Matrix::new(1, cols2),
        };

        l.gen_ws();
        l
    }



    pub fn gen_ws(&mut self) {
        let mut rng = rand::rng();
        for j in 0..self.w1.cols {
            for i in 0..self.w1.rows     {
                let val: f32 = rng.random_range(-1.0..1.0);
                self.w1.set(i, j, val);
            }
            let val: f32 = rng.random_range(-1.0..1.0);
            self.b1.set(0, j, val);
        }

        for j in 0..self.w2.cols {
            for i in 0..self.w2.rows {
                let val: f32 = rng.random_range(-1.0..1.0);
                self.w2.set(i, j, val);
            }
            let val: f32 = rng.random_range(-1.0..1.0);
            self.b2.set(0, j, val);
        }
    }
}

use std::fmt;


impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                write!(f, "{:.2} ", self.w1.get(i, j))?;
            }
            writeln!(f)?;
        }
        writeln!(f, "-----------------")?;
        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                write!(f, "{:.2} ", self.w2.get(i, j))?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
