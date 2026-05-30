use rand::prelude::*;
use crate::nn::mat::Matrix;
pub struct Layer {
    pub w: Matrix,
    pub b: Matrix,
}

impl Layer {
    pub fn new(rows1: usize, cols1: usize) -> Self {
        let mut l = Layer {
            w: Matrix::new(rows1, cols1),
            b: Matrix::new(1, cols1),
        };

        l.gen_ws();
        l
    }



    pub fn gen_ws(&mut self) {
        let mut rng = rand::rng();
        for j in 0..self.w.cols {
            for i in 0..self.w.rows     {
                let val: f32 = rng.random_range(-1.0..1.0);
                self.w.set(i, j, val);
            }
            let val: f32 = rng.random_range(-1.0..1.0);
            self.b.set(0, j, val);
        }
    }
}

use std::fmt;


impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for i in 0..self.w.rows {
            for j in 0..self.w.cols {
                write!(f, "{:.2} ", self.w.get(i, j))?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
