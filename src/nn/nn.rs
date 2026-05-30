use crate::nn::mat::Matrix;     
use crate::nn::layer::Layer;
pub struct NeuralNet {
    pub layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn f_foward(x: &Matrix, l: &Layer) -> Matrix {
        let mut z1 = x.dot(&l.w1);
        for i in 0..z1.rows {
            for j in 0..z1.cols {
                z1.set(i, j, z1.get(i, j) + l.b1.get(0, j));
            }
        }
        let a1: Matrix = sigmoid(&z1); 
        let mut z2: Matrix = a1.dot(&l.w2);
        for i in 0..z2.rows {
            for j in 0..z2.cols {
                z2.set(i, j, z2.get(i, j) + l.b2.get(0, j));
            }
        }
        let a2: Matrix = sigmoid(&z2)      ; 
        return a2;
    }
    pub fn loss(out: &Matrix, y: &Matrix) -> f32 {
        let s = &(out - y) * &(out - y)   ;
        let r = s.sum();

        return r / (s.rows * s.cols) as f32;
    }
    pub fn back_prop(x: &Matrix, y: &Matrix, l: &mut Layer, lr: f32) {
        // Later add all layers
        let mut z1 = x.dot(&l.w1);
        for i in 0..z1.rows {
            for j in 0..z1.cols {
                z1.set(i, j, z1.get(i, j) + l.b1.get(0, j));
            }
        } 
        let a1 = sigmoid(&z1);   
        let mut z2 = a1.dot(&l.w2);
        for i in 0..z2.rows {
            for j in 0..z2.cols {
                z2.set(i, j, z2.get(i, j) + l.b2.get(0, j));
            }
        } 
        let a2 = sigmoid(&z2); 

        let d_a2 = &a2 - y;
        let d_z2 = &d_a2 * (&sigmoid_derivative(&z2));
        let mut d_b2: Matrix = Matrix::new(1, d_z2.cols);
        for j in 0..d_z2.cols {
            let mut sum: f32 = 0.0;
            for i in 0..d_z2.rows {
                sum += d_z2.get(i, j);
            }
            d_b2.set(0, j, sum);
        }   
        let d_w2 = a1.transpose().dot(&d_z2);

        let d_a1 = d_z2.dot(&l.w2.transpose());
        let d_z1 = &d_a1 * (&sigmoid_derivative(&z1));
        let mut d_b1: Matrix = Matrix::new(1, d_z1.cols)    ;
        for j in 0..d_z1.cols { 
            let mut sum: f32 = 0.0;
            for i in 0..d_z1.rows {
                sum += d_z1.get(i, j);
            }
            d_b1.set(0, j, sum);
        }
        let d_w1 = x.transpose().dot(&d_z1);

        // Update weights

        let w1_final = &l.w1 - &(&d_w1 * lr); // learning rate
        let w2_final = &l.w2 - &(&d_w2 * lr); // learning rate

        let b1_final = &l.b1 - &(&d_b1 * lr); // learning rate
        let b2_final = &l.b2 - &(&d_b2 * lr); // learning rate

        
        l.w1.full_set(w1_final);
        l.b1.full_set(b1_final);    
        l.w2.full_set(w2_final);
        l.b2.full_set(b2_final);
    }



    
}

fn sigmoid(x: &Matrix) -> Matrix {
    let mut result: Matrix = Matrix::new(x.rows, x.cols);
    let mut r: f32;
    for i in 0..x.rows {
        for j in 0..x.cols {
            r = 1.0 + ((-x).get(i, j).exp());
            r = 1.0 / r;
            result.set(i, j, r)
        }
    }
    return result;
}

fn sigmoid_derivative(x: &Matrix) -> Matrix {
    let s = sigmoid(x);
    let mut result: Matrix = Matrix::new(x.rows, x.cols);
    let mut r: f32;
    for i in 0..x.rows {
        for j in 0..x.cols {
            r = s.get(i, j) * (1.0 - s.get(i, j));
            result.set(i, j, r)
        }
    }
    return result;
}

pub fn train(data: &Vec<Matrix>, y: &Vec<Matrix>, model: &mut NeuralNet, epochs: usize, lr: f32) {

    for _ in 0..epochs {
        let mut mean_loss: f32 = 0.0;   
        for i in 0..data.len() {
            let out = NeuralNet::f_foward(&data[i], &model.layers[0]);            
            NeuralNet::back_prop(&data[i], &y[i], &mut model.layers[0], lr);
            let loss = NeuralNet::loss(&out, &y[0]);
            mean_loss += loss;
        }
        mean_loss /= data.len() as f32;
        println!("Loss: {mean_loss}");
    }
}

pub fn predict(data: Matrix, model: &NeuralNet) -> Matrix {
    NeuralNet::f_foward(&data, &model.layers[0])
}
