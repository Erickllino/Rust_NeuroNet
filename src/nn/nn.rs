use crate::nn::mat::Matrix;
use crate::nn::layer::Layer;
use std::io::Write;
pub struct NeuralNet {
    pub layers: Vec<Layer>,
}

impl NeuralNet {
    pub fn f_foward(&self, x: &Matrix) -> Matrix {

        let mut a: Matrix = x.clone();
        if self.layers.len() == 0 { 
            return a;
        }

        for l in &self.layers {
            let mut z1 = a.dot(&l.w);
            for i in 0..z1.rows {
                for j in 0..z1.cols {
                    z1.set(i, j, z1.get(i, j) + l.b.get(0, j));
                }
            }
            a  = sigmoid(&z1);
            
        }


        return a;
        
    }
    pub fn loss(out: &Matrix, y: &Matrix) -> f32 {
        let s = &(out - y) * &(out - y)   ;
        let r = s.sum();

        return r / (s.rows * s.cols) as f32;
    }
    pub fn back_prop(&mut self, x: &Matrix, y: &Matrix, lr: f32) {
        if self.layers.is_empty() {
            return;
        }

        // --- Forward pass, caching z (pre-activation) and a (activation) of every layer ---
        let mut zs: Vec<Matrix> = Vec::new();
        let mut activations: Vec<Matrix> = Vec::new();
        for l in 0..self.layers.len() {
            let layer = &self.layers[l];
            // input to this layer: the network input for layer 0, else previous activation
            let a_prev: &Matrix = if l == 0 { x } else { &activations[l - 1] };

            let mut z = a_prev.dot(&layer.w);
            for i in 0..z.rows {
                for j in 0..z.cols {
                    z.set(i, j, z.get(i, j) + layer.b.get(0, j));
                }
            }
            let a = sigmoid(&z);
            zs.push(z);
            activations.push(a);
        }

        // --- Backward pass ---
        let last = self.layers.len() - 1;
        // output-layer error: dLoss/dz = (a_L - y) ⊙ σ'(z_L)
        let mut delta = &(&activations[last] - y) * &sigmoid_derivative(&zs[last]);

        for l in (0..self.layers.len()).rev() {
            // input that fed layer l
            let a_prev: &Matrix = if l == 0 { x } else { &activations[l - 1] };

            // gradients for this layer (use current delta)
            let grad_w = a_prev.transpose().dot(&delta);

            let mut grad_b = Matrix::new(1, delta.cols);
            for j in 0..delta.cols {
                let mut sum: f32 = 0.0;
                for i in 0..delta.rows {
                    sum += delta.get(i, j);
                }
                grad_b.set(0, j, sum);
            }

            // propagate error to the previous layer BEFORE overwriting this layer's weights
            let next_delta = if l > 0 {
                let d_a = delta.dot(&self.layers[l].w.transpose());
                Some(&d_a * &sigmoid_derivative(&zs[l - 1]))
            } else {
                None
            };

            // update weights and biases of layer l
            let w_final = &self.layers[l].w - &(&grad_w * lr);
            let b_final = &self.layers[l].b - &(&grad_b * lr);
            self.layers[l].w.full_set(w_final);
            self.layers[l].b.full_set(b_final);

            if let Some(nd) = next_delta {
                delta = nd;
            }
        }
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

pub fn train(data: Vec<Matrix>, y: Vec<Matrix>, model:&mut NeuralNet, epochs: usize, lr: f32) {
    let mut losses: Vec<f32> = Vec::new();
    let mut out: Matrix ;

    let starting_time = std::time::Instant::now();
    for k in 0..epochs {
        tqdm_bar(k, epochs);
        println!("");
        for i in 0..data.len() {
            
            model.back_prop(&data[i], &y[i], lr);

            // Update time and print time remaining
            let elapsed = starting_time.elapsed();
            print!("\rElapsed: {:.2?}", elapsed);

        }
        println!("");
        out = model.f_foward(&data[0]);
        let loss = NeuralNet::loss(&out, &y[0]);

        losses.push(loss);
       
        print!("\x1B[2J\x1B[1;1H");
        std::io::stdout().flush().unwrap();
    }
    println!()
}

pub fn predict(data: Matrix, model: NeuralNet) -> Matrix {
    let out: Matrix = model.f_foward(&data);
    out
}

#[allow(dead_code)]
pub fn tqdm(i:usize, total: usize) {
    let percent = (i as f32 / total as f32) * 100.0;
    print!("\rProgress: {:.2}%", percent);         
}
#[allow(dead_code)]
pub fn tqdm_bar(i:usize, total: usize) {
    let percent = (i as f32 / total as f32) * 100.0;
    let bar_length = 50; // length of the progress bar
    let filled_length = (percent / 100.0 * bar_length as f32).round() as usize;
    let bar = "█".repeat(filled_length) + &" ".repeat(bar_length - filled_length);
    print!("\rProgress: |{}| {:.2}%", bar, percent);      
    
}