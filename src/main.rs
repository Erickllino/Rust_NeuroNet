mod nn;

use nn::layer::Layer;
use nn::mat::Matrix;
use nn::nn::NeuralNet;
use nn::nn::{train, predict};





fn main() {
    // Usando dados do XOR como exemplo
    let mut data: Vec<Matrix> = vec![
        Matrix::new(1, 2),
        Matrix::new(1, 2),
        Matrix::new(1, 2),
        Matrix::new(1, 2),
    ];

    data[0].set(0, 0, 0.0);
    data[0].set(0, 1, 0.0); // [0, 0]
    data[1].set(0, 0, 0.0); 
    data[1].set(0, 1, 1.0); // [0, 1]
    data[2].set(0, 0, 1.0);
    data[2].set(0, 1, 0.0); // [1, 0]
    data[3].set(0, 0, 1.0);
    data[3].set(0, 1, 1.0); // [1, 1]

    let mut y: Vec<Matrix> = vec![
        Matrix::new(1, 1),
        Matrix::new(1, 1),
        Matrix::new(1, 1),
        Matrix::new(1, 1),  
    ];

    y[0].set(0, 0, 0.0); // [0, 0] -> 0
    y[1].set(0, 0, 1.0); // [0, 1] -> 1
    y[2].set(0, 0, 1.0); // [1, 0] -> 1
    y[3].set(0, 0, 0.0); // [1, 1] -> 0

    let mut model = NeuralNet {
        layers: vec![Layer::new(2, 4, 4, 1)],
    };

    let epochs = 100000;
    let lr: f32 = 0.01; // learning rate    

    train(&data, &y, &mut model, epochs, lr);

    let mut test = Matrix::new(1, 2);
    let a = 0.0;
    let b: f32 = 1.0;


    test.set(0, 0, a);
    test.set(0, 1, b);   


    println!("Test input: [{a}, {b}]");
    println!("Prediction: {}", predict(test, &model));  
}
