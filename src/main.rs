mod nn;

use std::io::Write;

use nn::layer::Layer;
use nn::mat::Matrix;
use nn::nn::NeuralNet;
use nn::nn::{train, predict};

use std::fs;
const CSV_FILE_PATH: &str = "src/data/mnist_train.csv";  

fn main() {


    let contents = fs::read_to_string(CSV_FILE_PATH)
        .expect("Failed to read file");

    let mut label: Vec<Matrix> = Vec::new();
    let mut data: Vec<Matrix>  = Vec::new();
    for line in contents.lines().skip(1) { // skip header
        let fields: Vec<&str> = line.split(',').collect();

        let mut lbl = Matrix::new(1, 10);
        let idx = fields[0].parse::<usize>().unwrap();
        lbl.set(0, idx, 1.0); // parse label
        label.push(lbl);

        let mut d = Matrix::new(1, 784);
        for i in 1..fields.len() {
            d.set(0, i-1, fields[i].parse::<f32>().unwrap() / 255.0); // parse pixel values and normalize
        }
        data.push(d);


    }

    print!("\x1B[2J\x1B[1;1H");
    std::io::stdout().flush().unwrap();
    // Usando dados do XOR como exemplo
    // let mut data: Vec<Matrix> = vec![
    //     Matrix::new(1, 2),
    //     Matrix::new(1, 2),
    //     Matrix::new(1, 2),
    //     Matrix::new(1, 2),
    // 

    // data[0].set(0, 0, 0.0);
    // data[0].set(0, 1, 0.0); // [0, 0]
    // data[1].set(0, 0, 0.0); 
    // data[1].set(0, 1, 1.0); // [0, 1]
    // data[2].set(0, 0, 1.0);
    // data[2].set(0, 1, 0.0); // [1, 0]
    // data[3].set(0, 0, 1.0);
    // data[3].set(0, 1, 1.0); // [1, 1]

    // let mut label: Vec<Matrix> = vec![
    //     Matrix::new(1, 1),
    //     Matrix::new(1, 1),
    //     Matrix::new(1, 1),
    //     Matrix::new(1, 1),  
    // ];

    // label[0].set(0, 0, 0.0); // [0, 0] -> 0
    // label[1].set(0, 0, 1.0); // [0, 1] -> 1
    // label[2].set(0, 0, 1.0); // [1, 0] -> 1
    // label[3].set(0, 0, 0.0); // [1, 1] -> 0

    let mut model = NeuralNet {
        layers: vec![Layer::new(784, 128), Layer::new(128, 10), Layer::new(10, 10)],
    };

    let epochs = 20;
    let lr: f32 = 0.01; // learning rate    

    train(data, label, &mut model, epochs, lr);

    let mut test = Matrix::new(1, 784);
    // get the first test sample from the CSV file  
    for i in 1..785 {
        test.set(0, i-1, contents.lines().nth(1).unwrap().split(',').nth(i).unwrap().parse::<f32>().unwrap() / 255.0);
    }
    // 
    let test_label = contents.lines().nth(1).unwrap().split(',').nth(0).unwrap();           
    println!("Test sample: {}", test_label);
    println!("Prediction: {}", predict(test,model));  
}
