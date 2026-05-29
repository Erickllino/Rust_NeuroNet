

mod nn;
use nn::mat::Matrix;
use nn::layer::Layer;

struct NeuralNet{

}

impl NeuralNet {

    fn f_foward(){

    }

    fn loss(out:&Matrix, Y:&Matrix){
        
    }
    
    fn back_prop(){

    }
}



// fn train(){

// }

fn sigmoid(x:&Matrix) -> Matrix{
    let mut result:Matrix = Matrix::new(x.rows, x.cols); 
    let mut r:f32;
    for i in 0..x.rows{
        for j in 0..x.cols{
            
            r = 1.0 + (-x.get(i,j).exp());
            r = 1.0/r;    
            result.set(i,j, r)
        }
    }
    return result;
}

// fn evaluate(){

// }


fn main() {
    let mut a = Matrix::new(2, 3);
    // fill with some values
    for i in 0..2 {
        for j in 0..3 {
            a.set(i, j, (i * 3 + j + 1) as f32);
        }
    }

    let mut b = Matrix::new(3, 2);
    for i in 0..3 {
        for j in 0..2 {
            b.set(i, j, (i * 2 + j + 1) as f32);
        }
    }

    println!("{a}");

    let c = sigmoid(&a);
    println!("{}", c);

    let d = &a + &a;
    println!("{d}");
}