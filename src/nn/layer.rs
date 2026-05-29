use rand::Rng;

pub struct Layer {
    pub w1: Vec<f32>,
    pub rows1: usize,
    pub cols1: usize,

    pub w2: Vec<f32>,
    pub rows2:usize,
    pub cols2:usize
}


impl Layer{

    pub fn new(rows1: usize, cols1: usize, rows2:usize, cols2: usize) -> Self {
        let mut l = Layer {
            w1: vec![0.0; rows1 * cols1],
            rows1,
            cols1,

            w2: vec![0.0; rows2*cols2],
            rows2,
            cols2
        };

        l.gen_ws();
        l
    }

    pub fn get(){

    }

    pub fn set_w1(&mut self, i: usize, j: usize, val: f32) {
        self.w1[i * self.cols1 + j] = val;
    }

    pub fn set_w2(&mut self, i: usize, j: usize, val: f32) {
        self.w2[i * self.cols2 + j] = val;
    }
    
    pub fn gen_ws(&mut self){
        let mut rng = rand::rng();
        for i in 0..self.rows1{
            for j in 0..self.cols1{
                let val:f32 = rng.random();
                self.set_w1(i, j, val);
            }
        }

        for i in 0..self.rows2{
            for j in 0..self.cols2{
                let val:f32 = rng.random();
                self.set_w2(i, j, val);
            }
        }

    }


} 
    