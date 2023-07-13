use rand::Rng;

const INNODE:usize = 2;
const HIDENODE:usize = 16;
const OUTNODE:usize = 1;
const ALPHA:f64 = 0.1;
const BINARY_DIM:usize = 8;

#[derive(Default, Debug)]
struct RNN {
    w: [[f64; HIDENODE];INNODE],
    w1: [[f64; OUTNODE];HIDENODE],
    wh: [[f64; HIDENODE];HIDENODE],

    layer_0: [f64; INNODE],
    layer_2: [f64; OUTNODE],
}

impl RNN {
    fn winit(w: &mut [f64]) {
        let mut rng = rand::thread_rng();
        for item in w.iter_mut() {
            *item = rng.gen_range(0..2) as f64; 
        }
    }

    pub fn new() -> Self {
        let mut res = Self {
            ..Default::default()
        };

        for item in res.w.iter_mut() {
            Self::winit(item);
        }
        for item in res.w1.iter_mut() {
            Self::winit(item);
        }
        for item in res.wh.iter_mut() {
            Self::winit(item);
        }

        println!("{:?}", res.w);
        println!("{:?}", res.w1);
        println!("{:?}", res.wh);

        return res;
    }

    fn int_to_binary(mut n: i32, arr: &mut [i32]) {
        let mut i = 0;
        while n != 0 {
            arr[i] = n % 2;
            i += 1;
            n /= 2;
        }
        while i < BINARY_DIM {
            arr[i] = 0;
            i += 1;
        }
    }

    fn sigmoid(x: f64) -> f64 {
        return 1.0 / (1.0 + f64::exp(-x));
    }

    fn dsigmoid(y: f64) -> f64 {
        return y * (1.0 - y);
    }

    pub fn train(&mut self) {
        let mut rng = rand::thread_rng();
        let largest_number = BINARY_DIM.pow(2);

        let mut layer_1_vector = Vec::<[f64; HIDENODE]>::new();
        let mut layer_2_delta = Vec::<f64>::new();

        for epoch in 0..50000 {
            let mut e = 0.0f64;
            layer_1_vector.clear();
            layer_2_delta.clear();

            let a_int: i32 = rng.gen_range(0..largest_number / 2) as i32;
            let mut a:[i32; BINARY_DIM] = [0; BINARY_DIM];
            Self::int_to_binary(a_int, &mut a);
            //println!("{:?}", a_int);
            //println!("{:?}", a);

            let mut d = [0i32; BINARY_DIM];

            let b_int: i32 = rng.gen_range(0..largest_number / 2) as i32;
            let mut b:[i32; BINARY_DIM] = [0; BINARY_DIM];
            Self::int_to_binary(b_int, &mut b);

            let c_int: i32 = a_int + b_int;
            let mut c:[i32; BINARY_DIM] = [0; BINARY_DIM];
            Self::int_to_binary(c_int, &mut c);

            layer_1_vector.push([0f64; HIDENODE]);

            //正向传播
            for p in 0..BINARY_DIM {
                self.layer_0[0] = a[p] as f64;
                self.layer_0[1] = b[p] as f64;
                let y = c[p] as f64;

                let mut layer_1 = [0f64; HIDENODE];

                for j in 0..HIDENODE {
                    let mut o1: f64 = 0.0;
                    for m in 0..INNODE {
                        o1 += self.layer_0[m] * self.w[m][j]; 
                    }

                    let layer_1_pre = layer_1_vector.last_mut().unwrap();
                    for m in 0..HIDENODE {
                        o1 += layer_1_pre[m] * self.wh[m][j];
                    }

                    layer_1[j] = Self::sigmoid(o1);
                    assert!(layer_1[j] >= 0.0);
                    assert!(layer_1[j] <= 1.0);
                }

                for k in 0..OUTNODE {
                    let mut o2 = 0.0f64;
                    for j in 0..HIDENODE {
                        o2 += layer_1[j] * self.w1[j][k];
                    }
                    self.layer_2[k] = Self::sigmoid(o2);
                    assert!(self.layer_2[k] >= 0.0);
                    assert!(self.layer_2[k] <= 1.0);
                }

                d[p] = f64::floor(self.layer_2[0] + 0.5) as i32; 
                layer_1_vector.push(layer_1);

                layer_2_delta.push((y - self.layer_2[0]) * Self::dsigmoid(self.layer_2[0]));
                e += (y - self.layer_2[0]).abs();
            }

            //误差反向传播


            //隐含层偏差
            let mut layer_1_delta = [0f64; HIDENODE];
            let mut layer_1_future_delta = [0f64; HIDENODE];

            for fp in 0..BINARY_DIM {
                let p = BINARY_DIM - 1 - fp;
                self.layer_0[0] = a[p] as f64;
                self.layer_0[1] = b[p] as f64;

                let layer_1 = &layer_1_vector[p+1];
                let layer_1_pre = layer_1_vector[p];

                for k in 0..OUTNODE {
                    for j in 0..HIDENODE {
                        self.w1[j][k] += ALPHA * layer_2_delta[p] * layer_1[j]
                    }
                }

                for j in 0..HIDENODE { //对于
                    layer_1_delta[j] = 0.0;
                    for k in 0..OUTNODE {
                        layer_1_delta[j] += layer_2_delta[p] * self.w1[j][k];
                    }
                    for k in 0..HIDENODE {
                        layer_1_delta[j] += layer_1_future_delta[k] * self.wh[j][k];
                    }

                    //隐含层的误差矫正
                    layer_1_delta[j] = layer_1_delta[j] * Self::dsigmoid(layer_1[j]);

                    //更新输入层
                    for k in 0..INNODE {
                        self.w[k][j] += ALPHA * layer_1_delta[j] * self.layer_0[k];
                    }

                    //更新前一个隐含层
                    for k in 0..HIDENODE {
                        self.wh[k][j] += ALPHA * layer_1_delta[j] * layer_1_pre[k];
                    }
                }

                layer_1_future_delta = layer_1_delta;
            }

            if epoch % 1000 == 0 {
                println!("error:{}", e);
                print!("pred:");
                for fk in 0..BINARY_DIM {
                    let k = BINARY_DIM - 1 - fk;
                    print!("{}", d[k]);
                }
                println!("");

                print!("true:");
                for fk in 0..BINARY_DIM {
                    let k = BINARY_DIM - 1 - fk;
                    print!("{}",c[k]);
                }
                println!("");

                let mut out = 0i32;
                for fk in 0..BINARY_DIM {
                    let k = BINARY_DIM - 1 - fk;
                    out += d[k] * 2i32.pow(k as u32);
                }
                println!("{}+{}={}", a_int, b_int, out);
            }
        }
    }
}

fn main() {
    let mut rnn:RNN = RNN::new();
    rnn.train();
}
