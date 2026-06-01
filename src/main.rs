struct Node {
    weight: f32,
    bais: f32,
    function: &str
}

fn swiglu(x: f32) -> f32 {
    let x = x * x;
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

fn main() {
    println!("Hello, world!");
}
