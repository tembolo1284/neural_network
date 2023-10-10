use rand::prelude::*; 

fn sigmoid(x: f64) -> f64 {
  1.0 / (1.0 + (-x).exp())
}

fn dsigmoid(x: f64) -> f64 {
  x * (1.0 - x)
}

fn shuffle(vec: &mut Vec<usize>, n: usize) {
  if n > 1 {
    let mut rng = thread_rng();
      
    for i in 0..n-1 {
      let j = rng.gen_range(0..n);
      vec.swap(i, j);
    }
  }
}

fn init_weights() -> f64 {
  let mut rng = thread_rng();
  rng.gen_range(0.0..1.0)  
}

fn main() {
  const NUM_INPUTS: usize = 2;
  const NUM_HIDDEN_NODES: usize = 2;
  const NUM_OUTPUTS: usize = 1;
  const NUM_TRAINING_SETS: usize = 4;

  const LR: f64 = 0.2;
  
  let mut hidden_layer = vec![0.0; NUM_HIDDEN_NODES];
  let mut output_layer = vec![0.0; NUM_OUTPUTS];

  let mut hidden_layer_bias = vec![0.0; NUM_HIDDEN_NODES];
  let mut output_layer_bias = vec![0.0; NUM_OUTPUTS];

  let mut hidden_weights = vec![vec![0.0; NUM_HIDDEN_NODES]; NUM_INPUTS];
  let mut output_weights = vec![vec![0.0; NUM_OUTPUTS]; NUM_HIDDEN_NODES];

  let training_inputs = vec![
      vec![0.0, 0.0],
      vec![1.0, 0.0],
      vec![0.0, 1.0],
      vec![1.0, 1.0]
  ];
  
  let training_outputs = vec![
      vec![0.0],
      vec![1.0],
      vec![1.0],
      vec![0.0]
  ];

  // Initialize weights
  for i in 0..NUM_INPUTS {
    for j in 0..NUM_HIDDEN_NODES {
      hidden_weights[i][j] = init_weights();
    }
  }

  for i in 0..NUM_HIDDEN_NODES {
    for j in 0..NUM_OUTPUTS {
      output_weights[i][j] = init_weights();
    }
  }

  for i in 0..NUM_OUTPUTS {
    output_layer_bias[i] = init_weights();
  }
  
  let mut training_set_order = vec![0, 1, 2, 3];
  
  let num_epochs = 10_000;

  for _ in 0..num_epochs {
    shuffle(&mut training_set_order, NUM_TRAINING_SETS);

    for &x in &training_set_order {
      
      // Forward pass
      for j in 0..NUM_HIDDEN_NODES {
        let mut activation = hidden_layer_bias[j];
        for k in 0..NUM_INPUTS {
          activation += training_inputs[x][k] * hidden_weights[k][j]; 
        }
        hidden_layer[j] = sigmoid(activation);
      }

      for j in 0..NUM_OUTPUTS {
        let mut activation = output_layer_bias[j];
        for k in 0..NUM_HIDDEN_NODES {
          activation += hidden_layer[k] * output_weights[k][j];
        }
        output_layer[j] = sigmoid(activation);  
      }

      println!(
        "Input: {} {} Output: {} Pred Output: {}",
        training_inputs[x][0],
        training_inputs[x][1], 
        output_layer[0],
        training_outputs[x][0]
      );

      // Backpropagation
      let mut delta_output = vec![0.0; NUM_OUTPUTS];

      for j in 0..NUM_OUTPUTS {
        let error = training_outputs[x][j] - output_layer[j];
        delta_output[j] = error * dsigmoid(output_layer[j]);
      }

      let mut delta_hidden = vec![0.0; NUM_HIDDEN_NODES];

      for j in 0..NUM_HIDDEN_NODES {
        let mut error = 0.0;
        for k in 0..NUM_OUTPUTS {
          error += delta_output[k] * output_weights[j][k];
        }
        delta_hidden[j] = error * dsigmoid(hidden_layer[j]); 
      }

      for j in 0..NUM_OUTPUTS {
        output_layer_bias[j] += delta_output[j] * LR;
        for k in 0..NUM_HIDDEN_NODES {
          output_weights[k][j] += hidden_layer[k] * delta_output[j] * LR; 
        }
      }

      for j in 0..NUM_HIDDEN_NODES {
        hidden_layer_bias[j] += delta_hidden[j] * LR;
        for k in 0..NUM_INPUTS {
          hidden_weights[k][j] += training_inputs[x][k] * delta_hidden[j] * LR;
        }
      }
    }
  }

  // Print final weights  
  println!("Final Hidden Weights:");
  for j in 0..NUM_HIDDEN_NODES {
    for k in 0..NUM_INPUTS {
      print!("{} ", hidden_weights[k][j]); 
    }
    println!();
  }
  
  println!("Final Output Weights:");
  for j in 0..NUM_OUTPUTS {
    for k in 0..NUM_HIDDEN_NODES {
      print!("{} ", output_weights[k][j]);
    }
    println!();
  }
  
  println!("Final Output Biases:");
  for j in 0..NUM_OUTPUTS {
    print!("{} ", output_layer_bias[j]); 
  }
  println!();
}