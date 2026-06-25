/// Feed-forward network (FFN).
///
/// The BERT feed-forward block:
/// ```text
/// intermediate = GELU(x @ W1 + b1)
/// output = intermediate @ W2 + b2
/// ```
///
/// W1: [intermediate_size, hidden_size] (stored transposed in HF convention)
/// W2: [hidden_size, intermediate_size]
use crate::error::Result;
use crate::model::weights::FeedForwardWeights;
use crate::tensor::activation;
use crate::tensor::matmul;
use crate::tensor::ops;
use crate::tensor::{Shape, Tensor};

/// Run the feed-forward block.
///
/// # Arguments
/// - `hidden`: Input tensor [batch, seq, hidden_size]
/// - `weights`: Feed-forward weights
///
/// # Returns
/// Output tensor [batch, seq, hidden_size]
pub fn feed_forward(hidden: &Tensor, weights: &FeedForwardWeights) -> Result<Tensor> {
    let batch_size = hidden.shape().dim(0)?;
    let seq_len = hidden.shape().dim(1)?;
    let hidden_size = hidden.shape().dim(2)?;

    // Flatten to 2D for matmul
    let hidden_2d = hidden.reshape(Shape::new(vec![batch_size * seq_len, hidden_size]))?;

    // Intermediate: GELU(x @ W1^T + b1)
    let w1_t = weights.intermediate_weight.transpose_2d()?;
    let intermediate = ops::add_bias(
        &matmul::matmul(&hidden_2d, &w1_t)?,
        &weights.intermediate_bias,
    )?;
    let intermediate = activation::gelu(&intermediate);

    // Output: intermediate @ W2^T + b2
    let w2_t = weights.output_weight.transpose_2d()?;
    let output = ops::add_bias(&matmul::matmul(&intermediate, &w2_t)?, &weights.output_bias)?;

    // Reshape back to 3D
    output.reshape(Shape::new(vec![batch_size, seq_len, hidden_size]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ff_shape() {
        let hidden_size = 4;
        let inter_size = 8;
        let batch = 2;
        let seq = 3;

        let hidden = Tensor::ones(Shape::new(vec![batch, seq, hidden_size]));
        let weights = FeedForwardWeights {
            intermediate_weight: Tensor::ones(Shape::new(vec![inter_size, hidden_size])),
            intermediate_bias: Tensor::zeros(Shape::new(vec![inter_size])),
            output_weight: Tensor::ones(Shape::new(vec![hidden_size, inter_size])),
            output_bias: Tensor::zeros(Shape::new(vec![hidden_size])),
            output_ln_weight: Tensor::ones(Shape::new(vec![hidden_size])),
            output_ln_bias: Tensor::zeros(Shape::new(vec![hidden_size])),
        };

        let result = feed_forward(&hidden, &weights).unwrap();
        assert_eq!(result.shape().dims(), &[batch, seq, hidden_size]);
    }
}
