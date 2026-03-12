/// Encoder layer and encoder stack.
///
/// A single BERT encoder layer applies:
/// ```text
/// attn_output = MultiHeadAttention(hidden, mask)
/// hidden = LayerNorm(hidden + attn_output)     // residual connection
/// ff_output = FeedForward(hidden)
/// hidden = LayerNorm(hidden + ff_output)         // residual connection
/// ```
///
/// The encoder stack is N layers applied sequentially.

use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops;
use crate::tensor::layernorm;
use crate::model::config::ModelConfig;
use crate::model::weights::EncoderLayerWeights;
use crate::model::attention;
use crate::model::ff;

/// Run a single encoder layer.
pub fn encoder_layer(
    hidden: &Tensor,
    attention_mask: &[Vec<u32>],
    weights: &EncoderLayerWeights,
    config: &ModelConfig,
) -> Result<Tensor> {
    // 1. Self-attention
    let attn_output = attention::multi_head_attention(
        hidden,
        attention_mask,
        &weights.attention,
        config.num_attention_heads,
    )?;

    // 2. Residual + LayerNorm (post-attention)
    let hidden = ops::add(hidden, &attn_output)?;
    let hidden = layernorm::layer_norm(
        &hidden,
        &weights.attention.output_ln_weight,
        &weights.attention.output_ln_bias,
        config.ln_eps(),
    )?;

    // 3. Feed-forward
    let ff_output = ff::feed_forward(&hidden, &weights.ff)?;

    // 4. Residual + LayerNorm (post-ff)
    let hidden = ops::add(&hidden, &ff_output)?;
    let hidden = layernorm::layer_norm(
        &hidden,
        &weights.ff.output_ln_weight,
        &weights.ff.output_ln_bias,
        config.ln_eps(),
    )?;

    Ok(hidden)
}

/// Run the full encoder stack (N layers).
pub fn encoder_forward(
    mut hidden: Tensor,
    attention_mask: &[Vec<u32>],
    layers: &[EncoderLayerWeights],
    config: &ModelConfig,
) -> Result<Tensor> {
    for layer_weights in layers {
        hidden = encoder_layer(&hidden, attention_mask, layer_weights, config)?;
    }
    Ok(hidden)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Shape;
    use crate::model::weights::*;

    fn make_tiny_config() -> ModelConfig {
        ModelConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            intermediate_size: 8,
            max_position_embeddings: 16,
            layer_norm_eps: 1e-12,
            hidden_act: "gelu".into(),
            type_vocab_size: 2,
            model_type: None,
        }
    }

    fn make_tiny_layer_weights(hidden: usize, inter: usize) -> EncoderLayerWeights {
        EncoderLayerWeights {
            attention: AttentionWeights {
                query_weight: Tensor::full(Shape::new(vec![hidden, hidden]), 0.01),
                query_bias: Tensor::zeros(Shape::new(vec![hidden])),
                key_weight: Tensor::full(Shape::new(vec![hidden, hidden]), 0.01),
                key_bias: Tensor::zeros(Shape::new(vec![hidden])),
                value_weight: Tensor::full(Shape::new(vec![hidden, hidden]), 0.01),
                value_bias: Tensor::zeros(Shape::new(vec![hidden])),
                output_weight: Tensor::full(Shape::new(vec![hidden, hidden]), 0.01),
                output_bias: Tensor::zeros(Shape::new(vec![hidden])),
                output_ln_weight: Tensor::ones(Shape::new(vec![hidden])),
                output_ln_bias: Tensor::zeros(Shape::new(vec![hidden])),
            },
            ff: FeedForwardWeights {
                intermediate_weight: Tensor::full(Shape::new(vec![inter, hidden]), 0.01),
                intermediate_bias: Tensor::zeros(Shape::new(vec![inter])),
                output_weight: Tensor::full(Shape::new(vec![hidden, inter]), 0.01),
                output_bias: Tensor::zeros(Shape::new(vec![hidden])),
                output_ln_weight: Tensor::ones(Shape::new(vec![hidden])),
                output_ln_bias: Tensor::zeros(Shape::new(vec![hidden])),
            },
        }
    }

    #[test]
    fn test_encoder_layer_shape() {
        let config = make_tiny_config();
        let weights = make_tiny_layer_weights(4, 8);
        let hidden = Tensor::ones(Shape::new(vec![2, 3, 4])); // [batch=2, seq=3, hidden=4]
        let mask = vec![vec![1u32, 1, 1], vec![1, 1, 0]];

        let output = encoder_layer(&hidden, &mask, &weights, &config).unwrap();
        assert_eq!(output.shape().dims(), &[2, 3, 4]);
    }

    #[test]
    fn test_encoder_stack_shape() {
        let config = make_tiny_config();
        let layers = vec![make_tiny_layer_weights(4, 8)];
        let hidden = Tensor::ones(Shape::new(vec![1, 5, 4]));
        let mask = vec![vec![1u32, 1, 1, 1, 0]];

        let output = encoder_forward(hidden, &mask, &layers, &config).unwrap();
        assert_eq!(output.shape().dims(), &[1, 5, 4]);
    }
}
