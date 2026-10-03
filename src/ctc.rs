//! Greedy Connectionist Temporal Classification (CTC) decoding.
//!
//! Each time step selects one class. Adjacent repetitions collapse, then
//! blanks disappear: `a, a, blank, a` becomes `aa`. Dictionary positions must
//! match the model's output classes, including the blank class.

use crate::dictionary::RecDictionary;
use ndarray::{s, Array3, Axis};

/// Configuration options for greedy CTC decoding.
#[derive(Debug, Clone, Default)]
pub struct CtcGreedyDecoderConfig {
    /// Class that separates repeated tokens without emitting text (default: 0).
    pub blank_id: usize,
    /// Replacement for a selected class missing from the dictionary.
    /// `None` (the default) makes such a class an error.
    pub fallback_token: Option<String>,
}

/// Errors that can be produced during greedy CTC decoding.
#[derive(Debug)]
pub enum CtcGreedyDecoderError {
    /// The specified blank id is not representable within the class dimension.
    BlankIdOutOfRange { blank_id: usize, class_count: usize },
    /// The provided `valid_timesteps` slice length does not match the batch size.
    TimestepsLengthMismatch { expected: usize, actual: usize },
    /// The decoder encountered an index that is not present in the dictionary.
    DictionaryIndexMissing {
        index: usize,
        dictionary_size: usize,
    },
    /// The decoder was asked to process an empty batch.
    EmptyBatch,
}

impl std::fmt::Display for CtcGreedyDecoderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CtcGreedyDecoderError::BlankIdOutOfRange {
                blank_id,
                class_count,
            } => write!(
                f,
                "blank id {} is out of range for class dimension size {}",
                blank_id, class_count
            ),
            CtcGreedyDecoderError::TimestepsLengthMismatch { expected, actual } => write!(
                f,
                "valid_timesteps length {} does not match batch size {}",
                actual, expected
            ),
            CtcGreedyDecoderError::DictionaryIndexMissing {
                index,
                dictionary_size,
            } => write!(
                f,
                "decoded index {} is not present in dictionary (size {})",
                index, dictionary_size
            ),
            CtcGreedyDecoderError::EmptyBatch => {
                write!(f, "ctc decoder received an empty batch")
            }
        }
    }
}

impl std::error::Error for CtcGreedyDecoderError {}

/// Result of decoding a single sequence.
#[derive(Debug, Clone)]
pub struct DecodedSequence {
    /// Concatenated dictionary tokens after blank and repetition removal.
    pub text: String,
    /// Emitted model class IDs, including IDs replaced by the fallback token.
    pub token_indices: Vec<usize>,
    /// Mean probability of emitted tokens, excluding blanks and repetitions.
    /// Empty output has confidence 1.0; this does not imply text was found.
    pub confidence: f32,
    /// Number of emitted tokens replaced by the configured fallback.
    pub fallback_count: usize,
}

/// Greedy CTC decoder producing text using the recognition dictionary.
#[derive(Debug, Clone)]
pub struct CtcGreedyDecoder {
    config: CtcGreedyDecoderConfig,
}

impl CtcGreedyDecoder {
    /// Stores decoding options; class bounds are checked when decoding.
    pub fn new(config: CtcGreedyDecoderConfig) -> Self {
        Self { config }
    }

    /// Decodes a `[batch, time, class]` tensor in batch order.
    ///
    /// `valid_timesteps` supplies one prefix length per sample to exclude
    /// right-hand padding. Lengths exceeding the time dimension are clamped;
    /// zero produces empty text. Rows resembling probability distributions
    /// are used directly; other rows use softmax for confidence calculation.
    /// Inputs are expected to contain finite scores.
    ///
    /// # Errors
    /// Returns an error for an empty batch, an invalid blank ID, a mismatched
    /// prefix-length count, or an unknown selected class without a fallback.
    pub fn decode(
        &self,
        logits: &Array3<f32>,
        valid_timesteps: &[usize],
        dictionary: &RecDictionary,
    ) -> Result<Vec<DecodedSequence>, CtcGreedyDecoderError> {
        let batch_size = logits.len_of(Axis(0));
        let time_steps = logits.len_of(Axis(1));
        let class_count = logits.len_of(Axis(2));

        if batch_size == 0 {
            return Err(CtcGreedyDecoderError::EmptyBatch);
        }

        if self.config.blank_id >= class_count {
            return Err(CtcGreedyDecoderError::BlankIdOutOfRange {
                blank_id: self.config.blank_id,
                class_count,
            });
        }

        if valid_timesteps.len() != batch_size {
            return Err(CtcGreedyDecoderError::TimestepsLengthMismatch {
                expected: batch_size,
                actual: valid_timesteps.len(),
            });
        }

        let mut results = Vec::with_capacity(batch_size);
        for (batch_index, &valid_steps) in valid_timesteps.iter().enumerate() {
            let max_steps = valid_steps.min(time_steps);
            let mut previous_symbol: Option<usize> = None;
            let mut text = String::new();
            let mut token_indices = Vec::new();
            let mut probability_sum = 0.0f64;
            let mut confidence_count = 0usize;
            let mut fallback_count = 0usize;

            for t in 0..max_steps {
                let step = logits.slice(s![batch_index, t, ..]);

                let mut best_index = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                let mut row_sum = 0.0f32;
                let mut min_value = f32::INFINITY;
                let mut max_value = f32::NEG_INFINITY;
                for (idx, value) in step.iter().enumerate() {
                    if *value > best_value {
                        best_value = *value;
                        best_index = idx;
                    }
                    row_sum += *value;
                    if *value < min_value {
                        min_value = *value;
                    }
                    if *value > max_value {
                        max_value = *value;
                    }
                }

                // Some exports already include softmax. Applying it again
                // would preserve argmax but distort the reported confidence.
                let is_probability_distribution = min_value.is_finite()
                    && max_value.is_finite()
                    && min_value >= -1e-4
                    && max_value <= 1.0 + 1e-4
                    && (row_sum - 1.0).abs() <= 1e-3;

                if best_index == self.config.blank_id {
                    // A blank breaks a run: a, blank, a must emit two tokens.
                    previous_symbol = None;
                    continue;
                }

                if Some(best_index) == previous_symbol {
                    continue;
                }

                let probability = if is_probability_distribution {
                    best_value.clamp(0.0, 1.0)
                } else {
                    // Subtract the maximum before exponentiation to avoid
                    // overflow without changing the softmax probabilities.
                    let max_logit = best_value as f64;
                    let mut sum_exp = 0.0f64;
                    for value in step.iter() {
                        sum_exp += ((*value as f64) - max_logit).exp();
                    }

                    if sum_exp.is_finite() && sum_exp > 0.0 {
                        (((best_value as f64) - max_logit).exp() / sum_exp).clamp(0.0, 1.0) as f32
                    } else {
                        0.0
                    }
                };

                if best_index >= dictionary.len() {
                    if let Some(fallback) = &self.config.fallback_token {
                        text.push_str(fallback);
                        token_indices.push(best_index);
                        fallback_count += 1;
                        probability_sum += probability as f64;
                        confidence_count += 1;
                        previous_symbol = Some(best_index);
                        continue;
                    } else {
                        return Err(CtcGreedyDecoderError::DictionaryIndexMissing {
                            index: best_index,
                            dictionary_size: dictionary.len(),
                        });
                    }
                }

                let token = dictionary
                    .token(best_index)
                    .expect("dictionary bounds verified above");
                text.push_str(token);
                token_indices.push(best_index);

                probability_sum += probability as f64;
                confidence_count += 1;

                previous_symbol = Some(best_index);
            }

            let confidence = if confidence_count == 0 {
                1.0
            } else {
                (probability_sum / confidence_count as f64).clamp(0.0, 1.0) as f32
            };

            results.push(DecodedSequence {
                text,
                token_indices,
                confidence,
                fallback_count,
            });
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn dictionary_from_tokens(tokens: &[&str]) -> RecDictionary {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("ctc_dict_{}.txt", timestamp));
        fs::write(&path, tokens.join("\n")).unwrap();
        let dict = RecDictionary::from_path(&path).unwrap();
        fs::remove_file(path).ok();
        dict
    }

    #[test]
    fn decodes_sequence_with_duplicates_and_blank() {
        let logits = Array3::from_shape_vec(
            (1, 5, 3),
            vec![
                5.0, 0.1, -0.2, //
                -3.0, 4.0, 0.2, //
                -4.0, 3.8, -0.1, //
                -5.0, -2.0, 4.5, //
                4.2, -3.0, -2.0, //
            ],
        )
        .unwrap();

        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig {
            blank_id: 0,
            fallback_token: None,
        });
        let dictionary = dictionary_from_tokens(&["a", "b"]);
        let sequences = decoder
            .decode(&logits, &[5], &dictionary)
            .expect("decoding should succeed");

        assert_eq!(sequences.len(), 1);
        let first = &sequences[0];
        assert_eq!(first.text, "ab");
        assert_eq!(first.token_indices, vec![1, 2]);
        assert!(first.confidence > 0.0);
        assert!(first.confidence <= 1.0);
        assert_eq!(first.fallback_count, 0);
    }

    #[test]
    fn handles_all_blank_sequence() {
        let logits = Array3::from_shape_vec(
            (1, 3, 2),
            vec![
                1.0, 0.1, //
                1.2, 0.0, //
                1.1, -0.5, //
            ],
        )
        .unwrap();

        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig {
            blank_id: 0,
            fallback_token: None,
        });
        let dictionary = dictionary_from_tokens(&["a"]);
        let sequences = decoder
            .decode(&logits, &[3], &dictionary)
            .expect("decoding should succeed");

        assert_eq!(sequences[0].text, "");
        assert!(sequences[0].token_indices.is_empty());
        assert_eq!(sequences[0].confidence, 1.0);
        assert_eq!(sequences[0].fallback_count, 0);
    }

    #[test]
    fn error_when_blank_id_out_of_range() {
        let logits = Array3::<f32>::zeros((1, 1, 2));
        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig {
            blank_id: 3,
            fallback_token: None,
        });
        let dictionary = dictionary_from_tokens(&["a"]);
        let error = decoder
            .decode(&logits, &[1], &dictionary)
            .expect_err("blank id out of range should error");

        assert!(matches!(
            error,
            CtcGreedyDecoderError::BlankIdOutOfRange {
                blank_id: 3,
                class_count: 2
            }
        ));
    }

    #[test]
    fn applies_fallback_when_dictionary_missing() {
        let logits = Array3::from_shape_vec(
            (1, 2, 3),
            vec![
                -0.5, 0.1, 1.0, //
                -0.2, 0.0, 1.2, //
            ],
        )
        .unwrap();

        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig {
            blank_id: 0,
            fallback_token: Some("[UNK]".to_string()),
        });
        let dictionary = dictionary_from_tokens(&["a"]);
        let sequences = decoder
            .decode(&logits, &[2], &dictionary)
            .expect("decoder should fallback instead of error");

        assert_eq!(sequences.len(), 1);
        let first = &sequences[0];
        assert_eq!(first.text, "[UNK]");
        assert_eq!(first.fallback_count, 1);
        assert_eq!(first.token_indices, vec![2]);
    }

    #[test]
    fn computes_confidence_from_probabilities() {
        let probability_rows = [
            [0.92f32, 0.03, 0.03, 0.02], // blank
            [0.05, 0.8, 0.1, 0.05],      // 'a'
            [0.05, 0.7, 0.2, 0.05],      // duplicate 'a' (ignored)
            [0.9, 0.05, 0.03, 0.02],     // blank
            [0.05, 0.05, 0.85, 0.05],    // 'b'
            [0.02, 0.01, 0.02, 0.95],    // 'c'
            [0.9, 0.05, 0.03, 0.02],     // blank
        ];

        let mut logits = Vec::new();
        for row in probability_rows.iter() {
            for value in row.iter() {
                logits.push(value.ln());
            }
        }

        let logits =
            Array3::from_shape_vec((1, probability_rows.len(), 4), logits).expect("shape valid");

        let decoder = CtcGreedyDecoder::new(CtcGreedyDecoderConfig {
            blank_id: 0,
            fallback_token: None,
        });
        let dictionary = dictionary_from_tokens(&["a", "b", "c"]);
        let sequences = decoder
            .decode(&logits, &[probability_rows.len()], &dictionary)
            .expect("decoding should succeed");

        assert_eq!(sequences.len(), 1);
        let first = &sequences[0];
        assert_eq!(first.text, "abc");

        let expected = (0.8f32 + 0.85f32 + 0.95f32) / 3.0;
        let diff = (first.confidence - expected).abs();
        assert!(
            diff < 1e-5,
            "confidence {} differs from expected {}",
            first.confidence,
            expected
        );
    }
}
