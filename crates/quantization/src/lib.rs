pub use nn_rust_inference::{
    dtypes::Bf16,
    model::{
        AllLinearQuantizationProbe, AttentionFfnQuantizationProbe, AttentionQuantizationProbe,
        FfnQuantizationProbe, OutputProjectionQuantizationProbe, QuantizationGenerationSuiteProbe,
        QuantizationSuiteProbe, QuantizationSuiteVariantProbe,
        run_ministral_all_linear_quantization_probe,
        run_ministral_attention_ffn_quantization_probe, run_ministral_attention_quantization_probe,
        run_ministral_ffn_quantization_probe,
        run_ministral_output_projection_export_quantization_probe,
        run_ministral_output_projection_quantization_probe,
        run_ministral_quantization_generation_suite_probe, run_ministral_quantization_suite_probe,
        validate_ministral_all_linear_int8_export_archive,
    },
    rowwise_scaled::{
        DeviceRowwiseScaledI8Matrix, RowwiseScaledI8Matrix, rowwise_scaled_i8_export_file_paths,
    },
};

pub fn row_symmetric_i8_from_bf16_rows(
    weight: &[Bf16],
    rows: usize,
    cols: usize,
) -> RowwiseScaledI8Matrix {
    RowwiseScaledI8Matrix::from_bf16_rows_symmetric(weight, rows, cols)
}
