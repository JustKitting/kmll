use std::{
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
    sync::Arc,
};

use cuda_core::{CudaStream, DeviceBuffer, DriverError};

use crate::{dtypes::Bf16, safetensors::Result as SafetensorsResult};

#[derive(Debug, Clone)]
pub struct RowwiseScaledI8Matrix {
    pub values: Vec<i8>,
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl RowwiseScaledI8Matrix {
    pub fn from_bf16_rows_symmetric(weight: &[Bf16], rows: usize, cols: usize) -> Self {
        assert_eq!(
            weight.len(),
            rows * cols,
            "rowwise scaled i8 matrix shape does not match weight length"
        );

        let mut values = Vec::with_capacity(weight.len());
        let mut scales = Vec::with_capacity(rows);

        for row in weight.chunks_exact(cols) {
            let max_abs = row
                .iter()
                .map(|value| value.to_f32().abs())
                .fold(0.0_f32, f32::max);
            let scale = if max_abs == 0.0 { 1.0 } else { max_abs / 127.0 };
            scales.push(scale);

            for value in row {
                let scaled = (value.to_f32() / scale).round().clamp(-127.0, 127.0);
                values.push(scaled as i8);
            }
        }

        Self {
            values,
            scales,
            rows,
            cols,
        }
    }

    pub fn scaled_value(&self, row: usize, col: usize) -> f32 {
        assert!(
            row < self.rows,
            "rowwise scaled i8 matrix row out of bounds"
        );
        assert!(
            col < self.cols,
            "rowwise scaled i8 matrix column out of bounds"
        );
        self.values[row * self.cols + col] as f32 * self.scales[row]
    }

    pub fn from_i8_rows_symmetric_parts(
        values: Vec<i8>,
        scales: Vec<f32>,
        rows: usize,
        cols: usize,
    ) -> SafetensorsResult<Self> {
        if values.len() != rows * cols {
            return Err(invalid_data(format!(
                "rowwise scaled i8 matrix values length {} does not match shape [{rows}, {cols}]",
                values.len()
            ))
            .into());
        }
        if scales.len() != rows {
            return Err(invalid_data(format!(
                "rowwise scaled i8 matrix scales length {} does not match row count {rows}",
                scales.len()
            ))
            .into());
        }

        Ok(Self {
            values,
            scales,
            rows,
            cols,
        })
    }

    pub fn read_exported(
        export_dir: impl AsRef<Path>,
        index: usize,
        name: &str,
        rows: usize,
        cols: usize,
    ) -> SafetensorsResult<Self> {
        let (values_path, scales_path) =
            rowwise_scaled_i8_export_file_paths(export_dir, index, name);
        let values = read_i8_file(&values_path)?;
        let scales = read_f32_file(&scales_path)?;
        Self::from_i8_rows_symmetric_parts(values, scales, rows, cols)
    }

    pub fn matvec_cpu(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(
            input.len(),
            self.cols,
            "rowwise scaled i8 matvec input length mismatch"
        );

        let mut output = vec![0.0; self.rows];
        for (row, out) in output.iter_mut().enumerate() {
            let row_offset = row * self.cols;
            let scale = self.scales[row];
            let mut acc = 0.0_f32;
            for (col, x) in input.iter().copied().enumerate() {
                acc += self.values[row_offset + col] as f32 * scale * x;
            }
            *out = acc;
        }
        output
    }

    pub fn to_device(
        &self,
        stream: &Arc<CudaStream>,
    ) -> Result<DeviceRowwiseScaledI8Matrix, DriverError> {
        Ok(DeviceRowwiseScaledI8Matrix {
            values: DeviceBuffer::from_host(stream, &self.values)?,
            scales: DeviceBuffer::from_host(stream, &self.scales)?,
            rows: self.rows,
            cols: self.cols,
        })
    }
}

pub fn rowwise_scaled_i8_export_file_paths(
    export_dir: impl AsRef<Path>,
    index: usize,
    name: &str,
) -> (PathBuf, PathBuf) {
    let stem = format!("{index:04}_{}", rowwise_scaled_i8_export_file_stem(name));
    (
        export_dir.as_ref().join(format!("{stem}.i8")),
        export_dir.as_ref().join(format!("{stem}.scales.f32")),
    )
}

fn rowwise_scaled_i8_export_file_stem(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

fn read_i8_file(path: &Path) -> SafetensorsResult<Vec<i8>> {
    Ok(fs::read(path)?.into_iter().map(|byte| byte as i8).collect())
}

fn read_f32_file(path: &Path) -> SafetensorsResult<Vec<f32>> {
    let bytes = fs::read(path)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid_data(format!(
            "f32 file {} has {} bytes, expected a multiple of 4",
            path.display(),
            bytes.len()
        ))
        .into());
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

fn invalid_data(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

pub struct DeviceRowwiseScaledI8Matrix {
    pub values: DeviceBuffer<i8>,
    pub scales: DeviceBuffer<f32>,
    pub rows: usize,
    pub cols: usize,
}
