use super::super::super::*;

pub(in crate::autotune) fn required_field<'a>(
    value: &'a Value,
    name: &str,
) -> Result<&'a Value, KernelGenerationError> {
    value
        .get(name)
        .ok_or_else(|| invalid_selection(format!("missing field {name:?}")))
}

pub(in crate::autotune) fn required_str<'a>(
    value: &'a Value,
    name: &str,
) -> Result<&'a str, KernelGenerationError> {
    required_field(value, name)?
        .as_str()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a string")))
}

pub(in crate::autotune) fn required_bool(
    value: &Value,
    name: &str,
) -> Result<bool, KernelGenerationError> {
    required_field(value, name)?
        .as_bool()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a bool")))
}

pub(in crate::autotune) fn required_array<'a>(
    value: &'a Value,
    name: &str,
) -> Result<&'a [Value], KernelGenerationError> {
    required_field(value, name)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be an array")))
}

pub(in crate::autotune) fn required_u64(
    value: &Value,
    name: &str,
) -> Result<u64, KernelGenerationError> {
    required_field(value, name)?
        .as_u64()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be a u64")))
}

pub(in crate::autotune) fn required_usize(
    value: &Value,
    name: &str,
) -> Result<usize, KernelGenerationError> {
    usize::try_from(required_u64(value, name)?)
        .map_err(|_| invalid_selection(format!("field {name:?} exceeds usize")))
}

pub(in crate::autotune) fn required_u32(
    value: &Value,
    name: &str,
) -> Result<u32, KernelGenerationError> {
    u32::try_from(required_u64(value, name)?)
        .map_err(|_| invalid_selection(format!("field {name:?} exceeds u32")))
}

pub(in crate::autotune) fn required_u8(
    value: &Value,
    name: &str,
) -> Result<u8, KernelGenerationError> {
    value_as_u8(required_field(value, name)?)
        .ok_or_else(|| invalid_selection(format!("field {name:?} must fit in u8")))
}

pub(in crate::autotune) fn optional_u8(
    value: &Value,
    name: &str,
) -> Result<Option<u8>, KernelGenerationError> {
    match value.get(name) {
        Some(Value::Null) | None => Ok(None),
        Some(value) => value_as_u8(value)
            .ok_or_else(|| invalid_selection(format!("field {name:?} must be null or a u8")))
            .map(Some),
    }
}

pub(in crate::autotune) fn value_as_u8(value: &Value) -> Option<u8> {
    value.as_u64().and_then(|value| u8::try_from(value).ok())
}

pub(in crate::autotune) fn required_f64(
    value: &Value,
    name: &str,
) -> Result<f64, KernelGenerationError> {
    required_field(value, name)?
        .as_f64()
        .ok_or_else(|| invalid_selection(format!("field {name:?} must be an f64")))
}

pub(in crate::autotune) fn invalid_selection(reason: impl Into<String>) -> KernelGenerationError {
    KernelGenerationError::InvalidSelection {
        reason: reason.into(),
    }
}
