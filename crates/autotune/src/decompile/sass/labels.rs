pub(in crate::decompile) fn label_in_text(raw: &str) -> Option<String> {
    let start = raw.find("`(")?;
    let rest = &raw[start + 2..];
    let end = rest.find(')')?;
    Some(rest[..end].to_string())
}
