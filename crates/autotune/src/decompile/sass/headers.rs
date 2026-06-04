pub(super) fn parse_text_section(line: &str) -> Option<String> {
    let rest = line.strip_prefix(".section")?.trim();
    let section = rest.split(',').next()?.trim();
    section
        .strip_prefix(".text.")
        .map(|name| name.trim().to_string())
}

pub(super) fn parse_global(line: &str) -> Option<String> {
    line.strip_prefix(".global")
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
}

pub(super) fn parse_cuobjdump_function(line: &str) -> Option<String> {
    let rest = line.strip_prefix("Function")?.trim_start();
    let name = rest.strip_prefix(':')?.trim();
    (!name.is_empty()).then(|| name.to_string())
}

pub(super) fn is_label_line(line: &str) -> bool {
    line.ends_with(':')
        && !line.contains(' ')
        && !line.contains('\t')
        && !line.starts_with(".type")
        && !line.starts_with(".size")
}
