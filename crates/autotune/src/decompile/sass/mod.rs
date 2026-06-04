mod parser;
mod types;

pub use self::{
    parser::parse_nvidia_sass,
    types::{
        RegisterClass, SassFunction, SassInstruction, SassModule, SassOperand, SassOperandKind,
        SassParseError, SassPredicate, SassRegister, SassSourcePosition,
    },
};

pub(super) fn label_in_text(raw: &str) -> Option<String> {
    parser::label_in_text(raw)
}
