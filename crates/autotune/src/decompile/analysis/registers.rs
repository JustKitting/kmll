use super::super::RegisterRef;

pub(super) fn push_register_refs(
    registers: impl IntoIterator<Item = RegisterRef>,
    out: &mut Vec<RegisterRef>,
) {
    for register in registers {
        push_register_ref(register, out);
    }
}

fn push_register_ref(register: RegisterRef, out: &mut Vec<RegisterRef>) {
    if !register.is_pseudo() && !out.contains(&register) {
        out.push(register);
    }
}
