use std::fmt::Write as _;

use crate::autotune::GemmSchedulePlan;

impl GemmSchedulePlan {
    pub(in crate::autotune) fn per_thread_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.m_per_thread > 1 {
            write!(&mut suffix, "_mt{}", plan.m_per_thread).expect("write to string");
        }
        if plan.n_per_thread > 1 {
            write!(&mut suffix, "_nt{}", plan.n_per_thread).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn per_thread_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.m_per_thread > 1 {
            write!(&mut suffix, "-mt{}", plan.m_per_thread).expect("write to string");
        }
        if plan.n_per_thread > 1 {
            write!(&mut suffix, "-nt{}", plan.n_per_thread).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_unroll_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.a_load_unroll > 1 {
            write!(&mut suffix, "_au{}", plan.a_load_unroll).expect("write to string");
        }
        if plan.b_load_unroll > 1 {
            write!(&mut suffix, "_bu{}", plan.b_load_unroll).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_unroll_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.a_load_unroll > 1 {
            write!(&mut suffix, "-au{}", plan.a_load_unroll).expect("write to string");
        }
        if plan.b_load_unroll > 1 {
            write!(&mut suffix, "-bu{}", plan.b_load_unroll).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn reduce_group_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_reduce_group() {
            write!(&mut suffix, "_kg{}", plan.reduce_group_size()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn reduce_group_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_reduce_group() {
            write!(&mut suffix, "-kg{}", plan.reduce_group_size()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_thread_group_symbol_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_a_load_thread_group() {
            write!(&mut suffix, "_atg{}", plan.a_load_thread_count()).expect("write to string");
        }
        if plan.has_custom_b_load_thread_group() {
            write!(&mut suffix, "_btg{}", plan.b_load_thread_count()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn load_thread_group_operation_suffix(self) -> String {
        let plan = self.normalized();
        let mut suffix = String::new();
        if plan.has_custom_a_load_thread_group() {
            write!(&mut suffix, "-atg{}", plan.a_load_thread_count()).expect("write to string");
        }
        if plan.has_custom_b_load_thread_group() {
            write!(&mut suffix, "-btg{}", plan.b_load_thread_count()).expect("write to string");
        }
        suffix
    }

    pub(in crate::autotune) fn thread_order_symbol_suffix(self) -> &'static str {
        self.normalized().thread_order.symbol_suffix()
    }

    pub(in crate::autotune) fn thread_order_operation_suffix(self) -> &'static str {
        self.normalized().thread_order.operation_suffix()
    }
}
