#[cfg(test)]
mod tests {
    use super::*;

    fn test_generated_root() -> PathBuf {
        runtime::default_artifact_dir()
            .join("test-generated")
            .join(format!(
                "{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock should be after unix epoch")
                    .as_nanos()
            ))
    }

    fn remove_test_generated_root(root: &Path) {
        fs::remove_dir_all(root).expect("test-generated kernel artifact directory should clean up");
        if let Some(parent) = root.parent() {
            let _ = fs::remove_dir(parent);
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct MatvecWithoutUnrollSpace(MatvecSearchProblem);

    impl KernelMetadataSearchProblem for MatvecWithoutUnrollSpace {
        fn seed(&self) -> KernelCandidateMetadata {
            self.0.seed()
        }

        fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
            expand_with_schedule_actions(self, candidate)
        }

        fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
            self.0.score(candidate)
        }
    }

    impl KernelActionSearchProblem for MatvecWithoutUnrollSpace {
        fn search_space(&self) -> KernelActionSpaceSet {
            let mut spaces = self.0.search_space();
            spaces
                .spaces
                .retain(|space| !matches!(space, KernelActionSpace::Unroll { .. }));
            spaces
        }

        fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
            let mut spaces = self.0.action_spaces(candidate);
            spaces
                .spaces
                .retain(|space| !matches!(space, KernelActionSpace::Unroll { .. }));
            spaces
        }

        fn apply_schedule_action(
            &self,
            candidate: &KernelCandidateMetadata,
            action: &KernelScheduleAction,
        ) -> Option<KernelCandidateMetadata> {
            if matches!(action.op, KernelScheduleActionOp::Unroll) {
                return None;
            }
            self.0.apply_schedule_action(candidate, action)
        }
    }

    include!("tests/matvec.rs");
    include!("tests/gemm.rs");
    include!("tests/search.rs");
    include!("tests/operation.rs");
    include!("tests/artifacts.rs");
}
