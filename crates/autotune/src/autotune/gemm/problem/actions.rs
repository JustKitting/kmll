use super::*;

mod apply;
mod spaces;

impl KernelActionSearchProblem for GemmSearchProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        spaces::search_space(self)
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        spaces::action_spaces(self, candidate)
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        apply::apply_schedule_action(self, candidate, action)
    }
}
