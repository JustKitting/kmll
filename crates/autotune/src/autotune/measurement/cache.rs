use super::super::{KernelArtifactStore, KernelCandidateMetadata, SearchScore};
use super::KernelAutotuneMeasureResult;

pub fn cached_measured_score<F>(
    store: &KernelArtifactStore,
    score_namespace: &str,
    candidate: &KernelCandidateMetadata,
    first_measure_error: &mut Option<String>,
    score_cache_error: &mut Option<String>,
    mut measure_candidate: F,
) -> Option<SearchScore>
where
    F: FnMut(&KernelCandidateMetadata) -> KernelAutotuneMeasureResult<Option<SearchScore>>,
{
    if score_cache_error.is_some() {
        return None;
    }
    match store.read_score_cache_for_candidate(score_namespace, candidate) {
        Ok(Some(score)) => return Some(score),
        Ok(None) => {}
        Err(error) => {
            if score_cache_error.is_none() {
                *score_cache_error = Some(error.to_string());
            }
            return None;
        }
    }

    let score = match measure_candidate(candidate) {
        Ok(Some(score)) => score,
        Ok(None) => return None,
        Err(error) => {
            if first_measure_error.is_none() {
                *first_measure_error = Some(error.to_string());
            }
            return None;
        }
    };

    let mut scored_candidate = candidate.clone();
    scored_candidate.score = Some(score);
    if let Err(error) = store.emit_score_cache_for_candidate(score_namespace, &scored_candidate) {
        if score_cache_error.is_none() {
            *score_cache_error = Some(error.to_string());
        }
        return None;
    }
    Some(score)
}
