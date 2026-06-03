#[derive(Debug, Clone, PartialEq)]
pub struct InferenceKernelAutoOptimize {
    pub operation: TypedOperationSpec,
    pub problem: InferenceKernelOptimizationProblem,
    pub result: AutoOptimizeResult,
}

impl InferenceKernelAutoOptimize {
    pub fn best_candidate(&self) -> Option<&KernelCandidateMetadata> {
        self.result.best.as_ref()
    }

    pub fn auto_optimization_report(
        &self,
        config: AutoOptimizeConfig,
    ) -> AutoOptimizationSearchReport {
        self.result.auto_optimization_report_with_action_space(
            self.problem.family(),
            config,
            &self.problem.search_space(),
        )
    }

    pub fn render_best_source(&self) -> Result<GeneratedKernelSource, KernelGenerationError> {
        let candidate =
            self.best_candidate()
                .ok_or_else(|| KernelGenerationError::NoOptimizationCandidate {
                    name: self.operation.name.clone(),
                    kind: self.operation.kind,
                })?;
        InferenceKernelRustCudaGenerator.source_for(candidate)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedInferenceKernelAutoOptimize {
    pub optimization: InferenceKernelAutoOptimize,
    pub cache_key: KernelOptimizationCacheKey,
    pub cache_status: SelectionCacheStatus,
    pub cache_write: Option<EmittedKernelOptimizationSelection>,
}

impl CachedInferenceKernelAutoOptimize {
    pub fn result(&self) -> &AutoOptimizeResult {
        &self.optimization.result
    }

    pub fn best_candidate(&self) -> Option<&KernelCandidateMetadata> {
        self.optimization.best_candidate()
    }

    pub fn auto_optimization_report(
        &self,
        config: AutoOptimizeConfig,
    ) -> AutoOptimizationSearchReport {
        self.optimization.auto_optimization_report(config)
    }

    pub fn render_best_source(&self) -> Result<GeneratedKernelSource, KernelGenerationError> {
        self.optimization.render_best_source()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeneratedInferenceKernelSource {
    pub optimization: InferenceKernelAutoOptimize,
    pub candidate: KernelCandidateMetadata,
    pub source: GeneratedKernelSource,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CachedGeneratedInferenceKernelSource {
    pub optimization: CachedInferenceKernelAutoOptimize,
    pub candidate: KernelCandidateMetadata,
    pub source: GeneratedKernelSource,
}

pub fn auto_optimize_inference_kernel(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
) -> Result<InferenceKernelAutoOptimize, KernelGenerationError> {
    let problem = InferenceKernelOptimizationProblem::from_operation(operation)?;
    let result = auto_optimize_metadata(&problem, config);
    Ok(InferenceKernelAutoOptimize {
        operation: operation.clone(),
        problem,
        result,
    })
}

pub fn auto_optimize_inference_kernel_with_scorer<F>(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    mut score_candidate: F,
) -> Result<InferenceKernelAutoOptimize, KernelGenerationError>
where
    F: FnMut(&KernelCandidateMetadata) -> Option<SearchScore>,
{
    let problem = InferenceKernelOptimizationProblem::from_operation(operation)?;
    let result = auto_optimize_metadata_with_scorer(&problem, config, |candidate| {
        score_candidate(candidate)
    });
    Ok(InferenceKernelAutoOptimize {
        operation: operation.clone(),
        problem,
        result,
    })
}

pub fn auto_optimize_inference_kernel_with_selection_cache(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> Result<CachedInferenceKernelAutoOptimize, KernelGenerationError> {
    auto_optimize_inference_kernel_with_selection_cache_scorer(
        store,
        operation,
        config,
        score_namespace,
        |candidate, problem| problem.score(candidate),
    )
}

pub fn auto_optimize_inference_kernel_with_selection_cache_scorer<F>(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    score_namespace: &str,
    mut score_candidate: F,
) -> Result<CachedInferenceKernelAutoOptimize, KernelGenerationError>
where
    F: FnMut(&KernelCandidateMetadata, &InferenceKernelOptimizationProblem) -> Option<SearchScore>,
{
    let problem = InferenceKernelOptimizationProblem::from_operation(operation)?;
    let cached = auto_optimize_metadata_with_selection_cache(
        store,
        &problem,
        config,
        score_namespace,
        |candidate| score_candidate(candidate, &problem),
    )?;
    Ok(CachedInferenceKernelAutoOptimize {
        optimization: InferenceKernelAutoOptimize {
            operation: operation.clone(),
            problem,
            result: cached.result,
        },
        cache_key: cached.cache_key,
        cache_status: cached.cache_status,
        cache_write: cached.cache_write,
    })
}

pub fn generate_inference_kernel_source(
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
) -> Result<GeneratedInferenceKernelSource, KernelGenerationError> {
    let optimization = auto_optimize_inference_kernel(operation, config)?;
    let candidate = optimization.best_candidate().cloned().ok_or_else(|| {
        KernelGenerationError::NoOptimizationCandidate {
            name: operation.name.clone(),
            kind: operation.kind,
        }
    })?;
    let source = InferenceKernelRustCudaGenerator.source_for(&candidate)?;
    Ok(GeneratedInferenceKernelSource {
        optimization,
        candidate,
        source,
    })
}

pub fn generate_inference_kernel_source_with_selection_cache(
    store: &KernelArtifactStore,
    operation: &TypedOperationSpec,
    config: AutoOptimizeConfig,
    score_namespace: &str,
) -> Result<CachedGeneratedInferenceKernelSource, KernelGenerationError> {
    let optimization =
        auto_optimize_inference_kernel_with_selection_cache(store, operation, config, score_namespace)?;
    let candidate = optimization.best_candidate().cloned().ok_or_else(|| {
        KernelGenerationError::NoOptimizationCandidate {
            name: operation.name.clone(),
            kind: operation.kind,
        }
    })?;
    let source = InferenceKernelRustCudaGenerator.source_for(&candidate)?;
    Ok(CachedGeneratedInferenceKernelSource {
        optimization,
        candidate,
        source,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceKernelOptimizationProblem {
    MatvecBf16RowMajor(MatvecSearchProblem),
    GemmF32Bf16RowColRow(GemmSearchProblem),
}

impl InferenceKernelOptimizationProblem {
    pub fn from_operation(operation: &TypedOperationSpec) -> Result<Self, KernelGenerationError> {
        if operation.route != OperationRoute::CudaKernel {
            return Err(unsupported_operation(
                operation,
                format!(
                    "route {} is not a generated CUDA kernel route",
                    operation.route.label()
                ),
            ));
        }

        match operation.kind {
            OperationKind::Matvec => Self::matvec_from_operation(operation),
            OperationKind::Gemm => Self::gemm_from_operation(operation),
            kind => Err(unsupported_operation(
                operation,
                format!("operation kind {} has no inference autotune problem", kind.label()),
            )),
        }
    }

    pub const fn family(self) -> &'static str {
        match self {
            Self::MatvecBf16RowMajor(_) => "matvec-bf16-row-major",
            Self::GemmF32Bf16RowColRow(_) => "gemm-f32-bf16-row-col-row",
        }
    }

    fn matvec_from_operation(
        operation: &TypedOperationSpec,
    ) -> Result<Self, KernelGenerationError> {
        let [input, weight] = tensor_pair(operation.inputs.as_slice()).ok_or_else(|| {
            unsupported_operation(operation, "matvec requires exactly two input tensors")
        })?;
        let output = single_tensor(operation.outputs.as_slice()).ok_or_else(|| {
            unsupported_operation(operation, "matvec requires exactly one output tensor")
        })?;
        let [cols] = tensor_shape_1(input).ok_or_else(|| {
            unsupported_operation(operation, "matvec input must have shape [cols]")
        })?;
        let [rows, weight_cols] = tensor_shape_2(weight).ok_or_else(|| {
            unsupported_operation(operation, "matvec weight must have shape [rows, cols]")
        })?;
        let [output_rows] = tensor_shape_1(output).ok_or_else(|| {
            unsupported_operation(operation, "matvec output must have shape [rows]")
        })?;
        if weight_cols != cols || output_rows != rows {
            return Err(unsupported_operation(
                operation,
                "matvec tensor shapes must satisfy [cols] x [rows, cols] -> [rows]",
            ));
        }
        if input.dtype != NumericKind::F32
            || input.accumulator != NumericKind::F32
            || weight.dtype != NumericKind::Bf16
            || weight.accumulator != NumericKind::F32
            || output.dtype != NumericKind::F32
            || output.accumulator != NumericKind::F32
        {
            return Err(unsupported_operation(
                operation,
                "matvec currently supports f32 input, bf16 row-major weights, and f32 output",
            ));
        }
        if !layout_is(input, "contiguous")
            || !layout_is(weight, "row-major")
            || !layout_is(output, "contiguous")
        {
            return Err(unsupported_operation(
                operation,
                "matvec currently supports contiguous input/output and row-major weights",
            ));
        }

        Ok(Self::MatvecBf16RowMajor(
            MatvecSearchProblem::bf16_row_major(rows, cols),
        ))
    }

    fn gemm_from_operation(operation: &TypedOperationSpec) -> Result<Self, KernelGenerationError> {
        let [a, b] = tensor_pair(operation.inputs.as_slice()).ok_or_else(|| {
            unsupported_operation(operation, "GEMM requires exactly two input tensors")
        })?;
        let c = single_tensor(operation.outputs.as_slice()).ok_or_else(|| {
            unsupported_operation(operation, "GEMM requires exactly one output tensor")
        })?;
        let [m, k] = tensor_shape_2(a).ok_or_else(|| {
            unsupported_operation(operation, "GEMM lhs must have shape [m, k]")
        })?;
        let [b_k, n] = tensor_shape_2(b).ok_or_else(|| {
            unsupported_operation(operation, "GEMM rhs must have shape [k, n]")
        })?;
        let [c_m, c_n] = tensor_shape_2(c).ok_or_else(|| {
            unsupported_operation(operation, "GEMM output must have shape [m, n]")
        })?;
        if b_k != k || c_m != m || c_n != n {
            return Err(unsupported_operation(
                operation,
                "GEMM tensor shapes must satisfy [m, k] x [k, n] -> [m, n]",
            ));
        }
        if a.dtype != NumericKind::F32
            || a.accumulator != NumericKind::F32
            || b.dtype != NumericKind::Bf16
            || b.accumulator != NumericKind::F32
            || c.dtype != NumericKind::F32
            || c.accumulator != NumericKind::F32
        {
            return Err(unsupported_operation(
                operation,
                "GEMM currently supports f32 lhs, bf16 rhs, and f32 output",
            ));
        }
        if !layout_is(a, "row-major") || !layout_is(b, "column-major") || !layout_is(c, "row-major")
        {
            return Err(unsupported_operation(
                operation,
                "GEMM currently supports row-major lhs/output and column-major rhs",
            ));
        }

        Ok(Self::GemmF32Bf16RowColRow(
            GemmSearchProblem::f32_bf16_row_col_row(m, n, k),
        ))
    }
}

impl KernelMetadataSearchProblem for InferenceKernelOptimizationProblem {
    fn seed(&self) -> KernelCandidateMetadata {
        match self {
            Self::MatvecBf16RowMajor(problem) => problem.seed(),
            Self::GemmF32Bf16RowColRow(problem) => problem.seed(),
        }
    }

    fn expand(&self, candidate: &KernelCandidateMetadata) -> Vec<KernelCandidateMetadata> {
        match self {
            Self::MatvecBf16RowMajor(problem) => problem.expand(candidate),
            Self::GemmF32Bf16RowColRow(problem) => problem.expand(candidate),
        }
    }

    fn score(&self, candidate: &KernelCandidateMetadata) -> Option<SearchScore> {
        match self {
            Self::MatvecBf16RowMajor(problem) => problem.score(candidate),
            Self::GemmF32Bf16RowColRow(problem) => problem.score(candidate),
        }
    }
}

impl KernelActionSearchProblem for InferenceKernelOptimizationProblem {
    fn search_space(&self) -> KernelActionSpaceSet {
        match self {
            Self::MatvecBf16RowMajor(problem) => problem.search_space(),
            Self::GemmF32Bf16RowColRow(problem) => problem.search_space(),
        }
    }

    fn action_spaces(&self, candidate: &KernelCandidateMetadata) -> KernelActionSpaceSet {
        match self {
            Self::MatvecBf16RowMajor(problem) => problem.action_spaces(candidate),
            Self::GemmF32Bf16RowColRow(problem) => problem.action_spaces(candidate),
        }
    }

    fn apply_schedule_action(
        &self,
        candidate: &KernelCandidateMetadata,
        action: &KernelScheduleAction,
    ) -> Option<KernelCandidateMetadata> {
        match self {
            Self::MatvecBf16RowMajor(problem) => {
                problem.apply_schedule_action(candidate, action)
            }
            Self::GemmF32Bf16RowColRow(problem) => {
                problem.apply_schedule_action(candidate, action)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct InferenceKernelRustCudaGenerator;

impl KernelSourceGenerator for InferenceKernelRustCudaGenerator {
    fn name(&self) -> &'static str {
        "inference-rust-cuda-source-generator"
    }

    fn source_for(
        &self,
        candidate: &KernelCandidateMetadata,
    ) -> Result<GeneratedKernelSource, KernelGenerationError> {
        match candidate.family.as_str() {
            "matvec-bf16-row-major" => MatvecRustCudaGenerator.source_for(candidate),
            "gemm-f32-bf16-row-col-row" => GemmRustCudaGenerator.source_for(candidate),
            _ => Err(KernelGenerationError::UnsupportedCandidate {
                family: candidate.family.clone(),
                generator: self.name(),
            }),
        }
    }
}

fn unsupported_operation(
    operation: &TypedOperationSpec,
    reason: impl Into<String>,
) -> KernelGenerationError {
    KernelGenerationError::UnsupportedOperation {
        name: operation.name.clone(),
        kind: operation.kind,
        reason: reason.into(),
    }
}

fn tensor_pair(tensors: &[TensorTypeSpec]) -> Option<[&TensorTypeSpec; 2]> {
    match tensors {
        [lhs, rhs] => Some([lhs, rhs]),
        _ => None,
    }
}

fn single_tensor(tensors: &[TensorTypeSpec]) -> Option<&TensorTypeSpec> {
    match tensors {
        [tensor] => Some(tensor),
        _ => None,
    }
}

fn tensor_shape_1(tensor: &TensorTypeSpec) -> Option<[usize; 1]> {
    match tensor.shape.as_slice() {
        [x] => Some([*x]),
        _ => None,
    }
}

fn tensor_shape_2(tensor: &TensorTypeSpec) -> Option<[usize; 2]> {
    match tensor.shape.as_slice() {
        [x, y] => Some([*x, *y]),
        _ => None,
    }
}

fn layout_is(tensor: &TensorTypeSpec, expected: &str) -> bool {
    tensor.layout.as_deref() == Some(expected)
}
