use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::Instant,
};

use cuda_core::{CudaModule, CudaStream, LaunchConfig};
use nn_rust_profiling::{
    CudaLaunchSpec, F32ProfileType, OperationKind, OperationRoute, ProfileDuration,
    QueueOperationProfile, TensorType, TypedOperationSpec,
};
use tokio::sync::{mpsc, oneshot};

use crate::AppResult;

pub(crate) const SMOKE_LAUNCH_TAPE: [CudaLaunchDescriptor; 3] = [
    CudaLaunchDescriptor::new(CudaLaunchOp::SmokeRelu),
    CudaLaunchDescriptor::new(CudaLaunchOp::SmokeSwiglu),
    CudaLaunchDescriptor::new(CudaLaunchOp::SmokeVecAdd),
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CudaLaunchDescriptor {
    op: CudaLaunchOp,
}

impl CudaLaunchDescriptor {
    pub(crate) const fn new(op: CudaLaunchOp) -> Self {
        Self { op }
    }

    fn label(self) -> &'static str {
        self.op.label()
    }

    fn operation_spec(self) -> TypedOperationSpec {
        self.op.operation_spec()
    }

    fn run(self, stream: &Arc<CudaStream>, module: &Arc<CudaModule>) -> Result<(), String> {
        match self.op {
            CudaLaunchOp::SmokeRelu => crate::run_relu(stream, module),
            CudaLaunchOp::SmokeSwiglu => crate::run_swiglu(stream, module),
            CudaLaunchOp::SmokeVecAdd => crate::run_vecadd(stream, module),
        }
        .map_err(|error| error.to_string())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaLaunchOp {
    SmokeRelu,
    SmokeSwiglu,
    SmokeVecAdd,
}

impl CudaLaunchOp {
    fn label(self) -> &'static str {
        match self {
            Self::SmokeRelu => "smoke-relu",
            Self::SmokeSwiglu => "smoke-swiglu",
            Self::SmokeVecAdd => "smoke-vecadd",
        }
    }

    fn kernel_name(self) -> &'static str {
        match self {
            Self::SmokeRelu => "relu_kernel",
            Self::SmokeSwiglu => "swiglu_kernel",
            Self::SmokeVecAdd => "vecadd_kernel",
        }
    }

    fn operation_spec(self) -> TypedOperationSpec {
        let vector = TensorType::<F32ProfileType, 1>::new([crate::N])
            .with_static_layout("contiguous")
            .erase();
        let launch = LaunchConfig::for_num_elems(crate::N as u32);
        let operation = TypedOperationSpec::new(
            self.label(),
            OperationKind::Elementwise,
            OperationRoute::TokioCudaWorker,
        )
        .with_launch(CudaLaunchSpec::new(
            self.kernel_name(),
            launch.grid_dim,
            launch.block_dim,
            launch.shared_mem_bytes,
        ));

        match self {
            Self::SmokeRelu | Self::SmokeSwiglu => {
                operation.with_input(vector.clone()).with_output(vector)
            }
            Self::SmokeVecAdd => operation
                .with_input(vector.clone())
                .with_input(vector.clone())
                .with_output(vector),
        }
    }
}

enum CudaWorkerMessage {
    Run {
        descriptor: CudaLaunchDescriptor,
        queued_at: Instant,
        complete: oneshot::Sender<CudaWorkerJobResult>,
    },
}

struct CudaWorkerJobResult {
    result: Result<(), String>,
    queue_to_worker: ProfileDuration,
    worker_execute: ProfileDuration,
}

pub(crate) struct CudaWorkerPool {
    senders: Vec<mpsc::Sender<CudaWorkerMessage>>,
    joins: Vec<thread::JoinHandle<()>>,
    next_worker: AtomicUsize,
}

impl CudaWorkerPool {
    pub(crate) fn new(
        worker_count: usize,
        queue_depth: usize,
        device_index: usize,
    ) -> AppResult<Self> {
        if worker_count == 0 {
            return Err(invalid_worker_config("CUDA worker count must be nonzero"));
        }
        if queue_depth == 0 {
            return Err(invalid_worker_config(
                "CUDA worker queue depth must be nonzero",
            ));
        }

        let mut senders = Vec::with_capacity(worker_count);
        let mut joins = Vec::with_capacity(worker_count);
        for worker_index in 0..worker_count {
            let (sender, receiver) = mpsc::channel(queue_depth);
            let join = thread::Builder::new()
                .name(format!("cuda-launch-worker-{worker_index}"))
                .spawn(move || run_worker(receiver, device_index, worker_index))
                .map_err(|error| {
                    invalid_worker_config(format!("failed to spawn CUDA worker thread: {error}"))
                })?;
            senders.push(sender);
            joins.push(join);
        }

        Ok(Self {
            senders,
            joins,
            next_worker: AtomicUsize::new(0),
        })
    }

    pub(crate) fn worker_count(&self) -> usize {
        self.senders.len()
    }

    pub(crate) async fn submit_profiled(
        &self,
        descriptor: CudaLaunchDescriptor,
    ) -> AppResult<QueueOperationProfile> {
        let worker_index = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        let operation = descriptor.operation_spec();
        let total_started_at = Instant::now();
        let (complete, finished) = oneshot::channel();
        let send_started_at = Instant::now();
        let queued_at = send_started_at;
        self.senders[worker_index]
            .send(CudaWorkerMessage::Run {
                descriptor,
                queued_at,
                complete,
            })
            .await
            .map_err(|_| invalid_worker_config("CUDA worker queue is closed"))?;
        let send_wait = ProfileDuration::from_duration(send_started_at.elapsed());

        let completion_started_at = Instant::now();
        let job = finished
            .await
            .map_err(|_| invalid_worker_config("CUDA worker stopped before completing the job"))?;
        let completion_wait = ProfileDuration::from_duration(completion_started_at.elapsed());
        job.result.map_err(|error| {
            invalid_worker_config(format!(
                "CUDA worker job {} failed: {error}",
                descriptor.label()
            ))
        })?;

        Ok(QueueOperationProfile {
            operation,
            worker_index,
            send_wait,
            queue_to_worker: job.queue_to_worker,
            worker_execute: job.worker_execute,
            completion_wait,
            total: ProfileDuration::from_duration(total_started_at.elapsed()),
        })
    }
}

impl Drop for CudaWorkerPool {
    fn drop(&mut self) {
        self.senders.clear();
        for join in self.joins.drain(..) {
            let _ = join.join();
        }
    }
}

fn run_worker(
    mut receiver: mpsc::Receiver<CudaWorkerMessage>,
    device_index: usize,
    _worker_index: usize,
) {
    let handles =
        crate::cuda_worker_handles_for_device(device_index).map_err(|error| error.to_string());
    while let Some(message) = receiver.blocking_recv() {
        match message {
            CudaWorkerMessage::Run {
                descriptor,
                queued_at,
                complete,
            } => {
                let received_at = Instant::now();
                let execute_started_at = Instant::now();
                let result = match &handles {
                    Ok((stream, module)) => descriptor.run(stream, module),
                    Err(error) => Err(error.clone()),
                };
                let _ = complete.send(CudaWorkerJobResult {
                    result,
                    queue_to_worker: ProfileDuration::from_duration(
                        received_at.duration_since(queued_at),
                    ),
                    worker_execute: ProfileDuration::from_duration(execute_started_at.elapsed()),
                });
            }
        }
    }
}

fn invalid_worker_config(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}
