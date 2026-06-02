use std::{
    io,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
};

use cuda_core::{CudaModule, CudaStream};
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
}

enum CudaWorkerMessage {
    Run {
        descriptor: CudaLaunchDescriptor,
        complete: oneshot::Sender<Result<(), String>>,
    },
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
                .spawn(move || run_worker(receiver, device_index))
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

    pub(crate) async fn submit(&self, descriptor: CudaLaunchDescriptor) -> AppResult<()> {
        let worker_index = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        let (complete, finished) = oneshot::channel();
        self.senders[worker_index]
            .send(CudaWorkerMessage::Run {
                descriptor,
                complete,
            })
            .await
            .map_err(|_| invalid_worker_config("CUDA worker queue is closed"))?;

        let result = finished
            .await
            .map_err(|_| invalid_worker_config("CUDA worker stopped before completing the job"))?;
        result.map_err(|error| {
            invalid_worker_config(format!(
                "CUDA worker job {} failed: {error}",
                descriptor.label()
            ))
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

fn run_worker(mut receiver: mpsc::Receiver<CudaWorkerMessage>, device_index: usize) {
    let handles =
        crate::cuda_worker_handles_for_device(device_index).map_err(|error| error.to_string());
    while let Some(message) = receiver.blocking_recv() {
        match message {
            CudaWorkerMessage::Run {
                descriptor,
                complete,
            } => {
                let result = match &handles {
                    Ok((stream, module)) => descriptor.run(stream, module),
                    Err(error) => Err(error.clone()),
                };
                let _ = complete.send(result);
            }
        }
    }
}

fn invalid_worker_config(message: impl Into<String>) -> Box<dyn std::error::Error> {
    Box::new(io::Error::new(io::ErrorKind::InvalidInput, message.into()))
}
