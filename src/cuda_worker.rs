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

type CudaWorkerJob =
    Box<dyn FnOnce(&Arc<CudaStream>, &Arc<CudaModule>) -> Result<(), String> + Send + 'static>;

enum CudaWorkerMessage {
    Run {
        job: CudaWorkerJob,
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

    pub(crate) async fn submit<F>(&self, job: F) -> AppResult<()>
    where
        F: FnOnce(&Arc<CudaStream>, &Arc<CudaModule>) -> Result<(), String> + Send + 'static,
    {
        let worker_index = self.next_worker.fetch_add(1, Ordering::Relaxed) % self.senders.len();
        let (complete, finished) = oneshot::channel();
        self.senders[worker_index]
            .send(CudaWorkerMessage::Run {
                job: Box::new(job),
                complete,
            })
            .await
            .map_err(|_| invalid_worker_config("CUDA worker queue is closed"))?;

        let result = finished
            .await
            .map_err(|_| invalid_worker_config("CUDA worker stopped before completing the job"))?;
        result.map_err(|error| invalid_worker_config(format!("CUDA worker job failed: {error}")))
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
            CudaWorkerMessage::Run { job, complete } => {
                let result = match &handles {
                    Ok((stream, module)) => job(stream, module),
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
