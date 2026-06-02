#!/bin/bash
# Build and run nn-rust with cuda-oxide
# Usage: ./run.sh [build|run|clean]

set -e

export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
export CUDA_TOOLKIT_PATH=/opt/cuda
export BINDGEN_EXTRA_CLANG_ARGS="-I/opt/cuda/include"
export CUDA_OXIDE_BACKEND=${CUDA_OXIDE_BACKEND:-~/.cargo/cuda-oxide/librustc_codegen_cuda.so}

cd "$(dirname "$0")"
export NN_RUST_CUDA_ARTIFACT_DIR="$PWD/target/cuda-oxide/inference"
export CUDA_OXIDE_PTX_DIR="$NN_RUST_CUDA_ARTIFACT_DIR"

prepare_cuda_artifact_dir() {
    mkdir -p "$NN_RUST_CUDA_ARTIFACT_DIR"
}

case "${1:-build}" in
    build)
        shift
        prepare_cuda_artifact_dir
        (cd crates/inference && cargo oxide build "$@")
        cargo build --release -p nn-rust
        ;;
    run)
        shift
        prepare_cuda_artifact_dir
        (cd crates/inference && cargo oxide build)
        cargo build --release -p nn-rust
        ./target/release/nn-rust "$@"
        ;;
    clean)
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust.ptx
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust.ll
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust.ltoir
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust.cubin
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust_inference.ptx
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust_inference.ll
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust_inference.ltoir
        rm -f "$NN_RUST_CUDA_ARTIFACT_DIR"/nn_rust_inference.cubin
        rm -f nn_rust.ptx nn_rust.ll nn_rust.ltoir nn_rust.cubin
        rm -f nn_rust_inference.ptx nn_rust_inference.ll
        rm -f nn_rust_inference.ltoir nn_rust_inference.cubin
        rm -f crates/inference/nn_rust.ptx crates/inference/nn_rust.ll
        rm -f crates/inference/nn_rust.ltoir crates/inference/nn_rust.cubin
        rm -f crates/inference/nn_rust_inference.ptx crates/inference/nn_rust_inference.ll
        rm -f crates/inference/nn_rust_inference.ltoir crates/inference/nn_rust_inference.cubin
        cargo clean
        ;;
    *)
        echo "Usage: $0 [build|run|clean]"
        exit 1
        ;;
esac
