#!/bin/bash
# Build and run nn-rust with cuda-oxide
# Usage: ./run.sh [build|run|clean]

set -e

export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}
export CUDA_TOOLKIT_PATH=/opt/cuda
export BINDGEN_EXTRA_CLANG_ARGS="-I/opt/cuda/include"
export CUDA_OXIDE_BACKEND=${CUDA_OXIDE_BACKEND:-~/.cargo/cuda-oxide/librustc_codegen_cuda.so}

cd "$(dirname "$0")"
export CUDA_OXIDE_PTX_DIR="$PWD/crates/inference"

case "${1:-build}" in
    build)
        shift
        (cd crates/inference && cargo oxide build "$@")
        cargo build --release -p nn-rust
        ;;
    run)
        shift
        (cd crates/inference && cargo oxide build)
        cargo build --release -p nn-rust
        ./target/release/nn-rust "$@"
        ;;
    clean)
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
