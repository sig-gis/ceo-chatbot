# ceo-build-index-gpu

GPU-accelerated variant of the `ceo-build-index` pipeline service.

Uses `faiss-gpu-cu12` and a CUDA-enabled torch wheel instead of their CPU counterparts.
The Python pipeline logic is identical to `ceo-build-index`; only the dependency closure
differs.

See the main [README.md](../../README.md) for build and run instructions, including how to
configure the torch wheel index for CUDA 12.x and older.
