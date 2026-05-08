# Discussion: 3-Stream Matmul on Ascend NPU

## Goal

Write a matmul example on Ascend NPU with 3 streams:

1. **Input stream** — copy matrices A and B from host to device (H2D), then signal completion
2. **Compute stream** — wait for input ready, launch a single matmul kernel, then signal completion
3. **Output stream** — wait for compute done, copy result C from device to host (D2H)

The main program synchronizes on the output stream to retrieve the result.

Target machine: `ssh -p 60008 kkrazy@47.103.62.251` — 16x Ascend 910 NPUs, toolkit 8.1.RC2 (aarch64).

---

## Findings

### 1. aclblas API returns error 100000 (ACL_ERROR_RT_PARAM_INVALID)

The older CBLAS-style API (`aclblasGemmEx`, `aclblasHgemm`) consistently fails with error code 100000 regardless of data type combination:

| Attempt | API | Data Types | Result |
|---------|-----|-----------|--------|
| 1 | `aclblasGemmEx` | FP32 (ACL_FLOAT) | Error 100000 |
| 2 | `aclblasHgemm` | FP16 (aclFloat16) | Error 100000 |
| 3 | `aclblasGemmEx` | FP16 inputs, FP32 output | Error 100000 |

Matrix sizes tested: 16x16 and 256x256 (both multiples of 16, aligned for Cube unit). All other ACL operations (malloc, memcpy, stream/event creation) worked correctly, confirming the issue is specific to the aclblas library on this toolkit version.

**Hypothesis**: The `aclblas*` family may be deprecated or broken in toolkit 8.1.RC2. The library (`libacl_cblas.so`) exists and links fine but the runtime rejects the parameters.

### 2. ACLNN API (aclnnMatmul) works correctly

The newer ACLNN op API succeeded. It uses a 2-phase execution model:

- **Phase 1**: `aclnnMatmulGetWorkspaceSize(tensorA, tensorB, tensorC, cubeMathType, &wsSize, &executor)` — plans the operation and returns workspace requirements
- **Phase 2**: `aclnnMatmul(workspace, wsSize, executor, stream)` — executes on the given stream

Tensors are described via `aclCreateTensor()` with shape, stride, data type, and device pointer.

**Result**: 256x256 FP16 matmul produced correct output (expected=64.0, max_error=0.0, PASS).

### 3. Stream-Event Synchronization Pattern

The inter-stream dependency chain uses events:

```
input_stream:    H2D(A), H2D(B) → recordEvent(input_done)
compute_stream:  streamWaitEvent(input_done) → aclnnMatmul → recordEvent(compute_done)
output_stream:   streamWaitEvent(compute_done) → D2H(C)
main thread:     synchronizeStream(output_stream)
```

This ensures the matmul only starts after input data is on-device, and the D2H copy only starts after the matmul finishes — all without blocking the host.

### 4. Build & Link Notes

The ACLNN matmul requires linking against:

- `libascendcl.so` — core ACL runtime
- `libaclnn_ops_infer.so` — ACLNN op implementations
- `libnnopbase.so` — ACLNN base (provides `aclnnInit`/`aclnnFinalize`)
- `libacl_cblas.so` — linked but not actually used in the working version

Include paths need both `.../include` and `.../include/aclnnop`.

`LD_LIBRARY_PATH` must include `.../aarch64-linux/lib64` AND `.../latest/lib64` (for `libnnopbase.so`).

### 5. FP16 is the native compute type

Ascend 910 Cube engine natively performs FP16 matmul. FP32 matmul is not directly supported by the hardware accelerator. The `aclFloat16` type is a `uint16_t` typedef; conversion functions `aclFloatToFloat16()` / `aclFloat16ToFloat()` are provided by ACL.

---

## Open Questions

1. **aclblas deprecation status** — Is `aclblas*` officially deprecated in toolkit 8.1? Should all new code use ACLNN? The old API may still work on older toolkit versions.
2. **aclblas error root cause** — Is the 100000 error a known bug, a configuration issue (e.g., missing OPP kernel binaries), or a genuine API mismatch? Worth checking with `ASCEND_LOG_LEVEL=0` debug logs.
3. **FP32 matmul via ACLNN** — Does `aclnnMatmul` support FP32 inputs/output (with internal FP16 compute + accumulation)? The `cubeMathType` parameter (set to 0 = KEEP_DTYPE here) may control this.
4. **Performance benchmarking** — The 3-stream pattern is a pipeline skeleton. How does it compare to single-stream for larger matrices or batched matmul? What's the overlap between H2D of next batch / compute of current batch / D2H of previous batch?
5. **Multi-NPU** — The machine has 16 NPUs. Can we extend this to a distributed matmul using HCCL for all-reduce?

---

## Files

- `matmul_3stream.cpp` — Working 3-stream FP16 matmul using ACLNN API
