/**
 * 3-Stream Matmul on Ascend NPU using ACLNN
 *
 * Stream topology (single matmul launch):
 *   input_stream    --[H2D A,B]--> record(input_done)
 *   compute_stream  wait(input_done) --[aclnnMatmul]--> record(compute_done)
 *   output_stream   wait(compute_done) --[D2H C]--> done
 *
 * Main thread synchronizes on output_stream to get the result.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_matmul.h"

#define DIM_M 256
#define DIM_K 256
#define DIM_N 256

static void check_acl(aclError err, const char *msg) {
    if (err != ACL_SUCCESS) {
        fprintf(stderr, "ACL error %d: %s\n", (int)err, msg);
        exit(1);
    }
}

static void check_nn(aclnnStatus s, const char *msg) {
    if (s != 0) {
        fprintf(stderr, "ACLNN error %d: %s\n", (int)s, msg);
        exit(1);
    }
}

int main() {
    // ---- 1. Init ACL + ACLNN ----
    check_acl(aclInit(nullptr), "aclInit");
    check_nn(aclnnInit(nullptr), "aclnnInit");

    check_acl(aclrtSetDevice(0), "aclrtSetDevice");
    aclrtContext ctx;
    check_acl(aclrtCreateContext(&ctx, 0), "aclrtCreateContext");
    check_acl(aclrtSetCurrentContext(ctx), "aclrtSetCurrentContext");

    // ---- 2. Prepare host data (FP32 for precision, FP16 on device) ----
    const int64_t shapeA[] = {DIM_M, DIM_K};
    const int64_t shapeB[] = {DIM_K, DIM_N};
    const int64_t shapeC[] = {DIM_M, DIM_N};
    size_t nA = DIM_M * DIM_K, nB = DIM_K * DIM_N, nC = DIM_M * DIM_N;
    size_t bytesA = nA * sizeof(aclFloat16);
    size_t bytesB = nB * sizeof(aclFloat16);
    size_t bytesC = nC * sizeof(aclFloat16);

    // Host pinned memory
    void *hA = nullptr, *hB = nullptr, *hC = nullptr;
    check_acl(aclrtMallocHost(&hA, bytesA), "mallocHost A");
    check_acl(aclrtMallocHost(&hB, bytesB), "mallocHost B");
    check_acl(aclrtMallocHost(&hC, bytesC), "mallocHost C");

    // Fill with FP16 value 0.5 (easier to verify: C = M*K*0.5*0.5 = 16384)
    aclFloat16 val = aclFloatToFloat16(0.5f);
    for (size_t i = 0; i < nA; i++) ((aclFloat16*)hA)[i] = val;
    for (size_t i = 0; i < nB; i++) ((aclFloat16*)hB)[i] = val;

    // Device memory
    void *dA = nullptr, *dB = nullptr, *dC = nullptr;
    check_acl(aclrtMalloc(&dA, bytesA, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dA");
    check_acl(aclrtMalloc(&dB, bytesB, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dB");
    check_acl(aclrtMalloc(&dC, bytesC, ACL_MEM_MALLOC_HUGE_FIRST), "malloc dC");

    // ---- 3. Create tensors for ACLNN ----
    // Row-major strides: for shape [M, K], stride = [K, 1]
    int64_t strideA[] = {DIM_K, 1};
    int64_t strideB[] = {DIM_N, 1};
    int64_t strideC[] = {DIM_N, 1};

    aclTensor *tensorA = aclCreateTensor(shapeA, 2, ACL_FLOAT16, strideA,
                                          0, ACL_FORMAT_ND, nullptr, 0, dA);
    aclTensor *tensorB = aclCreateTensor(shapeB, 2, ACL_FLOAT16, strideB,
                                          0, ACL_FORMAT_ND, nullptr, 0, dB);
    aclTensor *tensorC = aclCreateTensor(shapeC, 2, ACL_FLOAT16, strideC,
                                          0, ACL_FORMAT_ND, nullptr, 0, dC);

    // 2-phase ACLNN: get workspace size, then execute
    int8_t cubeMathType = 0;  // 0 = KEEP_DTYPE
    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnMatmulGetWorkspaceSize(tensorA, tensorB, tensorC,
                                                   cubeMathType, &wsSize, &executor);
    check_nn(ret, "aclnnMatmulGetWorkspaceSize");

    void *workspace = nullptr;
    if (wsSize > 0) {
        check_acl(aclrtMalloc(&workspace, wsSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace");
    }

    // ---- 4. Create 3 streams + 2 events ----
    aclrtStream input_stream, compute_stream, output_stream;
    check_acl(aclrtCreateStream(&input_stream),   "create input_stream");
    check_acl(aclrtCreateStream(&compute_stream), "create compute_stream");
    check_acl(aclrtCreateStream(&output_stream),  "create output_stream");

    aclrtEvent input_done, compute_done;
    check_acl(aclrtCreateEvent(&input_done),   "create input_done");
    check_acl(aclrtCreateEvent(&compute_done), "create compute_done");

    printf("=== 3-Stream Matmul (ACLNN) FP16 [%dx%d] * [%dx%d] = [%dx%d] ===\n",
           DIM_M, DIM_K, DIM_K, DIM_N, DIM_M, DIM_N);

    // ---- 5. Input stream: H2D copy, then signal ----
    printf("[input_stream]   H2D copy A, B ...\n");
    check_acl(aclrtMemcpyAsync(dA, bytesA, hA, bytesA,
                               ACL_MEMCPY_HOST_TO_DEVICE, input_stream), "H2D A");
    check_acl(aclrtMemcpyAsync(dB, bytesB, hB, bytesB,
                               ACL_MEMCPY_HOST_TO_DEVICE, input_stream), "H2D B");
    check_acl(aclrtRecordEvent(input_done, input_stream), "record input_done");

    // ---- 6. Compute stream: wait for input, launch matmul (once), signal ----
    printf("[compute_stream] wait for input ...\n");
    check_acl(aclrtStreamWaitEvent(compute_stream, input_done), "compute wait input");
    printf("[compute_stream] launch aclnnMatmul ...\n");
    check_nn(aclnnMatmul(workspace, wsSize, executor, compute_stream), "aclnnMatmul");
    check_acl(aclrtRecordEvent(compute_done, compute_stream), "record compute_done");

    // ---- 7. Output stream: wait for compute, D2H copy ----
    printf("[output_stream]  wait for compute ...\n");
    check_acl(aclrtStreamWaitEvent(output_stream, compute_done), "output wait compute");
    check_acl(aclrtMemcpyAsync(hC, bytesC, dC, bytesC,
                               ACL_MEMCPY_DEVICE_TO_HOST, output_stream), "D2H C");

    // ---- 8. Main thread: wait for output ----
    printf("[main]           waiting for output ...\n");
    check_acl(aclrtSynchronizeStream(output_stream), "sync output_stream");

    // ---- 9. Verify ----
    float expected = (float)DIM_K * 0.5f * 0.5f;
    float max_err = 0.0f;
    for (size_t i = 0; i < nC; i++) {
        float v = aclFloat16ToFloat(((aclFloat16*)hC)[i]);
        float err = fabsf(v - expected);
        if (err > max_err) max_err = err;
    }
    printf("\nExpected: %.4f   Max error: %.6f   %s\n",
           expected, max_err, max_err < 0.5f ? "PASS" : "FAIL");
    printf("C[0..4] = ");
    for (int i = 0; i < 5; i++)
        printf("%.4f ", aclFloat16ToFloat(((aclFloat16*)hC)[i]));
    printf("\n");

    // ---- 10. Cleanup ----
    aclDestroyTensor(tensorA);
    aclDestroyTensor(tensorB);
    aclDestroyTensor(tensorC);
    aclDestroyAclOpExecutor(executor);

    check_acl(aclrtDestroyEvent(input_done),   "destroy input_done");
    check_acl(aclrtDestroyEvent(compute_done), "destroy compute_done");
    check_acl(aclrtDestroyStream(input_stream),   "destroy input_stream");
    check_acl(aclrtDestroyStream(compute_stream), "destroy compute_stream");
    check_acl(aclrtDestroyStream(output_stream),  "destroy output_stream");

    if (workspace) aclrtFree(workspace);
    check_acl(aclrtFree(dA), "free dA");
    check_acl(aclrtFree(dB), "free dB");
    check_acl(aclrtFree(dC), "free dC");
    check_acl(aclrtFreeHost(hA), "free hA");
    check_acl(aclrtFreeHost(hB), "free hB");
    check_acl(aclrtFreeHost(hC), "free hC");

    check_acl(aclrtDestroyContext(ctx), "destroy context");
    check_acl(aclrtResetDevice(0), "reset device");
    check_nn(aclnnFinalize(), "aclnnFinalize");
    check_acl(aclFinalize(), "aclFinalize");

    printf("Done.\n");
    return 0;
}
