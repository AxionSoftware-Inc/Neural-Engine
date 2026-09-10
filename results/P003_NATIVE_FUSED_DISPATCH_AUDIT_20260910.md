# P-003 native factorized fused-dispatch audit

Date: 2026-09-10  
Branch: `exp/track-runtime`  
Device: NVIDIA GeForce RTX 3060, CUDA 12.4

## Question

Can the current factorized-additive native circuit bank be executed by one
custom CUDA dispatch per selected `(token, circuit)` pair, preserving the
route output while removing the PyTorch/einsum dispatch overhead?

## Scope and parity contract

The kernel implements the exact current ordered factorized-additive formula:

- two factor rows are mixed with the learned per-address coefficients;
- GELU is applied to the mixed down projection;
- the mixed up projection and bias are accumulated with route weights;
- the result is written to the same `[tokens, state_dim]` correction output.

It is inference-only and currently supports the 500M configuration’s ordered
two-slot factor bank with no pair/product/hidden-gate/address-residual terms.
Unsupported configurations remain on the PyTorch implementation.

## Three-seed runtime result

Each timing used five warm-ups and 20 synchronized CUDA repeats. `native_fused`
uses the custom kernel; the corresponding `torch` variant is the existing
PyTorch/einsum path. Parity is measured against the corresponding PyTorch
output on the same batch.

### Balanced batch 480

| Variant | PyTorch | Native fused | Speed change | Max error |
|---|---:|---:|---:|---:|
| Fixed K=8 | 37.20 ms | 22.03 ms | **−40.8%** | 5.72e-6 |
| Fixed K=16 | 58.38 ms | 35.32 ms | **−39.5%** | 5.72e-6 |
| Learned K=8/16 | 41.32 ms | 30.90 ms | **−25.2%** | 7.63e-6 |

### Balanced batch 960

| Variant | PyTorch | Native fused | Speed change | Max error |
|---|---:|---:|---:|---:|
| Fixed K=8 | 65.85 ms | 34.78 ms | **−47.2%** | 5.72e-6 |
| Fixed K=16 | 105.70 ms | 57.49 ms | **−45.6%** | 7.63e-6 |
| Learned K=8/16 | 68.32 ms | 50.28 ms | **−26.4%** | 7.63e-6 |

The learned selector keeps the same active widths and route choices; the
kernel changes execution only. The larger-batch result confirms that the
gain is not limited to launch noise at one shape.

## Serving batch-shape sweep

The same three checkpoints were measured at balanced batch sizes 15, 120, 240,
480, and 960 (five warm-ups and 20 synchronized CUDA repeats). The table is
the three-seed mean; the percentage is the reduction relative to the matching
PyTorch path.

| Batch | Fixed K=16 torch → fused | Speed change | Learned torch → fused | Speed change |
|---:|---:|---:|---:|---:|
| 15 | 7.52 → 7.25 ms | −3.5% | 8.51 → 6.59 ms | −22.5% |
| 120 | 23.17 → 11.35 ms | **−51.0%** | 19.60 → 15.40 ms | −21.4% |
| 240 | 38.75 → 17.91 ms | **−53.8%** | 28.50 → 18.91 ms | −33.7% |
| 480 | 65.48 → 29.76 ms | **−54.6%** | 44.57 → 29.21 ms | −34.5% |
| 960 | 116.43 → 45.31 ms | **−61.1%** | 67.88 → 41.56 ms | −38.8% |

All fused cases stayed within `5.72e-6` maximum logit error of their PyTorch
reference. At batch 15 the learned-width guard correctly selected full K=16;
at batches 120–960 its mean active width was approximately 8.4–8.9. Thus the
kernel is useful for both fixed-width throughput and learned-width serving,
but the small-batch result is too close to launch noise to justify a universal
automatic dispatch policy.

## Unsupported-feature fallback validation

The bank now exposes one explicit eligibility predicate for the native kernel.
Nine CUDA tests verified that unsupported configurations do not call the
native extension and instead complete through the existing PyTorch path:
unordered slots, shared factor mix, query-conditioned mix, pair interaction,
factor product, hidden product, hidden gate, serial composition, and address
residual. This is a safety check, not an implementation of those features in
the fused kernel.

## Sequence-shape sweep

At balanced batch 120, three seeds were also tested with the same padded
examples truncated to sequence lengths 6, 8, 16, and 32. The three-seed mean
speed reductions were:

| Sequence length | Fixed K=16 | Learned K=8/16 |
---:|---:|---:|
| 6 | **40.7%** | **29.5%** |
| 8 | **48.9%** | **29.6%** |
| 16 | **50.2%** | **30.7%** |
| 32 | **50.9%** | **27.9%** |

Every fused sequence case remained within `5.72e-6` maximum logit error of
the matching PyTorch path. The model's configured `slot_count=5` was respected;
length 6 is the shortest valid serving shape for this checkpoint. This closes
the current batch/sequence parity sweep for the tested 500M configuration, but
it is not yet a production integration test with request-shape caching or
concurrent requests.

## Serving reuse and stream-safety smoke

For each of the three seeds, B=120 requests at sequence lengths 6 and 32 were
run repeatedly in alternating shape order and then concurrently on two CUDA
streams. Both fixed K=16 and learned K=8/16 fused paths stayed within
`5.72e-6` maximum logit error of their torch references in both tests. No
shape-switch or cross-stream output contamination was observed. This validates
the kernel's current read-only serving contract, but it does not yet provide a
request pool, shape-cache eviction policy, or a production server integration.

## Shape-cache serving caller

The opt-in `NativeFusedShapeCache` caller now provides a bounded LRU cache of
fixed-shape CUDA Graph entries. Keys include batch size, sequence length, and
CUDA stream, so separate streams do not share mutable graph input buffers.
Dynamic-width requests, non-CUDA inputs, training mode, and graph-capture
failures use the eager model and increment explicit fallback counters.

On the real seed-17 500M checkpoint, B=1 with sequence lengths 6 and 32
captured two shapes, produced 52 cache hits, and had zero eager fallbacks.
Cached latency was `2.10/2.17 ms` versus eager `9.07/6.07 ms`; maximum graph
versus eager logit error was `1.91e-6`. B=8 captured the same two shapes with
zero fallbacks; cached latency was `2.42/2.55 ms` versus eager `5.98/5.59 ms`,
with maximum error `1.91e-6`. Unit tests also cover LRU eviction, dynamic-width
eager fallback, and synthetic graph-capture failure fallback.

This is a reusable serving caller and a real-checkpoint smoke, not a complete
multi-worker server. Request admission, cross-process ownership, and policy
for concurrent requests on the same CUDA stream remain caller responsibilities.

The same-stream policy was then exercised on the real seed-17 500M checkpoint:
four Python workers issued 16 B=1, seq=32 requests through one cached entry.
The run produced one graph capture and 16 cache hits; every output matched the
eager reference with maximum error `1.91e-6`. The per-entry replay lock makes
same-shape calls host-serialized, while different streams retain independent
entries. The cache now adds a stream-level lock around capture and replay, so
different shapes on the same CUDA stream are serialized too. A cross-process
reuse unit test rejects inherited cache ownership, so each worker must
construct its own model and cache after process creation.

## HTTP server entry-point smoke

`serve_native.py` and `NativeFusedService` now provide a small threaded HTTP
entry point around the same process-local model/cache contract. `GET /health`
reports device/backend/cache ownership, `GET /stats` reports cache counters, and
`POST /infer` accepts a rectangular JSON `inputs` list and returns hard
predictions (optionally logits). Input shape, sequence limit, token range, and
process ownership are validated before dispatch. The service has no global
request lock: same-stream graph capture/replay is protected by the cache
stream lock, while independent CUDA stream entries can proceed independently.

The real seed-17 500M checkpoint was served in-process through the actual HTTP
handler at B=1/B=8 and sequence lengths 6/32. All four shapes captured once and
then hit the cache once (`4 capture / 4 hit / 0 eager fallback`); predictions
matched on repeat requests and maximum logit error versus eager was `1.91e-6`.
The first HTTP calls took approximately `91.6–124.7 ms` (including graph
capture and JSON transport), and repeated calls took `9.3–24.9 ms`. This closes
the local server-entry smoke. It is still not a production deployment: TLS,
authentication, batching/admission control, process supervision, and a tested
multi-process launcher remain outside this repository entry point.

The HTTP concurrency benchmark then sent 16 requests through four client
workers, alternating B=1/B=8 and sequence lengths 6/32. Every response matched
the eager reference; the maximum logit error was `1.91e-6` and there were zero
prediction mismatches or eager fallbacks. The four unique shapes produced four
captures and 12 cache hits in the shared process; the measured maximum logit
error was `3.81e-6`. This validates the threaded HTTP path locally, but it is
not a substitute for a deployment-specific multi-process launcher or
admission/batching stress test.

`serve_native_workers.py` now supplies that process-lifecycle primitive using
the Windows-safe `spawn` start method. Each child loads the checkpoint and
constructs its own model/cache, then listens on a consecutive port. The parent
terminates the group if a worker fails; no CUDA object is inherited from the
parent. This is a local launcher, not a complete deployment: an external
load-balancer, health-aware admission policy, TLS/authentication, and memory
capacity planning are still required.

Each child also watches the multiprocessing parent and shuts its HTTP server
down if that parent disappears unexpectedly. This prevents orphan workers and
stale CUDA/extension handles when a Windows benchmark or controller is killed.

When multiple workers share one physical CUDA device, the launcher enables the
cache's crash-releasing device graph lock automatically. It serializes both
capture and replay across those processes because the RTX 3060 rejected
overlapping graph operations; single-worker and separate-device deployments do
not need this cross-process serialization.

The launcher was integration-tested with a small CPU checkpoint and two real
child processes: both ports answered `/health` and `/infer`, the PIDs were
distinct, and each reported a cache owner equal to its own PID. This confirms
the ownership boundary without duplicating the 500M CUDA checkpoint merely for
a lifecycle test.

The round-robin client benchmark then sent 16 parallel requests (B=1/B=8,
sequence 6/32) across those two workers. All four shapes were exercised by
both workers, cross-worker logit error was `0`, and prediction mismatch was
`0`. Because this controlled run used CPU with graphs disabled, each worker
reported eight expected eager fallbacks; it validates routing and ownership,
not CUDA-Graph throughput.

The same benchmark was then run on the real seed-17 500M checkpoint with two
workers on the RTX 3060. Sixteen parallel requests produced four captures and
four cache hits per worker, with zero eager fallbacks and zero capture failures.
Cross-worker maximum logit error was `2.86e-6` and prediction mismatch was `0`.
An earlier run without the device lock reproduced a CUDA
`operation failed due to a previous error during capture` at the second replay
round; the final run confirms the lock fixes that race. The trade-off is that
two workers on one GPU are safe but graph operations are serialized across the
device, so throughput scaling should use one worker per GPU or a future tested
batching policy.

### Shared-GPU worker scaling

To quantify that trade-off, the same seed-17 500M checkpoint was tested with
the same four concurrent HTTP clients, 16 requests, and the same B=1/B=8,
sequence 6/32 shape cycle. With one worker, the request wall time was
`815.3 ms`, mean client latency `202.1 ms`, and p95 latency `784.4 ms`. With
two workers on the same RTX 3060, the wall time was `3245.0 ms`, mean latency
`810.0 ms`, and p95 latency `1612.6 ms`. Thus the two-worker configuration
was about `3.98x` slower for this small local workload; it did not provide
throughput scaling because the correctness lock serializes graph capture/replay
on the shared device and two processes add scheduling/IPC overhead.

This is not a quality regression: both configurations had zero prediction
mismatches, all four shapes captured successfully, and the maximum cross-worker
logit difference in the two-worker run was `2.86e-6`. The deployment policy is
therefore one worker per GPU unless a tested admission/batching layer can
combine requests before graph replay. The result is a small-batch serving
measurement, not a claim about all batch sizes or multiple physical GPUs.

### Shape-homogeneous micro-batching

An offline grouped-request control used the same 32 logical seed-23004
requests, four client workers, and sequence length 32. Grouping requests into
fixed shapes preserved predictions exactly and produced no mismatches. Logical
throughput was `385.2 req/s` at B=1, `630.7 req/s` at B=2, `1130.3 req/s` at
B=4, and `3083.6 req/s` at B=8. The corresponding wall times were `83.1`,
`50.7`, `28.3`, and `10.4 ms`; maximum logit difference from the B=1 fused
reference stayed between `4.29e-6` and `5.72e-6`.

This is a positive kernel/HTTP-shape signal, but it is not by itself an
admission result: the client explicitly formed each grouped request.

The real opt-in admission queue was then tested with the same individual
requests, four client workers, `max_batch_size=8`, and a 2 ms shape window.
Even after prewarming B=1…8 graph shapes, direct serving took `94.3 ms`
(`339.3 req/s`) while the queue took `161.5 ms` (`198.2 req/s`). It did reach
an observed batch of 8 and preserved output parity (`0` prediction mismatches,
maximum logit error `5.72e-6`), but its queue/response overhead outweighed the
compute saving at this local arrival rate. A cold run was worse because
variable arrival groups caused extra graph captures. The queue therefore
remains opt-in and is not accepted as a default serving policy; a future
version needs request-rate-aware admission, shape prewarming, and a latency
budget before claiming benefit.

A shorter 0.25 ms window was also checked with the same prewarmed B=1…8
shapes and four clients. Direct serving measured `75.4 ms` (`424.2 req/s`),
while admission measured `155.7 ms` (`205.5 req/s`); it still reached an
observed batch of 8 and retained `0` prediction mismatches. Reducing the wait
window therefore did not change the decision: the current per-request HTTP
queue is not a speed path for this workload.

Finally, a mixed-shape upstream control alternated sequence lengths 6 and 32
across 32 logical requests, then bucketed each sequence length before grouping.
At B=1 it reached `368.8 req/s`; B=4 reached `1478.0 req/s` (`4.0x`), and
B=8 reached `3769.2 req/s` (`10.2x`). Both sequence buckets used their own
cached graph shape, with `0` prediction mismatches and maximum logit difference
`5.72e-6`. This confirms that the usable runtime path is an upstream
shape-bucketing/grouped `/infer` API; the internal per-request admission queue
should remain opt-in and is not part of the default.

The explicit `/infer_batch` endpoint was then validated on the same 32 mixed
seq=6/32 logical requests. Direct serving used 32 HTTP calls and reached
`326.4 req/s`; one grouped endpoint call used four B=8 shape groups and reached
`1227.4 req/s` (`3.76x`). The endpoint kept both sequence buckets separate,
used two cached graph shapes, produced `0` prediction mismatches, and stayed
within `5.72e-6` maximum logit difference from direct serving. This makes the
upstream grouped endpoint the preferred runtime integration point; the
per-request admission queue remains an opt-in fallback experiment.

## Long quality control

The fused learned checkpoints were rerun through the 96-batch-per-condition
OOD audit and compared with the existing PyTorch learned control. Exact,
hard-task, active-width, and wide-width metrics were identical for all three
seeds and four conditions at reported precision. The maximum absolute CE
difference was `1.12e-8`. This confirms that the small floating-point ordering
difference from atomic accumulation did not change the recurrent route or the
reported quality metrics in this audit.

## Independent long quality control

To separate a kernel regression from checkpoint variance, the three learned
checkpoints were evaluated for 48 balanced batches per condition with both
`native_cuda_fused` and `torch` backends. This is 1,536 examples per task per
condition. Across all three seeds and all four conditions, exact-accuracy
difference was `0`; the largest CE difference was `1.61e-8`.

The fused three-seed means were uniform exact `77.56%`, combination-heldout
`77.76%`, low-edge `95.76%`, and high-edge `95.20%`. Seed 19 itself was much
weaker on the uniform/deep composition conditions than seeds 17/18, but the
same weakness appeared in its torch control exactly. It is therefore a
training/checkpoint-seed stability issue, not a native-kernel quality issue.

## CUDA Graph check

Fixed K=16 with the native kernel captured successfully at batch 1 and 32.
Graph/eager ratios were `0.968x` and `0.837x`, with max errors
`1.91e-6` and `3.81e-6`. Learned dynamic routing remains graph-unsafe because
its route partition is variable; this kernel does not hide that separate issue.

## Decision

`PROMISING OPT-IN — HTTP SERVER SMOKE VALIDATED; PRODUCTION HARDENING OPEN`.

Keep native fused dispatch opt-in and leave PyTorch as the default. The
shape-cache caller, batch/sequence parity, fallback, stream-safety, same-stream
concurrency, HTTP entry-point smoke, and long quality checks now pass. Keep the
native path opt-in until a deployment-specific launcher gives every worker its
own model/cache and request admission policy. The kernel must never silently
approximate a configuration it does not support.

## Raw evidence and reproduction

- [480-batch runtime JSON](diagnostic_native_fused_runtime_all3_480_20260910.json)
- [960-batch runtime JSON](diagnostic_native_fused_runtime_all3_960_20260910.json)
- [Batch-shape sweep JSON](diagnostic_native_fused_shape_sweep_all3_20260910.json)
- [Sequence-shape sweep JSON](diagnostic_native_fused_sequence_sweep_all3_20260910.json)
- [Serving reuse/stream smoke JSON](diagnostic_native_fused_serving_smoke_all3_20260910.json)
- [Shape-cache B=1 JSON](diagnostic_native_fused_shape_cache_s17_b1_20260910.json)
- [Shape-cache B=8 JSON](diagnostic_native_fused_shape_cache_s17_b8_20260910.json)
- [Same-stream concurrency JSON](diagnostic_native_fused_serving_concurrency_s17_b1_20260910.json)
- [HTTP server smoke JSON](diagnostic_native_fused_http_server_s17_20260910.json)
- [HTTP server concurrency JSON](diagnostic_native_fused_http_server_concurrency_s17_20260910.json)
- [Native fused CUDA multi-worker JSON](diagnostic_native_fused_multi_worker_cuda_s17_20260910.json)
- [Native fused one-vs-two-worker scaling JSON](diagnostic_native_fused_worker_scaling_s17_20260910.json)
- [Native fused grouped microbatch JSON](diagnostic_native_fused_microbatch_s17_20260910_32req.json)
- [Native fused admission JSON](diagnostic_native_fused_admission_s17_20260910_prewarmed.json)
- [Native fused admission, 8-client stress JSON](diagnostic_native_fused_admission_s17_20260910_8clients.json)
- [Native fused admission, 0.25 ms window JSON](diagnostic_native_fused_admission_s17_20260910_window025_prewarmed.json)
- [Native fused mixed-shape microbatch JSON](diagnostic_native_fused_mixed_microbatch_s17_20260910.json)
- [Native fused explicit batch endpoint JSON](diagnostic_native_fused_batch_endpoint_s17_20260910.json)
- [Seed17 fused runtime smoke JSON](diagnostic_native_fused_runtime_s17_480_20260910.json)
- [Long fused learned-width OOD JSON](diagnostic_native_fused_learned_ood_long96_20260910.json)
- [Independent long fused OOD JSON](diagnostic_native_fused_ood_long48_all3_20260910.json)
- [Matching long torch OOD control JSON](diagnostic_native_torch_ood_long48_all3_20260910.json)
- [Fused fixed K=16 Graph batch-1 JSON](diagnostic_native_cuda_graph_fused_fixed16_b1_20260910.json)
- [Fused fixed K=16 Graph batch-32 JSON](diagnostic_native_cuda_graph_fused_fixed16_b32_20260910.json)
- [Runtime benchmark](../benchmark_native_width_runtime.py)
- [CUDA Graph benchmark](../benchmark_native_cuda_graph.py)
- [Shape-cache serving caller](../neural_engine/native_fused_serving.py)
- [Shape-cache benchmark](../benchmark_native_fused_shape_cache.py)
- [Concurrency benchmark](../benchmark_native_fused_serving_concurrency.py)
- [HTTP server benchmark](../benchmark_native_server.py)
- [HTTP server concurrency benchmark](../benchmark_native_server_concurrency.py)
- [HTTP serving entry point](../serve_native.py)
- [Multi-process serving launcher](../serve_native_workers.py)
- [Multi-worker round-robin benchmark](../benchmark_native_workers.py)
- [Shared-GPU worker-scaling benchmark](../benchmark_native_worker_scaling.py)
- [Grouped microbatch benchmark](../benchmark_native_microbatch.py)
- [Admission-queue benchmark](../benchmark_native_admission.py)
- [Mixed-shape microbatch benchmark](../benchmark_native_mixed_microbatch.py)
- [Explicit batch endpoint benchmark](../benchmark_native_batch_endpoint.py)
- [HTTP service adapter](../neural_engine/native_fused_server.py)
- [Python wrapper](../neural_engine/native_fused_dispatch.py)
- [CUDA kernel](../neural_engine/native_fused_dispatch.cu)
- [Batch-shape sweep](../benchmark_native_fused_shape_sweep.py)
- [Sequence-shape sweep](../benchmark_native_fused_sequence_sweep.py)
- [Serving smoke](../benchmark_native_fused_serving_smoke.py)

```powershell
python benchmark_native_width_runtime.py `
  --checkpoints results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s18_3000.pt `
    results/checkpoints/ne500_stable_prefix_active16_s19_3000.pt `
  --examples-per-task 64 --warmup 5 --repeats 20 --include-native-fused `
  --output results/diagnostic_native_fused_runtime_all3_960_20260910.json
```

HTTP server-entry smoke:

```powershell
python benchmark_native_server.py `
  --checkpoint results/checkpoints/ne500_stable_prefix_active16_s17_3000.pt `
  --batch-sizes 1 8 --sequence-lengths 6 32 --warmup-iters 3 `
  --output results/diagnostic_native_fused_http_server_s17_20260910.json
```

For interactive local serving, use the same checkpoint with
`python serve_native.py --checkpoint <path> --port 8080`. The JSON API is
`POST /infer` with `{"inputs": [[token, ...], ...]}` and optional
`"return_logits": true`; each process must load its own model and cache.
