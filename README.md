# Multimodal Encrypted Vector Search

A research prototype for privacy-aware vector retrieval over medical images,
clinical text, and respiratory audio. The system combines **IVFFlat candidate
retrieval** with **CKKS encrypted reranking** to avoid an expensive homomorphic
scan over the entire database.

The project received a national third prize in the 10th National Cryptography
Technology Competition (China).

## System Overview

```text
Client
  ├─ embeds and encrypts the query with CKKS
  ├─ selects real and decoy IVF clusters
  v
Gateway ──> Index service ──> fixed-size candidate / diagonal-block plan
  |
  v
HECompute
  ├─ CKKS SIMD batched scoring
  ├─ diagonal matrix-vector multiplication
  └─ BSGS rotation optimization
  v
Client decrypts candidate scores and recovers Top-K
```

The original multimodal evaluation uses a unified 512-dimensional embedding
format for:

| Modality | Dataset | Retrieval task |
| --- | --- | --- |
| Image | ISIC 2020 | Similar skin-lesion images |
| Text | TREC-COVID | Related medical literature |
| Audio | ICBHI 2017 | Similar respiratory sounds |

## Main Features

- K-means-based IVFFlat indexing with configurable `nlist`, `nprobe`, and
  fixed per-cluster Top-R candidate retrieval.
- CKKS encryption for approximate real-valued inner-product scoring.
- SIMD and diagonal packing: one 4096-slot candidate pack outputs 4096 scores.
- Full group-level BSGS: for 512-dimensional scoring, online ciphertext
  rotations per pack decrease from 511 to 46; plaintext multiplications remain
  512.
- Numeric real/decoy cluster mixing, fixed response structure, HMAC, nonce, and
  timestamp checks for practical access-pattern and request-integrity
  mitigation.
- Reproducible evaluation scripts for exact search, candidate Recall@K, CKKS
  error, ranking consistency, latency, communication, and SIFT1M baselines.

## Optimized Implementation Map

The current SIMD, diagonal-packing, and BSGS implementation is located in:

| Optimization | Source file | Responsibility |
| --- | --- | --- |
| Production SIMD diagonal packing | [`backend/ckks/make_dia1_768.cpp`](backend/ckks/make_dia1_768.cpp) | Packs 4096 candidates across 512 encoded diagonals and offline pre-rotates BSGS plaintext groups |
| Full group-level BSGS kernel | [`backend/compute/hecompute_service.cpp`](backend/compute/hecompute_service.cpp) | Caches Baby rotations, performs group-local multiply/add, and applies one Giant rotation per group |
| Packing contract | [`backend/compute/hecompute_service.hpp`](backend/compute/hecompute_service.hpp) | Defines diagonal layouts, BSGS plans, and HE telemetry |
| Gateway BSGS plan | [`backend/gateway/backend/src/main.cpp`](backend/gateway/backend/src/main.cpp) | Generates the 32-by-16 Baby/Giant rotation plan for 512 dimensions |
| Layout propagation | [`adapters/index_http_adapter.py`](adapters/index_http_adapter.py) | Propagates the versioned `offset-major-bsgs-v1` layout to HECompute |
| SIMD/BSGS benchmark | [`backend/ckks/ckks_sift_rerank.cpp`](backend/ckks/ckks_sift_rerank.cpp) | Runs matched plaintext/CKKS correctness, rotation-count, and timing tests |

Legacy `offset-major` diagonal blocks remain supported. To activate full BSGS,
regenerate diagonal packs with `make_dia1_768 --full-bsgs --bsgs-baby 32`; the
embeddings and Faiss index do not need to be regenerated.

## Evaluation Summary

### Medical multimodal evaluation

On approximately 222,000 vectors across the three modalities, the project
reports:

- IVFFlat candidate-coverage Recall@16 above 98% for every modality, using
  plaintext exact Top-16 neighbors as the reference;
- millisecond-level plaintext candidate retrieval;
- CKKS-versus-plaintext score errors on the order of `1e-7` to `1e-6`, with
  stable Top-K ranking.

`Candidate Recall@16` is an ANN coverage metric, not a claim of clinician-rated
semantic relevance.

### Public SIFT1M evaluation

SIFT1M contains one million database vectors, 10,000 queries, and official
exact-neighbor ground truth. The privacy-compatible fixed-candidate path uses
`nlist=1024`, `nprobe=16`, and `R=1536`:

| Measurement | Result |
| --- | ---: |
| Fixed-candidate Recall@10, 10,000 queries | 90.790% |
| Candidates per query | 24,576 (6 packs) |
| Full-BSGS HE core, 10-query mean | 1.202 s/query |
| CKKS/plaintext Top-10 set and order consistency | 100% |
| Mean absolute score error | `2.06e-8` |
| Rotations / plaintext multiplications | 276 / 3,072 per query |

Against the same packed diagonal kernel before full group-level BSGS, the HE
core is approximately 5x faster. This speedup applies to the homomorphic core,
not to candidate loading, serialization, networking, or client-side work.

A one-query real-process validation through Client, Gateway, Index, and
HECompute used 16 real clusters plus 16 decoy clusters (49,152 slots):

| Measurement | Result |
| --- | ---: |
| Gateway / client wall time | 9.327 s / 9.399 s |
| HE telemetry | 3.161 s, 552 rotations, 6,144 plaintext multiplications |
| Online query payload | 18.38 MiB |
| CKKS/plaintext Top-10 set and order | Identical |

This single network query is a functional end-to-end validation, not a latency
mean or P95. See [the SIFT1M experiment audit](docs/sift1m_private_ann_comparison.md)
for definitions, matched A/B results, and comparison boundaries.

## Security Boundary

This prototype protects the query vector from direct observation by the
HECompute service and keeps decrypted scores and final ranking at the client.
It does **not** provide the same leakage profile as PIR/ORAM-based private ANN:

- Gateway and Index can observe the mixed numeric cluster-ID set;
- repeated access patterns may still leak statistical information;
- the client currently receives and decrypts all fixed candidate scores;
- a cryptographically secure server-side Top-K protocol is not implemented.

Real/decoy cluster mixing and fixed response sizes are engineering mitigations,
not formal full access-pattern hiding guarantees.

## Repository Structure

```text
adapters/                   HTTP-to-gRPC adapters
backend/ckks/               CKKS key, query, diagonal, decrypt, and benchmark tools
backend/compute/            HECompute core and gRPC service
backend/gateway/backend/    Gateway service
backend/services/index/     Candidate and diagonal-plan index service
scripts/                    Dataset preparation and evaluation scripts
docs/                       Competition material and experiment audit
```

## Requirements

- Linux or WSL2, CMake, and a C++17 compiler
- Microsoft SEAL 4.1.x
- Protobuf and gRPC C++ (`protoc` and `grpc_cpp_plugin`)
- SQLite3
- Python 3 with NumPy, Faiss, h5py, Flask, and gRPC packages

Python dependencies used by the WSL environment are listed in
[`requirements-wsl.txt`](requirements-wsl.txt).

## Build

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements-wsl.txt

cmake -S backend/compute -B backend/compute/build
cmake --build backend/compute/build -j

cmake -S backend/ckks -B backend/ckks/build
cmake --build backend/ckks/build -j

cmake -S backend/services/index -B backend/services/index/build
cmake --build backend/services/index/build -j

cmake -S backend/gateway/backend -B backend/gateway/backend/build
cmake --build backend/gateway/backend/build -j
```

If SEAL or gRPC is installed in a non-standard prefix, provide its CMake prefix
or the corresponding `SEAL_INCLUDE_DIR`, `SEAL_LIB_DIR`, and `PKG_CONFIG_PATH`.

## Reproduce the SIFT1M Core Evaluation

Place the ANN-Benchmarks SIFT1M HDF5 file at
`data/sift1m/sift-128-euclidean.hdf5`, then run:

```bash
python scripts/eval_sift1m_ivfflat.py \
  --data data/sift1m/sift-128-euclidean.hdf5 \
  --nlist 1024 --nprobe 16 \
  --cluster-top-r 1536 \
  --ckks-fixture results/sift1m/fixed-r1536.bin \
  --ckks-query-count 10 --ckks-nprobe 16 --ckks-fixed-top-r 1536

backend/ckks/build/ckks_sift_rerank \
  --fixture results/sift1m/fixed-r1536.bin \
  --queries 10 --full-dimension --full-bsgs \
  --details results/sift1m/ckks_details.csv \
  --summary results/sift1m/ckks_summary.json
```

For the real Gateway/Index/HECompute bundle, use
[`scripts/prepare_sift1m_network_e2e.py`](scripts/prepare_sift1m_network_e2e.py)
with `--full-bsgs`, upload evaluation keys with
[`backend/ckks/upload_evalkeys_v1.py`](backend/ckks/upload_evalkeys_v1.py), and
evaluate decrypted outputs using
[`scripts/eval_sift1m_network_e2e.py`](scripts/eval_sift1m_network_e2e.py).

## Data and Key Hygiene

Large datasets, generated embeddings, Faiss indexes, diagonal packs, model
weights, ciphertexts, and CKKS keys are intentionally excluded. Never commit:

```text
*.npy  *.f32  *.index  *.dia
*.bin  *.seal  sk.bin  pk.bin  galois.bin  relin.bin
```

Only generated evaluation keys should be uploaded to HECompute. Secret keys
must remain on the client.
