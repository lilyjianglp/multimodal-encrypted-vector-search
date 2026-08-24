# Multimodal Encrypted Vector Search

This repository contains a privacy-preserving multimodal vector search system for medical image, text, and audio embeddings.

The system combines IVFFlat candidate retrieval with CKKS encrypted reranking. It supports three modalities: ISIC 2020 skin lesion images, TREC-COVID medical text, and ICBHI 2017 respiratory audio.

## Main Features

- Multimodal vector retrieval for image, text, and audio
- Unified 512-dimensional embedding format
- IVFFlat candidate generation
- CKKS encrypted reranking
- SIMD packing for encrypted score computation
- BSGS diagonal matrix-vector multiplication
- Real/fake cluster mixing for access-pattern hiding
- Recall, ranking consistency, numerical error, and latency evaluation scripts

## Repository Structure

```text
scripts/
  Data preparation, embedding extraction, recall evaluation, and CKKS evaluation scripts.

backend/
  CKKS tools, HECompute service, Gateway service, and index service code.

adapters/
  HTTP adapter code between Gateway and HECompute.

docs/
  Project report, slides, and experiment notes.
## Modalities

| Modality | Dataset | Description |
|---|---|---|
| Image | ISIC 2020 | Skin lesion image retrieval |
| Text | TREC-COVID | Medical literature retrieval |
| Audio | ICBHI 2017 | Respiratory sound retrieval |

## Experimental Design

The evaluation follows four stages:

1. Exact baseline: use IndexFlatIP to perform full-corpus plaintext retrieval.
2. IVFFlat candidate recall: compare IVFFlat candidate pools with Exact Top-K results.
3. CKKS reranking correctness: compare plaintext scores with decrypted CKKS scores.
4. Final CKKS recall: compare CKKS reranked Top-K results with Exact Top-K results.

## Notes

Large datasets, generated embeddings, FAISS indexes, CKKS keys, ciphertexts, model weights, and runtime artifacts are intentionally excluded from this repository.

Do not upload files such as:

```text
*.npy
*.f32
*.index
*.dia
*.bin
*.seal
sk.bin
pk.bin
context.seal
galois.bin
relin.bin
Environment
The Python dependencies used in WSL are listed in:
requirements-wsl.txt
Some backend components require CMake, g++, FAISS, Microsoft SEAL, and gRPC.
