# COCCL Pipeline Stage Dispatch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dispatch COCCL pipeline operations through private uniform stage handler functions instead of a switch in `cocclRunPipelineStage`.

**Architecture:** `coccl_pipeline.cc` owns six handlers and a static table indexed by `cocclPipelineStageKind`. The existing executor resolves slice storage and metadata before dispatch and performs final scatter afterward.

**Tech Stack:** C++17, CUDA Runtime API, NCCL private runtime APIs, Make.

## Global Constraints

- Do not change primitive flow declarations or `coccl_pipeline.h`.
- Do not change stage semantics, workspace layout, or CUDA ordering.
- Keep handler types and dispatch table private to `coccl_pipeline.cc`.

---

### Task 1: Extract and dispatch stage handlers

**Files:**
- Modify: `src/coccl-extend/primitives/coccl_pipeline.cc`
- Test: `tests/coccl-tests/src/check_pipeline_stage_dispatch.sh`

**Interfaces:**
- Consumes: existing `cocclPipelineContext`, `cocclPipelineStage`, and `cocclPipelineValue`.
- Produces: six private handlers and `cocclPipelineStageHandlers`.

- [ ] **Step 1: Add the failing source contract**

  Require six named handlers, a handler typedef/table, and no operation switch
  in `cocclRunPipelineStage`.

- [ ] **Step 2: Run the contract and verify it fails**

  Run: `bash tests/coccl-tests/src/check_pipeline_stage_dispatch.sh`

  Expected: failure because the handler table does not exist.

- [ ] **Step 3: Extract the six handlers**

  Move each existing switch case into a uniform private function without
  changing its validation, NCCL call arguments, or output metadata.

- [ ] **Step 4: Add the function pointer table**

  Validate `stage.kind`, index `cocclPipelineStageHandlers`, and invoke the
  selected handler from `cocclRunPipelineStage`.

- [ ] **Step 5: Verify contracts and build**

  Run the source contract, API compile contract, `make -n src.build`, real
  `make -j4 src.build`, and relink all six overlap test binaries.

Git metadata is unavailable in this workspace, so no commit step is included.
