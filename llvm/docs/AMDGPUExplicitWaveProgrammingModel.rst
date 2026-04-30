=======================================
AMDGPU Explicit Wave Programming Model
=======================================

.. contents::
   :local:

Status
======

This document is a design proposal for a programming model that exposes the
AMDGPU wavefront as the primary unit of execution. It is not a description of
an existing source language or ABI.

The proposed compiler architecture uses an MLIR dialect as the communication
layer between high-level languages and the LLVM AMDGPU backend. Source
languages lower their wave semantics into this dialect; the dialect verifies and
canonicalizes those semantics; LLVM lowering then maps them to AMDGPU intrinsics,
metadata, and backend-recognized IR.

The goal is to replace the user-visible fiction of many independent virtual
threads with a model in which the program operates on a whole wavefront: a
32-lane or 64-lane SIMD vector with an explicit active-lane mask.

Motivation
==========

AMDGPU hardware executes wavefronts. A wavefront has a single scalar control
path, a set of scalar registers, a set of vector registers with one element per
lane, and an ``EXEC`` mask that controls which lanes participate in vector
operations. The conventional SIMT model hides this machine behind work-items
that appear to execute independently. This abstraction is useful for portability,
but it obscures the actual cost model:

* A VGPR value is one value per lane, not one scalar value owned by an
  independent hardware thread.
* A divergent condition is a lane mask.
* Divergent control flow is implemented by manipulating ``EXEC`` and executing
  both sides under different masks.
* Uniform values are materially different from lane-varying values because they
  live in SGPRs and execute on the scalar ALU.
* Moving a lane-varying value into scalar code is not a normal copy. It requires
  an explicit operation such as ``readfirstlane``, a reduction, a proof of
  uniformity, or a waterfall loop.

The proposed model makes these facts visible and type checked.

Existing Backend Model
======================

The LLVM AMDGPU backend already contains most of the target-level machinery
needed for an explicit wave model.

Register banks
--------------

The backend distinguishes scalar and vector register banks. SGPR values are
uniform for the wave. VGPR and AGPR values are lane-varying. ``VCC`` is used for
vector compare masks.

``llvm/lib/Target/AMDGPU/AMDGPURegisterBankInfo.cpp`` describes the central
constraint: copying from VGPR to SGPR is generally illegal unless the value is
known uniform. SGPR-required operations with divergent inputs are handled by
waterfall loops, which repeatedly execute scalar operations for groups of lanes
with the same input value.

Lane masks
----------

The backend represents divergent ``i1`` values as wave masks in scalar
registers. ``llvm/lib/Target/AMDGPU/SILowerI1Copies.cpp`` lowers ``VReg_1``
values into 32-bit or 64-bit scalar lane masks after machine control flow is in
wave form.

The register classes ``SReg_1`` and ``SReg_1_XEXEC`` in
``llvm/lib/Target/AMDGPU/SIRegisterInfo.td`` select a 32-bit or 64-bit scalar
mask class according to wave size. On wave32 targets the active mask is
``EXEC_LO`` and mask operations use 32-bit scalar instructions. On wave64
targets the active mask is ``EXEC`` and mask operations use 64-bit scalar
instructions.

Control flow
------------

``llvm/lib/Target/AMDGPU/SILowerControlFlow.cpp`` lowers pseudo control-flow
operations such as ``SI_IF``, ``SI_ELSE``, and ``SI_END_CF`` into scalar
instructions that save, update, and restore ``EXEC``. The resulting program is
not a collection of independent threads. It is a scalar program that updates an
active-lane mask around vector instructions.

Wave size
---------

Wave size is a subtarget property. GFX9 and older GCN targets are wave64.
GFX10 through GFX12 can support wave32 and wave64 depending on target features.
Newer targets may restrict this further. Backend predicates such as
``FeatureWavefrontSize32``, ``FeatureWavefrontSize64``, ``isWave32()``, and
``isWave64()`` select the correct register classes, instruction forms, and
special registers.

Design Overview
===============

The source programming model should expose wave values and masks explicitly,
while treating ordinary scalar values as uniform by default.

``T``
  A single value shared by all lanes in the wave. It maps naturally to SGPRs and
  SALU instructions. A source language may provide an optional spelling such as
  ``uniform<T>`` for documentation or generic programming, but it should not be
  required: a value is uniform unless it is explicitly a ``wave`` value or a
  ``mask``.

``wave<T, W>``
  A lane-varying value with one ``T`` element per lane of a wave of width
  ``W``. It maps naturally to VGPRs or AGPRs and VALU instructions. This is not
  an LLVM fixed vector type. It is a semantic wave value.

``mask<W>``
  A 32-bit or 64-bit lane mask. It maps naturally to ``EXEC``-width scalar mask
  registers and to ``VCC`` results from vector compares.

``W`` is either 32, 64, or a target-selected wave size known at code generation
time. Source languages may provide wave-size-polymorphic abstractions, but LLVM
lowering should know the chosen width before instruction selection.

These concepts should be represented first in MLIR. The MLIR layer is the
stable contract between source languages and target lowering: it carries wave
types, mask regions, uniformity information, resource annotations, and
generation-specific constraints before those concepts are lowered into LLVM IR.
Unlike the source language, the MLIR dialect may choose to make uniformity
explicit in types or attributes so the property survives canonicalization and
lowering.

MLIR Communication Layer
========================

The explicit wave model should be defined by an MLIR dialect, either as a new
wave-specific AMDGPU dialect layer or as a structured extension of the existing
MLIR GPU and AMDGPU dialects. This dialect is not merely an implementation
detail. It is the interface that high-level languages use to communicate
hardware-level wave intent to the backend.

Role
----

The dialect should:

* preserve the distinction between uniform values, lane-varying values, and lane
  masks;
* represent structured active-mask regions before they become ``EXEC``
  manipulation;
* carry wave-size and occupancy attributes;
* expose cross-lane communication, reductions, and wave-cooperative memory as
  target-level operations;
* verify AMDGPU-specific legality before the program is lowered to LLVM IR.

Types and attributes
--------------------

The dialect should provide types or type attributes for:

``uniform<T>`` or an equivalent uniformity attribute
  A value that is the same for all lanes in the current wave.

``wave<T, W>``
  A lane-varying value with one element per lane.

``mask<W>``
  A wave mask with one bit per lane.

Kernel and function attributes should describe requested wave size, work-group
shape, minimum waves per execution unit, SGPR/VGPR/AGPR budgets, LDS usage, and
target feature requirements.

Operations
----------

The dialect should contain operations for:

* lane identity and wave identity;
* a custom ``where`` operation with an optional ``otherwise`` region for
  structured lane-mask control;
* standard MLIR ``scf.for``, ``scf.if``, and ``scf.while`` operations for
  structured uniform control;
* mask algebra such as ``and``, ``or``, ``not``, ``ballot``, ``any``, ``all``,
  and ``popcount``;
* explicit crossings between uniform and lane-varying values, including
  ``broadcast``, ``read_first``, ``read_lane``, reductions, and
  ``assert_uniform``;
* lane permutations and DPP/permlane-style communication;
* masked global, LDS, and scratch memory operations;
* wave-cooperative matrix and packed-data operations that map to AMDGPU MFMA,
  WMMA, or related instructions.

Lowering contract
-----------------

The dialect should lower in stages:

1. High-level source dialects lower work-item or domain-specific constructs to
   explicit wave operations, custom ``where`` regions, and standard ``scf``
   structured control.
2. Wave canonicalization consumes ``where`` and ``scf.for``/``scf.if``/
   ``scf.while`` together, simplifies mask regions, folds uniform broadcasts,
   specializes wave-size-polymorphic code, and selects legal operation forms.
3. Conversion to LLVM IR lowers wave values to divergent scalar IR values,
   masks to ``i32`` or ``i64``, and structured regions to AMDGPU control-flow
   intrinsics.
4. The LLVM AMDGPU backend performs instruction selection, register bank
   selection, ``EXEC`` lowering, scheduling, and generation-specific codegen.

This boundary keeps source-language design independent from backend mechanics
while still preserving the facts the backend needs most: uniformity, lane masks,
wave size, convergence, and resource intent.

Relationship to Other GPU ISAs
==============================

The first target for this design is AMDGPU, but the MLIR layer should avoid
encoding AMD-only concepts in its portable core. The useful common abstraction
is a subgroup-sized SIMD program with:

* scalar-uniform values;
* lane-varying values;
* logical lane masks;
* structured predication through ``where``;
* uniform structured control through ``scf.for``, ``scf.if``, and
  ``scf.while``;
* explicit cross-lane operations, reductions, scans, and masked memory.

AMDGPU should then be one target-specific lowering of that model, where logical
lane masks become ``EXEC``/``VCC`` masks and scalar-uniform values are SGPR/SALU
candidates.

NVIDIA
------

NVIDIA hardware exposes a warp-oriented execution model with a practical warp
width of 32 lanes. The portable pieces of this proposal map naturally to NVIDIA
warp concepts:

* ``wave<T, W>`` or a portable ``subgroup<T, W>`` maps to one value per warp
  lane;
* ``mask<W>`` maps to a logical warp mask, ballot result, active mask, or
  collective member mask;
* ``where(mask)`` maps to logical predication, predicated instructions, or
  structured SIMT branches;
* ``scf`` remains uniform control when its conditions and loop bounds are scalar
  values;
* operations such as ballot, shuffle, broadcast, reductions, and scans map to
  PTX/NVVM warp-level operations.

The important difference is that NVIDIA does not expose an AMD-style writable
``EXEC`` register through the compiler model. PTX has per-thread registers and
predicate registers, and modern warp collectives use explicit member masks. A
portable ``where`` therefore cannot mean "install this mask as the hardware
execution mask" in the generic dialect. It must mean "execute this structured
region under this logical lane predicate".

NVIDIA-specific lowering must also preserve convergence and member-mask rules
for warp collectives, barriers, calls, and synchronization. Shared memory,
asynchronous copies, tensor-core operations, and occupancy controls should be
modeled through target-specific operations or attributes rather than AMDGPU
LDS/MFMA resource names.

Intel
-----

Intel GPUs also support the portable core, but the closest public compiler
model is not SPIR-V alone. Intel Graphics Compiler lowers through vISA and G4,
which expose a SIMD execution model directly.

The vISA execution model supports execution sizes 1, 2, 4, 8, 16, and 32. Each
instruction has execution-mask control such as ``M1`` through ``M8`` or
``NoMask``. Most instructions also support predicate operands. The effective
channel enable for an instruction is the combination of the current execution
mask selection and the predicate. Divergent control is represented with
``GOTO`` and reconvergence labels; uniform control can use jump-like control.

This maps well to the proposed structured split:

* ``scf.for``, ``scf.if``, and ``scf.while`` represent uniform structured
  control and can lower to uniform branches or scalar-control forms;
* ``where(mask)`` represents lane-varying predication and can lower either to
  predicated vISA operations or to ``GOTO``-style divergent control when a
  region cannot be represented as straight-line predication;
* ``mask<W>`` lowers to vISA predicate variables, execution-mask controls, or
  explicit mask values depending on the operation;
* lane-varying values lower to GRF-based SIMD values with legal vISA regions.

Intel also suggests two important refinements for the portable dialect. First,
uniformity should be richer than a boolean when used for memory lowering. IGC's
work-item analysis distinguishes patterns such as uniform, consecutive,
strided, and random. These categories are useful for selecting block, strided,
or gather/scatter memory messages. Second, memory operations should carry access
shape explicitly. Intel LSC operations distinguish gather/scatter, strided,
block2D, SLM, atomics, cache controls, transpose, VNNI transforms, and whether a
message honors channel enable.

Matrix operations also need target-specific layout information. Intel DPAS has
execution size 8 or 16 depending on platform and requires special GRF packing,
especially for the weight operand. This is analogous to AMDGPU MFMA/WMMA needing
target-specific fragment layouts, but the layouts are not interchangeable.

Public Intel Graphics Compiler documentation relevant to this mapping includes:

* `vISA execution model <https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/3_execution_model.md>`__;
* `vISA operands and predication <https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/5_operands.md>`__;
* `vISA LSC untyped memory operations <https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/LSC_UNTYPED.md>`__;
* `vISA GOTO <https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/GOTO.md>`__;
* `vISA BARRIER <https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/BARRIER.md>`__;
* `vISA DPAS <https://github.com/intel/intel-graphics-compiler/blob/master/documentation/visa/instructions/DPAS.md>`__.

Portable design consequence
---------------------------

The MLIR layer should be split conceptually into a portable subgroup core and
target-specific lowering dialects:

* the portable core defines scalar-uniform values, lane-varying values,
  ``mask<W>``, ``where``, uniform ``scf`` control, cross-lane collectives, and
  abstract memory shapes;
* AMDGPU lowering maps masks to ``EXEC``/``VCC``, scalar values to SGPR
  candidates, lane-varying values to VGPR/AGPR values, and shared memory to LDS;
* NVIDIA lowering maps masks to predicates, active masks, member masks, and
  SIMT branch structure;
* Intel lowering maps masks to vISA predicate and execution-mask controls,
  values to GRF regions, and memory operations to LSC or other message forms.

This keeps the AMDGPU design honest while avoiding a dialect that is portable in
name only. AMDGPU-specific terms such as ``EXEC``, ``VCC``, SGPR, VGPR, AGPR,
LDS, and MFMA should appear in the AMDGPU lowering layer, not in the generic
subgroup semantics.

Backend Reuse Strategy
======================

The explicit wave model does not require discarding the entire LLVM AMDGPU
backend. The existing backend contains a large amount of target knowledge that
is independent of the SIMT source abstraction and should be reused.

Useful existing pieces
----------------------

The following pieces are valuable even for a wave-native compiler:

* TableGen instruction definitions for SALU, VALU, VMEM, SMEM, LDS, image,
  MFMA, WMMA, and other AMDGPU instruction families;
* register definitions for SGPRs, VGPRs, AGPRs, ``EXEC``, ``VCC``, and special
  registers;
* subtarget feature modeling for GFX generations, wave32 and wave64 support,
  instruction availability, hazards, and target-specific restrictions;
* MC-layer assembly, disassembly, encoding, relocation, ELF, and code object
  support;
* HSA kernel descriptor and metadata emission;
* waitcnt insertion, memory legalizer logic, hazard recognition, scheduling
  models, and resource accounting;
* occupancy-related calculations for SGPRs, VGPRs, LDS, wave size, and waves per
  execution unit.

These components encode hardware facts rather than the virtual-thread
programming model. Reimplementing them would be a large and high-risk project.

Replaceable layers
------------------

The parts most worth replacing or bypassing are the layers that recover wave
semantics after they have been hidden by LLVM IR:

* lowering paths that start from scalar-looking per-work-item IR;
* divergence rediscovery where the source representation already knew which
  values were uniform or lane-varying;
* structurization and control-flow annotation whose purpose is to reconstruct
  ``EXEC`` manipulation from virtual-thread branches;
* late ``i1`` repair that converts divergent booleans into lane masks;
* late waterfall insertion caused by implicit divergent-to-uniform uses;
* instruction selection paths that must infer SGPR, VGPR, AGPR, and mask intent
  from generic IR.

The benefit of the wave model is not that all target complexity disappears. The
benefit is that the compiler no longer has to infer the core wave structure from
a representation that deliberately hides it.

Migration path
--------------

The recommended path is incremental:

1. Lower high-level languages to the MLIR wave dialect.
2. Lower the MLIR wave dialect to existing LLVM IR plus AMDGPU intrinsics, using
   the current backend as the first implementation target.
3. Introduce a wave-native lowering path that goes from MLIR to an
   AMDGPU-specific machine representation while still reusing existing
   instruction definitions, register definitions, subtarget data, MC emission,
   and machine-level passes.
4. Gradually replace SIMT-shaped instruction selection, divergence recovery, and
   control-flow reconstruction with direct lowering from uniform, wave, and mask
   operations.

A complete backend rewrite should not be the first step. A practical design
keeps the backend's hardware knowledge and replaces the semantic interface to
that backend.

Estimated difficulty
--------------------

The engineering cost depends on how much of the current backend is reused:

* lowering the MLIR wave dialect to the existing LLVM AMDGPU backend is an MVP
  project measured in months;
* lowering from the MLIR wave dialect directly to LLVM MIR while reusing AMDGPU
  machine passes is a multi-person-year effort;
* building a new AMDGPU code-generation pipeline while reusing TableGen, MC,
  metadata, and selected machine passes is a substantial multi-year project;
* rebuilding the entire backend, including encoders, object emission, hazards,
  scheduling, metadata, and resource modeling, is a very large project and
  unlikely to be the right starting point.

The productive target is a wave-native frontend and middle layer connected to a
selectively reused AMDGPU backend.

Example
=======

A wave-oriented kernel should look like a program over one hardware wave:

.. code-block:: c++

  kernel [[amdgpu_wave_size(32)]]
  void saxpy(float *x,
             float *y,
             float a,
             uint32_t n) {
    wave<uint32_t, 32> lane = lane_id<32>();
    uint32_t wave = wave_id_in_grid();

    wave<uint32_t, 32> i = wave * 32 + lane;
    mask<32> active = i < n;

    where (active) {
      wave<float, 32> xv = load(x + i);
      wave<float, 32> yv = load(y + i);
      store(y + i, a * xv + yv);
    }
  }

This program has one conceptual invocation per wave. ``a``, ``n``, ``x``, ``y``,
and ``wave`` are ordinary scalar values and therefore uniform. ``i`` is
lane-varying because it has a ``wave`` type, and the tail predicate is a lane
mask.

Type Rules
==========

The model should make SGPR/VGPR boundaries explicit.

Elementwise operations
  Arithmetic and logical operations on ``wave<T, W>`` are elementwise and execute
  only for active lanes.

Uniform operations
  Operations on ordinary scalar ``T`` values execute once for the wave.

Uniform broadcast
  A scalar ``T`` may be used in a ``wave<T, W>`` operation by broadcasting it to
  all active lanes. Code generation may still use an SGPR operand directly when
  the target instruction permits it.

Wave to uniform conversion
  A ``wave<T, W>`` cannot be implicitly converted to scalar ``T``. The program
  must use an explicit operation such as ``read_first``, ``read_lane``,
  ``reduce_*``, ``all_equal``, or ``assert_uniform``.

Mask creation
  Comparisons on ``wave<T, W>`` produce ``mask<W>``. Comparisons on
  scalar ``T`` values produce scalar ``bool``.

Mask application
  ``where (m)`` intersects the current active mask with ``m`` for the dynamic
  extent of the region. Nested masks compose by intersection.

Inactive lanes
  Values in inactive lanes are unspecified unless they are produced by an
  operation with explicit inactive-lane semantics, such as ``select``, ``merge``,
  or ``set_inactive``.

Control Flow
============

The model should prefer structured masked control flow over branch syntax that
pretends each lane has an independent program counter.

At the MLIR level, the dialect should consume two structured control-flow
families:

``where``
  A custom wave operation for lane-varying control. Its condition is a
  ``mask<W>``. It changes the active-lane mask for the dynamic extent of the
  region and may have an ``otherwise`` region for the complementary lanes.

``scf.for``, ``scf.if``, and ``scf.while``
  Standard MLIR structured control-flow operations for uniform control. Their
  bounds and conditions are ordinary scalar values, so the whole wave follows
  the same control path.

This split keeps the dialect small. It does not need custom loop and branch
operations for ordinary uniform control, and it does not abuse ``scf.if`` to
mean divergent lane control. A lane-varying condition produces a ``mask<W>`` and
must be consumed by ``where`` or by an explicit mask operation such as ``any``.

.. code-block:: c++

  mask<64> positive = x > 0.0f;

  where (positive) {
    y = sqrt(x);
  } otherwise {
    y = 0.0f;
  }

This lowers naturally to the existing AMDGPU control-flow pipeline:

* A lane-varying compare creates a ``VCC``/mask value.
* The ``where`` region saves the previous ``EXEC`` value and activates the
  matching lanes.
* The ``otherwise`` region activates the complementary lanes inside the original
  mask.
* The end of the construct restores or merges the saved mask.

Loops over masks should be expressed as uniform ``scf.while`` control around
explicit mask operations. For example, a source-level ``while_any`` construct can
lower to an ``scf.while`` whose condition is ``any(todo)``, with a nested
``where`` for the lanes selected in each iteration:

.. code-block:: c++

  mask<32> todo = active;
  while (any(todo)) {
    wave<uint32_t, 32> owner = choose_owner(todo);
    mask<32> group = todo & (owner == broadcast(read_first(owner)));
    where (group) {
      process_group();
    }
    todo = todo & ~group;
  }

This pattern makes waterfall-like execution explicit when the program needs it.

Primitive Operations
====================

The initial primitive set should map directly to existing AMDGPU concepts.

Lane identity
  ``lane_id<W>()`` returns a ``wave<uint32_t, W>`` containing the lane number.
  ``mbcnt``-style operations provide prefix counts within masks.

Masks
  ``ballot``, ``inverse_ballot``, ``any``, ``all``, ``none``, ``popcount``,
  ``first_lane``, and ``last_lane`` operate on ``mask<W>``.

Cross-lane movement
  ``read_first``, ``read_lane``, ``write_lane``, ``broadcast``, ``shuffle``,
  ``permute``, and DPP-style operations provide explicit lane communication.

Reductions and scans
  ``reduce_add``, ``reduce_min``, ``reduce_max``, ``reduce_and``,
  ``reduce_or``, and prefix-scan operations operate over active lanes.

Memory
  ``load`` and ``store`` over lane-varying addresses are wave memory operations.
  Scalar loads over uniform addresses remain scalar memory operations when legal.
  Higher-level operations can express coalesced global accesses, LDS tiling,
  global-to-LDS movement, and memory clauses.

Matrix operations
  MFMA, WMMA, and related operations should be expressed as wave-cooperative
  operations over typed fragments rather than as per-lane scalar operations.

Memory Model
============

The model should distinguish the following cases.

Uniform memory
  A load from a uniform pointer with a uniform address expression may lower to
  scalar memory instructions when the address space and target permit it.

Lane-varying memory
  A load or store from ``wave<ptr<T>, W>`` or from a uniform base plus
  lane-varying offset is a vector memory operation controlled by the active
  mask.

LDS memory
  LDS is shared by the work-group, not by the wave. The model should provide
  wave-level LDS operations and separate work-group synchronization primitives.
  Bank conflicts remain visible through address expressions and layout types.

Scratch/private memory
  Per-lane private storage should be modeled as lane-varying storage. The
  compiler should preserve the existing AMDGPU scratch layout constraints.

Ordering
  Same-wave ordering should be separate from cross-wave and cross-work-group
  ordering. GFX12 and newer targets have more explicit scoped cache and wait
  operations, which should be represented directly rather than hidden behind a
  generic barrier.

Lowering Strategy
=================

The source model should lower through the MLIR communication layer before
becoming ordinary LLVM IR plus AMDGPU intrinsics.

High-level language lowering
  Frontends lower their native syntax to the MLIR wave dialect. Multiple source
  languages can target the same dialect without each one reimplementing AMDGPU
  mask semantics, uniformity rules, or wave-size specialization.

MLIR wave dialect
  Preserve scalar-uniform values, ``wave<T, W>`` values, and ``mask<W>`` values
  until after type checking, uniformity checking, canonicalization, and
  structured ``where``/``scf`` regions are validated.

LLVM IR lowering
  Lower ``wave<T, W>`` values to ordinary scalar LLVM IR values that are known
  divergent. Do not lower them to LLVM ``<W x T>`` fixed vectors. LLVM vector
  types already have target meanings for packed per-lane values and register
  tuples in the AMDGPU backend.

Uniformity metadata
  Preserve uniformity information using existing divergence analysis inputs,
  attributes, metadata, or a target-specific intrinsic scheme. The backend
  should not have to rediscover facts that the source type system already
  proved.

Mask lowering
  Lower ``mask<32>`` to ``i32`` and ``mask<64>`` to ``i64``. Use existing
  ``llvm.amdgcn.*`` intrinsics for ballots, mask manipulation, and control-flow
  regions.

Control-flow lowering
  Lower uniform ``scf.for``, ``scf.if``, and ``scf.while`` as ordinary scalar
  structured control. Lower structured ``where`` regions to
  ``llvm.amdgcn.if``, ``llvm.amdgcn.else``, and ``llvm.amdgcn.end.cf`` or an
  equivalent target intrinsic sequence. The existing AMDGPU backend can then
  emit ``SI_IF``, ``SI_ELSE``, ``SI_END_CF``, and finally scalar ``EXEC``
  manipulation.

Register selection
  Use scalar-uniform values to guide SGPR selection and ``wave<T, W>`` values to
  guide VGPR/AGPR selection. Reject implicit divergent-to-uniform copies before
  instruction selection.

Generation-specific lowering
  Select wave32 or wave64 instruction forms from the subtarget. Use the existing
  wave-size predicates and ``LaneMaskConstants`` machinery for ``EXEC``/``VCC``
  register selection and scalar mask opcodes.

Verifier
========

The MLIR dialect verifier should reject programs that would otherwise silently
fall back to the SIMT illusion.

The verifier should check that:

* ``wave<T, W>`` values are not implicitly consumed by scalar-uniform
  operations.
* A ``mask<W>`` is only used with wave values of the same width.
* ``where`` regions are structured and do not leak temporary active-mask state.
* ``scf.if`` and ``scf.while`` conditions are scalar-uniform ``bool`` values,
  not ``mask<W>`` values.
* ``scf.for`` bounds and steps are scalar-uniform values.
* Control flow does not branch into or out of a ``where`` region except through
  structured region entry and exit.
* Cross-lane operations are marked convergent when required.
* Operations that require a uniform lane index, such as ``read_lane``, receive a
  scalar ``uint32_t`` index.
* Wave-size-polymorphic code is specialized before target instruction selection.

ABI and Launch Model
====================

A wave-oriented source language can still use the existing HSA kernel ABI, but
the logical launch shape changes.

The runtime still launches work-groups. Within each work-group, the program is
written in terms of waves rather than independent work-items. Work-group size
therefore determines the number of waves, and the final partial wave is
represented by an initial active mask.

Kernel inputs divide naturally into:

* uniform kernel arguments, passed through the normal kernarg mechanism and
  loaded into SGPRs when possible;
* wave-varying builtins such as lane id, derived from VGPR work-item id inputs;
* work-group and dispatch state, which is uniform for the wave.

Occupancy Controls
==================

The model should expose resource controls because they are central to wave-level
programming:

* requested wave size, 32 or 64;
* maximum VGPRs or AGPRs per wave;
* maximum SGPRs per wave;
* LDS bytes per work-group;
* minimum waves per execution unit;
* work-group size and waves per work-group;
* dynamic VGPR mode where supported.

These controls should map to existing AMDGPU function attributes, metadata, and
kernel descriptor fields where possible.

MVP
===

The first implementation should be deliberately small.

Required source concepts:

* ordinary scalar ``T`` values, which are uniform by default;
* ``wave<T, 32>`` and ``wave<T, 64>``;
* ``mask<32>`` and ``mask<64>``;
* fixed wave-size kernel attributes;
* custom ``where`` without arbitrary unstructured mask mutation;
* standard MLIR ``scf.for``, ``scf.if``, and ``scf.while`` for uniform
  structured control.

Required operations:

* lane id and wave id;
* elementwise arithmetic and compares;
* ``select`` and masked assignment;
* ``ballot``, ``any``, ``all``, and ``popcount``;
* ``read_first`` and ``broadcast``;
* simple reductions;
* masked global and LDS load/store.

Required lowering:

* source language constructs to the MLIR wave dialect;
* MLIR verification and canonicalization for wave operations, ``where`` regions,
  and consumed ``scf`` control flow;
* wave values to divergent scalar LLVM IR values;
* masks to ``i32`` or ``i64``;
* structured masks to existing AMDGPU control-flow intrinsics;
* explicit uniformity information to AMDGPU divergence analysis or register bank
  selection.

Risks and Open Questions
========================

Source language integration
  The MLIR dialect should be source-language neutral. C++, HIP-like extensions,
  domain-specific languages, and compiler-generated kernels should all lower to
  the same wave dialect rather than encoding AMDGPU wave semantics independently.

Dialect ownership
  The design must decide whether the communication layer is a new dialect, an
  extension of the existing MLIR AMDGPU dialect, or a layered design where a
  portable wave dialect lowers into AMDGPU-specific operations. The important
  invariant is that wave semantics remain explicit until AMDGPU-specific LLVM IR
  lowering.

Interaction with existing LLVM vector types
  The design must keep wave vectors distinct from LLVM fixed vectors. Reusing
  fixed vectors for lanes would conflict with existing AMDGPU lowering for
  packed data and register tuples.

Uniformity proof
  The model needs a robust way to carry source-level uniformity facts into LLVM
  without making existing middle-end optimizations unsound.

Inactive-lane semantics
  The language must choose precise rules for inactive lanes. Treating them as
  unspecified by default matches the hardware and avoids forcing unnecessary
  ``set_inactive`` operations.

Portability
  A wave-explicit model is intentionally AMDGPU-specific at first. A later
  design could abstract over other subgroup or warp machines, but that should
  not dilute the AMDGPU semantics.

Summary
=======

The AMDGPU backend already lowers programs to a scalar control path plus
wave-wide vector operations under ``EXEC``. An explicit wave programming model
should expose that structure to users:

* ordinary scalar ``T`` values map to SGPR/SALU by default;
* ``wave<T, W>`` maps to VGPR/VALU or AGPR operations;
* ``mask<W>`` maps to ``EXEC``/``VCC``-width scalar masks;
* ``where`` maps to structured ``EXEC`` manipulation;
* cross-lane communication, reductions, and wave-cooperative memory are explicit
  operations.

The result is a programming model that matches the hardware instead of
emulating independent threads on top of a wavefront machine.
