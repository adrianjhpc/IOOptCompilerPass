# IOOpt: Transparent I/O Coalescing, Hoisting & Prefetching for LLVM

[![LLVM: 20.0](https://img.shields.io/badge/LLVM-20.0%2B-blue.svg)](https://llvm.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**IOOpt** is a custom LLVM pass plugin that transparently reduces I/O syscall
overhead. It recognises a wide range of user-space I/O calls in the IR, proves
when it is safe to combine them, and rewrites them into fewer, larger, and
better-shaped operations — scatter/gather vectors (`readv`/`writev`/`preadv`/`pwritev`),
coalesced buffers, zero-copy kernel transfers, in-kernel copies (`copy_file_range`),
hoisted whole-loop transfers, loop-collapsed vectored writes, and optional kernel
read-ahead hints (`posix_fadvise`).

All transforms are guarded by Alias Analysis (AA), Scalar Evolution (SCEV),
Dominator/Post-Dominator trees and MemorySSA. When safety cannot be proven, the
original calls are left untouched.

---

## What it recognises

IOOpt classifies calls to the following families (respecting `nobuiltin` and user
redefinitions — only library *declarations* are treated as I/O):

* **C stdio:** `fwrite`, `fread`, `fputc`, `putc`, `_IO_putc`, `fputs`, `puts`
* **POSIX byte I/O:** `write`, `read`, `pwrite`/`pwrite64`, `pread`/`pread64`
* **POSIX vectored:** `preadv`/`preadv2`, `pwritev`/`pwritev2`
* **Zero-copy:** `splice`, `sendfile`/`sendfile64`
* **Async:** `io_submit`, `aio_write`/`aio_write64` (recognised, and treated as
  hard batch breakers rather than merge candidates)
* **MPI-IO:** `MPI_File_write_at`, `MPI_File_read_at` (and `PMPI_*` variants)
* **C++ streams:** `std::ostream::write`, `std::istream::read` (resolved via a
  cached, thread-safe demangler lookup)

`fwrite`/`fread` element/count pairs are normalised to a byte count; `fputs`/`puts`
lengths are materialised with `strlen` at emission; `puts` gets its trailing newline
handled explicitly.

---

## The pass pipeline

IOOpt is composed of several cooperating passes so the pass manager can recompute
analyses between mutating stages:

1. **`InterProceduralIOBatchingPass`** *(module)* — Detects small "I/O wrapper"
   functions (a function under ~80 instructions whose I/O target is one of its
   arguments) and inlines a call to such a wrapper when it directly follows I/O on
   the *same* file descriptor in the caller. This exposes cross-function I/O chains
   to the later batching passes. Runs iteratively until it reaches a fixed point.
   (Runs in the `io-lto-merge` pipeline, under standard `-flto`, or when
   `-io-opt-early-ipo` is set — *not* in the plain function-level `-passes=io-opt`
   path.)

2. **`IOLoopHoistingPass`** *(function)* — Uses `LoopInfo` + `ScalarEvolution` +
   `MemorySSA` + AA to transform per-iteration I/O loops in two ways:

   * **Whole-loop hoisting.** Proves that a per-iteration read/write over a
     contiguous, unit-stride buffer with a constant-length element and a computable
     trip count can be replaced by a **single** whole-range transfer. Reads are
     hoisted to the preheader; writes are lowered to the loop exit block. Turns
     *O(N)* syscalls into *O(1)*. By default the trip count must be a compile-time
     constant; `-io-opt-loop-hoist-dynamic-trips` additionally permits a runtime
     (symbolic) trip count, using a conservative unbounded AA range for the
     hazard proof.

   * **Dynamic Vectored Loop Collapse (DVLC)** *(opt-in)* — When the buffer is
     *scattered* rather than contiguous (constant stride **greater than** the
     per-call count, i.e. non-overlapping forward slots), a loop of
     `write`/`pwrite` with an **unused return** and a constant trip count in
     `[2, IO_MAX_IOV]` is collapsed into a single `writev`/`pwritev` executed in
     the loop exit block. Only pointers are moved into an `iovec` array — there is
     **no data copy**. Requires `-io-opt-loop-vectored`. Proven safe via a
     no-clobber MemorySSA/SCEV check that every intervening store lands in its own
     per-iteration slot, plus contiguity of the `pwrite` file offset. Counted as
     `NumLoopsVectorCollapsed`.

   After any mutating transform it invalidates and refetches analyses before
   touching further loops.

3. **`IOCopyFileRangePass`** *(function, opt-in)* — Recognises the userspace
   "bounce buffer" copy idiom:

   ```c
   n = read (src, buf, count);            write (dst, buf, n);
   n = pread(src, buf, count, off_in);    pwrite(dst, buf, n, off_out);
   ```

   and rewrites it into a single in-kernel `copy_file_range()`, eliminating the
   user↔kernel bounce entirely. Fires only when:
   * the read and write are in the **same** basic block, write strictly after read;
   * `buf` is a **non-escaping alloca** used *only* as the syscall buffer of the two
     calls (a genuine dedicated bounce buffer), so dropping the separate read cannot
     change any other observable value;
   * the write length is exactly the read's SSA result (the provably-correct
     short-read idiom), or — under `-io-opt-cfr-assume-full-reads` — a constant equal
     to the read count;
   * `src` and `dst` are distinct fds (avoids same-file overlap UB).

   By default (`-io-opt-cfr-fallback`) the promoted call is guarded by a runtime
   fallback that re-runs the original read+write on `ENOSYS` (old kernel) or `EXDEV`
   (cross-filesystem), merging both return values with phis; both errnos occur
   *before* any bytes are copied, so the fallback starts from an unperturbed
   position. Performs one transform per DominatorTree, then invalidates/refetches
   (the fallback path splits blocks). Synthesised fallback calls are tagged
   `io.cfr.nofold` so the re-scan never re-folds its own fallback.

   * **copy_file_range copy loops** *(opt-in, `-io-opt-cfr-loops`)* — Beyond the
     single-block idiom, IOOpt can recognise a whole **read/write copy loop** (one
     `read` filling a dedicated bounce alloca, one lagged `write` of the read's
     result, a count-controlled exit) and promote it to `copy_file_range`. When
     `-io-opt-cfr-fallback` is on, the original loop is **preserved** and reached
     via an `ENOSYS`/`EXDEV` guard: a probe `copy_file_range` runs first and, if it
     succeeds, an in-kernel copy loop drives the transfer to EOF; on the fallback
     errnos control drops into the untouched original loop. With the fallback off,
     each read is rewritten in place to `copy_file_range` and the lagged write is
     deleted (same-filesystem / current-kernel only). Counted as `NumCFRLoops`.

4. **`IOBatchingPass`** *(function)* — The core coalescing engine. A three-phase
   design:
   * **Decide:** walk each block read-only, growing per-fd batches and closing them
     on hazards, fd reassignment, opaque calls, sync points, or the high-water mark.
   * **Prepare:** classify each closed batch, apply the return-value availability
     gate, and pre-expand any non-dominating vectored buffers with `SCEVExpander`
     (insert-only, so analyses stay valid across all batches).
   * **Emit:** pure code generation that consults no analyses.

5. **`IOPrefetchPass`** *(function, opt-in)* — For analyzable, statically-bounded,
   forward-contiguous `pread` loops with a single count-controlled exit, inserts a
   single `posix_fadvise(POSIX_FADV_WILLNEED)` over the exact range the loop will
   consume.

6. **`IOSequentialPrefetchPass`** *(function, opt-in)* — Emits a whole-file
   `posix_fadvise` mode hint for read loops that iterate at least 4 times:
   * `SEQUENTIAL` for `read`/`fread` loops (unless a seek perturbs the position) and
     for contiguous / small-gap `pread` addrecs;
   * `RANDOM` for large-strided, backwards, non-affine, or non-constant-stride
     `pread` loops (to *suppress* wasteful read-ahead).

Both prefetch passes tag the calls they touch with metadata (`io.prefetched`,
`io.seq.hinted`) so re-running the pipeline is idempotent.

---

## The I/O Pattern Classifier

Each safe batch is routed to the cheapest correct primitive:

| Pattern | Trigger | Emitted form |
|---|---|---|
| **Contiguous** | buffers are physically adjacent (proved via SCEV or constant offset) | one call over the merged range; `splice`/`sendfile` counted as **zero-copy** |
| **Strided (SIMD gather)** | 2–64 writes of a uniform 1/2/4/8-byte element | vector load/insert into a 64-byte-aligned stack shadow, one store, one write |
| **Static ShadowBuffer** | all lengths constant, total ≤ `IO_SHADOW_BUFFER_MAX` | stack alloca + `memcpy` packing + one call |
| **Dynamic ShadowBuffer** | batch ≥ threshold but sizes not all constant | aligned heap buffer via `posix_memalign` (failure trap → `dprintf`+`abort`+`unreachable`), packing, one `fwrite`/call, then `free` |
| **Vectored** | POSIX read/write batches | `readv`/`writev`/`preadv`/`pwritev`; auto-split when `iovcnt > IO_MAX_IOV` and each chunk re-classified |
| **CharGather** | a run of `fputc`/`putc` | bytes gathered into one buffer, flushed with a single `fwrite` |

For explicit-offset I/O (`pread`/`pwrite`) and MPI `*_at`, contiguity is proved
**algebraically** on the offsets via SCEV, so dynamically-computed offsets are still
mergeable (MPI additionally requires matching datatype and file-handle operands).

### Faithful return-value reconstruction
Merging changes the return values callers would have seen. IOOpt rebuilds each
original call's result from the merged result: per-slice byte-count clamping, error
propagation, `fread`/`fwrite` element-count division (with divide-by-zero guard),
`fputc`/`fputs`/`puts` success codes, C++ stream identity, and MPI `MPI_SUCCESS`. If
any *used* return value would be needed *before* the merge point, the batch is
rejected outright rather than mis-optimised (`NumBatchesRejectedUnsafeUse`).

`copy_file_range` promotion likewise reconstructs both the read's and the write's
return values (via phis when the ENOSYS/EXDEV fallback is enabled). The DVLC
loop-collapse transform applies only to writes whose return is unused, so no
reconstruction is needed there.

---

## Safety model

* **Alias / hazard analysis:** intervening reads/writes that touch a batched buffer,
  the file-stream object, or the fd slot break the batch (RAW/WAW for reads,
  WAR for writes).
* **Opaque calls** break batches unless proven pure or provably I/O-free
  (recursive `isDeeplySafeFromIO` scan); a curated allow-list covers `strlen`,
  `strnlen`, `strcmp`, byte-swap/endian helpers (`htons`/`htonl`/`ntohs`/`ntohl`/
  `bswap_32`/`bswap_64`), and debug/assume intrinsics.
* **Control flow:** a batch may only extend across blocks that post-dominate the
  last call and remain dominated by it.
* **Sync points** (`fsync`, `fdatasync`, `msync`, `sync_file_range`, `close`,
  `fclose`, `fflush`, `posix_fadvise`/`posix_fadvise64`) flush the affected fd;
  `madvise` flushes everything.
* **Volatile** loads/stores/mem-intrinsics are always respected.
* **Async I/O** (`io_submit`, `aio_write`), and already-vectored `preadv`/`pwritev`,
  are never merged.
* **Loop collapse (hoisting / DVLC / CFR loops):** requires LoopSimplify + LCSSA
  form, a computable trip count, loop-invariant fd/stream, and a MemorySSA/SCEV
  proof that no interleaved I/O, opaque side-effecting call, or out-of-slot store
  can perturb the deferred transfer.

### ⚠️ Atomicity / message-boundary caveat
Merging N writes into one `writev`, or collapsing a loop into one transfer, changes
`PIPE_BUF` atomicity on pipes/FIFOs and message boundaries on datagram/seqpacket
sockets. IOOpt cannot prove "regular file" from IR, so this is governed by a single switch: **`-io-opt-assume-regular-files` (default: on)**. Disable it if your
batched fds may be pipes, FIFOs, or datagram/seqpacket sockets — this also disables
batching, loop hoisting, DVLC loop collapse, `copy_file_range` promotion, and
prefetch, all of which share the assumption.

---

## Control flags (CLI)

These are LLVM `cl::opt` booleans. Pass them to the plugin via `-mllvm` (or
`-Wl,-mllvm,...` at LTO link time).

| Flag | Default | Effect |
|---|---|---|
| `-enable-io-opt` | `true` | Master enable for batching/hoisting transforms. |
| `-io-opt-assume-regular-files` | `true` | Assume batched fds are regular files (see caveat). Disabling it turns off batching, hoisting, DVLC, `copy_file_range`, and prefetch. |
| `-io-opt-early-ipo` | `false` | Also inject interprocedural wrapper inlining at pipeline *start* (the explicit `io-lto-merge` pipeline and standard `-flto` always run it regardless). |
| `-io-opt-loop-vectored` | `false` | Collapse a loop of *scattered* (stride > count, non-overlapping) `write`/`pwrite` with an unused return into one `writev`/`pwritev` (no data copy). |
| `-io-opt-loop-hoist-dynamic-trips` | `false` | Allow whole-loop I/O collapse/hoist with a runtime (symbolic) trip count, using a conservative unbounded AA range. |
| `-io-opt-copy-file-range` | `false` | Promote read+write / pread+pwrite bounce-buffer copies into a single in-kernel `copy_file_range`. |
| `-io-opt-cfr-fallback` | `true` | Guard `copy_file_range` with a read/write fallback on `ENOSYS`/`EXDEV` (preserves old-kernel / cross-filesystem correctness). Also selects the loop-preserving fallback form of copy-loop promotion. |
| `-io-opt-cfr-assume-full-reads` | `false` | (requires `-io-opt-copy-file-range`) Also match copies whose write length is a constant equal to the read count (changes short-read/EOF behaviour). |
| `-io-opt-cfr-loops` | `false` | (requires `-io-opt-copy-file-range`) Recognise whole **read/write copy loops** and promote them to `copy_file_range`. |
| `-io-opt-prefetch` | `false` | **Master opt-in** for the entire `posix_fadvise` prefetch family. |
| `-io-opt-prefetch-willneed` | `true` | (requires master) `WILLNEED` for analyzable `pread` ranges. |
| `-io-opt-prefetch-sequential` | `true` | (requires master) `SEQUENTIAL` for monotonic contiguous read loops. |
| `-io-opt-prefetch-random` | `true` | (requires master) `RANDOM` to suppress read-ahead on strided/non-affine `pread` loops. |

---

## copy_file_range promotion (opt-in)

IOOpt can collapse the classic userspace "bounce buffer" copy — read into a
temporary, then write it back out — into a single in-kernel `copy_file_range()`,
removing the user↔kernel data bounce entirely.

This is **off by default**, because it changes the syscall surface and has real
`EXDEV`/`ENOSYS` edge cases. Enable it with:

    -io-opt-copy-file-range

### What it matches (single-block idiom)

    // implicit-offset form (uses the fd's current position)
    n = read(src, buf, count);
    write(dst, buf, n);

    // explicit-offset form (copy_file_range advances the pointed-to offsets,
    // exactly like pread/pwrite leave the fd offset untouched)
    n = pread(src, buf, count, off_in);
    pwrite(dst, buf, n, off_out);

Matching is deliberately conservative for correctness:

* read and write must be in the **same basic block**, write strictly after read;
* `buf` must be a **non-escaping alloca** used *only* by those two syscalls (plus
  pointer-forwarding casts/GEPs and lifetime/debug intrinsics) — a genuine dedicated
  bounce buffer;
* the write length must be **exactly the read's SSA result** (the provably-correct
  short-read idiom). Under `-io-opt-cfr-assume-full-reads` a *constant* write length
  equal to the read count is also matched, at the cost of changing behaviour on the
  short-read/EOF edge;
* `src` and `dst` must be **distinct fds** (avoids same-file overlap UB);
* no intervening I/O call, and no non-readonly opaque call, may sit between the read
  and the write (that would be reordered past the combined transfer).

### Copy loops (`-io-opt-cfr-loops`)

With `-io-opt-cfr-loops` (and `-io-opt-copy-file-range`), IOOpt also recognises the
loop form of the same idiom — a `read` filling a dedicated bounce alloca, a lagged
`write` of the read's result, and a count/EOF-controlled exit:

    while ((n = read(src, buf, count)) > 0)
        write(dst, buf, n);

* **With `-io-opt-cfr-fallback` (default):** the original loop is left **intact** and
  guarded. A probe `copy_file_range` runs first; on success an in-kernel copy loop
  drives the transfer to EOF, and on `ENOSYS`/`EXDEV` control falls into the
  untouched original read/write loop. This preserves old-kernel / cross-filesystem
  correctness while getting the in-kernel fast path everywhere else.
* **Without fallback:** each read is rewritten in place to `copy_file_range` and the
  lagged write is deleted. Documented as **same-filesystem / current-kernel only**.

Counted as `NumCFRLoops`.

### Fallback behaviour (default on)

With `-io-opt-cfr-fallback` (the default), the emitted single-block code guards the
`copy_file_range` call: if it fails with `ENOSYS` (kernel too old) or `EXDEV`
(cross-filesystem copy), the original `read`+`write` pair is re-run and the two
return values are merged back with phis. Because both errnos are reported *before*
any bytes are copied, the fallback re-read/-write starts from an unperturbed
position, so a working cross-mount copy is never turned into a hard failure.

Disabling the fallback (`-io-opt-cfr-fallback=false`) trusts `copy_file_range`
unconditionally and emits smaller code, but will fail on old kernels or cross-mount
copies — only use it when you control the deployment environment.

> **Note:** `copy_file_range` promotion also requires `-io-opt-assume-regular-files`
> (the default), since it only makes sense on seekable/regular files.

---

## Scatter-loop collapse to writev/pwritev (opt-in)

Beyond contiguous whole-loop hoisting, IOOpt can collapse a loop of **scattered**
writes into a single vectored call. This is **off by default**; enable it with:

    -io-opt-loop-vectored

### What it matches

    for (int i = 0; i < N; i++)
        write(fd, &buf[i * stride], count);   // stride > count, non-overlapping

* the call must be `write` or `pwrite` with an **unused return**;
* a **constant** trip count `N` in `[2, IO_MAX_IOV]` and a **constant** per-call byte
  count;
* the buffer must be an affine addrec with a **constant stride ≥ count** (forward,
  non-overlapping slots — a stride *equal* to count is the contiguous case already
  handled by hoisting);
* the file offset of a `pwrite` must advance contiguously (step == count);
* the fd/stream must be loop-invariant, and a MemorySSA/SCEV proof must show every
  intervening store lands in its own per-iteration slot, with no interleaved I/O or
  opaque side-effecting calls.

IOOpt then builds an `iovec` array (populated by a pointer-IV as the loop runs — no
data is copied) and emits **one** `writev`/`pwritev` in the loop exit block. Counted
as `NumLoopsVectorCollapsed`.

---

## Prefetch / Read-ahead hints (opt-in)

IOOpt can insert kernel read-ahead advice (`posix_fadvise`) for read loops. This is
**entirely opt-in and off by default**, because on random-access workloads
(databases, chunked HDF5) a `SEQUENTIAL` hint widens kernel read-ahead and pollutes
the page cache — which we observed as a probable read regression. It is only a win
for genuinely streaming access, so you must request it deliberately.

There are two independent prefetch passes, both gated behind one master switch:

| Advice emitted | When | Purpose | Pass |
|---|---|---|---|
| `WILLNEED` (`POSIX_FADV_WILLNEED`, 3) | analyzable, statically-bounded, forward-contiguous `pread` loop | pre-warm the exact byte range the loop will consume | `IOPrefetchPass` |
| `SEQUENTIAL` (`POSIX_FADV_SEQUENTIAL`, 2) | `read`/`fread` loops (no perturbing seek), or contiguous / small-gap `pread` | widen read-ahead for streaming | `IOSequentialPrefetchPass` |
| `RANDOM` (`POSIX_FADV_RANDOM`, 1) | large-strided, backwards, non-affine, or non-constant-stride `pread` loops | **suppress** wasteful read-ahead | `IOSequentialPrefetchPass` |

`SEQUENTIAL`/`RANDOM` mode hints are only emitted for loops that run at least 4
iterations. `WILLNEED` ranges must fall between `IO_PREFETCH_MIN_BYTES` and
`IO_PREFETCH_MAX_BYTES` (a non-zero length is required, since `posix_fadvise` treats
`len == 0` as "to EOF"). For `fread`, the FILE* is bridged to an integer fd via
`fileno`; POSIX reads already have one; C++ streams expose no portable fd and are
skipped.

### Opting IN

Nothing happens unless you set the master switch:

    # Turn the whole prefetch family on (sub-switches default to enabled)
    -io-opt-prefetch

With just `-io-opt-prefetch`, all three advice kinds (`WILLNEED`, `SEQUENTIAL`,
`RANDOM`) are active.

### Opting OUT of individual kinds

Once the master switch is on, each kind can be turned off independently:

    # WILLNEED range prefetch only; no whole-file SEQUENTIAL/RANDOM mode hints
    -io-opt-prefetch -io-opt-prefetch-sequential=false -io-opt-prefetch-random=false

    # SEQUENTIAL streaming hints only
    -io-opt-prefetch -io-opt-prefetch-willneed=false -io-opt-prefetch-random=false

    # RANDOM suppression only (stop the kernel over-reading on strided pread loops),
    # without ever emitting a SEQUENTIAL/WILLNEED hint
    -io-opt-prefetch -io-opt-prefetch-willneed=false -io-opt-prefetch-sequential=false

If both `-io-opt-prefetch-sequential` and `-io-opt-prefetch-random` are `false`, the
`IOSequentialPrefetchPass` skips itself entirely.

### Opting OUT entirely

Simply omit `-io-opt-prefetch` (the default). The sub-switches have **no effect**
unless the master switch is set, so leaving it off disables all read-ahead advice
regardless of the sub-switch values.

> **Note:** prefetch hints also require `-io-opt-assume-regular-files` (the default),
> since `posix_fadvise` is only meaningful on seekable/regular files.

At LTO link time, pass these through the linker, e.g.:

    LDFLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so \
             -Wl,-mllvm,-io-opt-prefetch -Wl,-mllvm,-io-opt-prefetch-random=false"

---

## Tuning knobs (environment variables)

Numeric thresholds are read from the environment at load time. A value of `0` (or an
unparseable value) falls back to the default for these threshold-style variables.

| Variable | Default | Meaning |
|---|---|---|
| `IO_BATCH_THRESHOLD` | `4` | Minimum scattered calls before preferring vectored/dynamic forms. |
| `IO_SHADOW_BUFFER_MAX` | `4096` | Max bytes packed on the stack (static ShadowBuffer / CharGather). |
| `IO_HIGH_WATER_MARK` | `65536` | Cumulative bytes that force a batch flush. |
| `IO_MAX_IOV` | `1024` | Max `iovcnt`; larger batches are split and re-classified. Also bounds DVLC loop-collapse trip counts. |
| `IO_PREFETCH_MIN_BYTES` | `65536` | Smallest `WILLNEED` range worth a syscall. |
| `IO_PREFETCH_MAX_BYTES` | `134217728` (128 MiB) | Largest `WILLNEED` range (avoid cache pollution). |
| `IO_PREFETCH_RANDOM_GAP` | `4096` | Stride gap beyond which a `pread` loop is classed `RANDOM`. |
| `IO_ENABLE_LOGGING` | `0` | Set non-zero to emit `[IOOpt]` diagnostics to stderr. |

### Statistics
Compile-time counters are exposed through LLVM's `-stats` machinery:

`NumFunctionsAnalyzed`, `NumBatchesMerged`, `NumLoopsHoisted`,
`NumLoopsVectorCollapsed`, `NumZeroCopy`, `NumIPAInlines`,
`NumBatchesRejectedUnsafeUse`, `NumCopyFileRange`, `NumCFRLoops`,
`NumPrefetchHints`, `NumSeqHints`, `NumRandomHints`.

---

## Bypassing IOOpt (manual opt-out)

For signalling writes (eventfd, sockets, IPC pipes) where immediate delivery
matters, you can force IOOpt to leave a call alone:

### Method 1 — opaque wrapper

    #include <unistd.h>
    __attribute__((noinline, optnone))
    static ssize_t write_signal(int fd, const void *buf, size_t count) {
        return write(fd, buf, count); // hidden from IOOpt
    }

### Method 2 — inline-assembly barrier

    #define IO_OPT_BARRIER() __asm__ volatile("" ::: "memory")

    write(fd, data1, 10);   // batched...
    write(fd, data2, 10);   // ...with this.
    IO_OPT_BARRIER();       // opaque memory hazard -> flush
    write(ipc_fd, sig, 1);  // executes immediately, alone
    IO_OPT_BARRIER();
    write(fd, data3, 10);   // new batch starts here

You can also disable batching globally with `-enable-io-opt=false`, or turn off
merges that change socket/pipe semantics with `-io-opt-assume-regular-files=false`.

---

## Building

### Prerequisites
* LLVM / Clang 20.0+
* CMake 3.10+
* A C++17 compiler
* `lit` (only to run the test suite)

> **ABI note:** the plugin defines `EnableABIBreakingChecks` /
> `DisableABIBreakingChecks` so it can be loaded by `clang`/`lld` during LTO. This is
> only safe if the plugin is built against an LLVM with the **identical**
> `LLVM_ENABLE_ABI_BREAKING_CHECKS` setting as the host tool that loads it. A
> mismatch changes the layout of core LLVM data structures and causes silent memory
> corruption rather than a clean load error — keep them in lockstep.

    git clone https://github.com/adrianjhpc/IOOptCompilerPass.git
    cd IOOpt
    mkdir build && cd build
    cmake ..
    make -j$(nproc)

### Running the tests

    make test

---

## Usage

### As part of an optimised / LTO build
Loading the plugin registers the function pipeline at *OptimizerLast*, so it runs at
`-O2`/`-O3`. Under full LTO (and via the explicit `io-lto-merge` pipeline) the
interprocedural wrapper inliner runs as well. Each function pipeline is prefixed with
`LoopSimplify` + `LCSSA` so the hoisting and SCEV-based analyses have the canonical
loop form they need.

**Makefile / C project:**

    export CFLAGS="-O3 -flto"
    export LDFLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so"
    make

**CMake project:**

    cmake . \
      -DCMAKE_C_COMPILER=clang \
      -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_C_FLAGS="-O3 -flto" \
      -DCMAKE_CXX_FLAGS="-O3 -flto" \
      -DCMAKE_EXE_LINKER_FLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so"

### Explicit `opt` pipelines

    # Function-level: hoist, batch, then (opt-in) prefetch
    opt -load-pass-plugin=./libIOOpt.so -passes=io-opt in.ll -S -o out.ll

    # Module-level: interprocedural wrapper inlining + full function pipeline
    opt -load-pass-plugin=./libIOOpt.so -passes=io-lto-merge in.ll -S -o out.ll

Enable prefetch hints:

    opt -load-pass-plugin=./libIOOpt.so \
        -io-opt-prefetch -io-opt-prefetch-sequential \
        -passes=io-opt in.ll -S -o out.ll

Enable `copy_file_range` promotion (including copy loops):

    opt -load-pass-plugin=./libIOOpt.so \
        -io-opt-copy-file-range -io-opt-cfr-loops \
        -passes=io-opt in.ll -S -o out.ll

Enable scatter-loop collapse to `writev`/`pwritev`:

    opt -load-pass-plugin=./libIOOpt.so \
        -io-opt-loop-vectored \
        -passes=io-opt in.ll -S -o out.ll

---

## Caveats & assumptions

* **`off_t` is assumed 64-bit** (Large File Support). Prefetch and
  `copy_file_range` offset/length arguments are emitted as `i64`.
* **`POSIX_FADV_*` constants** are hard-coded to their mainstream Linux/glibc values
  (`RANDOM=1`, `SEQUENTIAL=2`, `WILLNEED=3`).
* **`ENOSYS`/`EXDEV` constants** used by the `copy_file_range` fallback are likewise
  hard-coded to their mainstream Linux/glibc values (`ENOSYS=38`, `EXDEV=18`).
* **fd identity** is tracked through a load-from-slot, so a stored-then-reloaded fd
  compares equal across batching and prefetch decisions.
* Correctness of batch ordering never relies on fd-identity keys alone — every
  intervening I/O or opaque call is treated as a batch breaker regardless.

---

## Authors
Adrian Jackson

## License
Apache 2.0 — see [LICENSE](LICENSE).

