# IOOpt: Transparent I/O Coalescing, Hoisting & Prefetching for LLVM

[![LLVM: 20.0](https://img.shields.io/badge/LLVM-20.0%2B-blue.svg)](https://llvm.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**IOOpt** is a custom LLVM pass plugin that transparently reduces I/O syscall
overhead. It recognises a wide range of user-space I/O calls in the IR, proves
when it is safe to combine them, and rewrites them into fewer, larger, and
better-shaped operations — scatter/gather vectors (`readv`/`writev`/`preadv`/`pwritev`),
coalesced buffers, zero-copy kernel transfers, hoisted whole-loop transfers, and
optional kernel read-ahead hints (`posix_fadvise`).

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
* **Async:** `io_submit`, `aio_write`/`aio_write64` (recognised as batch breakers)
* **MPI-IO:** `MPI_File_write_at`, `MPI_File_read_at` (and `PMPI_*` variants)
* **C++ streams:** `std::ostream::write`, `std::istream::read` (resolved via a
  cached demangler lookup)

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
   to the later batching passes. Runs under LTO and under the explicit
   `io-lto-merge` pipeline.

2. **`IOLoopHoistingPass`** *(function)* — Uses `LoopInfo` + `ScalarEvolution` +
   `MemorySSA` + AA to prove that a per-iteration read/write over a contiguous,
   unit-stride buffer with a constant-length element and a computable trip count can
   be replaced by a **single** whole-range transfer. Reads are hoisted to the
   preheader; writes are lowered to the loop exit block. Turns *O(N)* syscalls into
   *O(1)*.

3. **`IOBatchingPass`** *(function)* — The core coalescing engine. A three-phase
   design:
   * **Decide:** walk each block read-only, growing per-fd batches and closing them
     on hazards, fd reassignment, opaque calls, sync points, or the high-water mark.
   * **Prepare:** classify each closed batch, apply the return-value availability
     gate, and pre-expand any non-dominating vectored buffers with `SCEVExpander`
     (insert-only, so analyses stay valid).
   * **Emit:** pure code generation that consults no analyses.

4. **`IOPrefetchPass`** *(function, opt-in)* — For analyzable, statically-bounded,
   forward-contiguous `pread` loops, inserts a single
   `posix_fadvise(POSIX_FADV_WILLNEED)` over the exact range the loop will consume.

5. **`IOSequentialPrefetchPass`** *(function, opt-in)* — Emits a whole-file
   `posix_fadvise` mode hint:
   * `SEQUENTIAL` for `read`/`fread` loops (unless a seek perturbs the position) and
     for contiguous / small-gap `pread` addrecs;
   * `RANDOM` for large-strided, backwards, non-affine, or non-constant-stride
     `pread` loops (to *suppress* wasteful read-ahead).

---

## The I/O Pattern Classifier

Each safe batch is routed to the cheapest correct primitive:

| Pattern | Trigger | Emitted form |
|---|---|---|
| **Contiguous** | buffers are physically adjacent (proved via SCEV or constant offset) | one call over the merged range; `splice`/`sendfile` counted as **zero-copy** |
| **Strided (SIMD gather)** | 2–64 writes of a uniform 1/2/4/8-byte element | vector load/insert into a stack shadow, one store, one write |
| **Static ShadowBuffer** | all lengths constant, total ≤ `IO_SHADOW_BUFFER_MAX` | stack alloca + `memcpy` packing + one call |
| **Dynamic ShadowBuffer** | batch ≥ threshold but sizes not all constant | aligned heap buffer via `posix_memalign` (with failure trap → `dprintf`+`abort`), packing, one `fwrite`/call, then `free` |
| **Vectored** | POSIX read/write batches | `readv`/`writev`/`preadv`/`pwritev`; auto-split when `iovcnt > IO_MAX_IOV` |
| **CharGather** | a run of `fputc`/`putc` | bytes gathered into one buffer, flushed with a single `fwrite` |

For explicit-offset I/O (`pread`/`pwrite`) and MPI `*_at`, contiguity is proved
**algebraically** on the offsets via SCEV, so dynamically-computed offsets are still
mergeable.

### Faithful return-value reconstruction
Merging changes the return values callers would have seen. IOOpt rebuilds each
original call's result from the merged result (byte-count clamping per slice, error
propagation, `fread`/`fwrite` element-count division, `fputc`/`fputs`/`puts` success
codes, C++ stream identity, MPI `MPI_SUCCESS`). If any *used* return value would be
needed *before* the merge point, the batch is rejected outright rather than
mis-optimised.

---

## Safety model

* **Alias / hazard analysis:** intervening reads/writes that touch a batched buffer,
  the file-stream object, or the fd slot break the batch (RAW/WAW for reads,
  WAR for writes).
* **Opaque calls** break batches unless proven pure or provably I/O-free
  (recursive `isDeeplySafeFromIO` scan); a curated allow-list covers `strlen`,
  byte-swap/endian helpers, etc.
* **Control flow:** a batch may only extend across blocks that post-dominate the
  last call and remain dominated by it.
* **Sync points** (`fsync`, `fdatasync`, `msync`, `sync_file_range`, `close`,
  `fclose`, `fflush`, `posix_fadvise`) flush the affected fd; `madvise` flushes
  everything.
* **Volatile** loads/stores/mem-intrinsics are always respected.

### ⚠️ Atomicity / message-boundary caveat
Merging N writes into one `writev`, or collapsing a loop into one transfer, changes
`PIPE_BUF` atomicity on pipes/FIFOs and message boundaries on datagram/seqpacket
sockets. IOOpt cannot prove "regular file" from IR, so this is governed by a single
honest switch: **`-io-opt-assume-regular-files` (default: on)**. Disable it if your
batched fds may be pipes, FIFOs, or datagram/seqpacket sockets.

---

## Control flags (CLI)

These are LLVM `cl::opt` booleans. Pass them to the plugin via `-mllvm` (or
`-Wl,-mllvm,...` at LTO link time).

| Flag | Default | Effect |
|---|---|---|
| `-enable-io-opt` | `true` | Master enable for batching/hoisting transforms. |
| `-io-opt-assume-regular-files` | `true` | Assume batched fds are regular files (see caveat). |
| `-io-opt-early-ipo` | `false` | Also inject interprocedural wrapper inlining at pipeline *start* (the explicit `io-lto-merge` pipeline always runs it regardless). |
| `-io-opt-prefetch` | `false` | **Master opt-in** for the entire `posix_fadvise` prefetch family. |
| `-io-opt-prefetch-willneed` | `true` | (needs `-io-opt-prefetch`) WILLNEED for analyzable `pread` ranges. |
| `-io-opt-prefetch-sequential` | `true` | (needs `-io-opt-prefetch`) SEQUENTIAL for monotonic contiguous read loops. |
| `-io-opt-prefetch-random` | `true` | (needs `-io-opt-prefetch`) RANDOM to suppress read-ahead on strided/non-affine `pread` loops. |

> **Why prefetch is off by default:** on random-access workloads (databases,
> chunked HDF5) a SEQUENTIAL hint widens kernel read-ahead and pollutes the page
> cache — a probable read regression. Prefetch is only requested deliberately.

## Tuning knobs (environment variables)

Numeric thresholds are read from the environment at load time. A value of `0` (or an
unparseable value) falls back to the default for threshold-style variables.

| Variable | Default | Meaning |
|---|---|---|
| `IO_BATCH_THRESHOLD` | `4` | Minimum scattered calls before preferring vectored/dynamic forms. |
| `IO_SHADOW_BUFFER_MAX` | `4096` | Max bytes packed on the stack (static ShadowBuffer / CharGather). |
| `IO_HIGH_WATER_MARK` | `65536` | Cumulative bytes that force a batch flush. |
| `IO_MAX_IOV` | `1024` | Max `iovcnt`; larger batches are split. |
| `IO_PREFETCH_MIN_BYTES` | `65536` | Smallest WILLNEED range worth a syscall. |
| `IO_PREFETCH_MAX_BYTES` | `134217728` | Largest WILLNEED range (avoid cache pollution). |
| `IO_PREFETCH_RANDOM_GAP` | `4096` | Stride gap beyond which a `pread` loop is classed RANDOM. |
| `IO_ENABLE_LOGGING` | `0` | Set non-zero to emit `[IOOpt]` diagnostics to stderr. |

Statistics (`NumBatchesMerged`, `NumLoopsHoisted`, `NumZeroCopy`, `NumIPAInlines`,
`NumPrefetchHints`, `NumSeqHints`, `NumRandomHints`, …) are available via LLVM's
`-stats` machinery.

---

## Bypassing IOOpt (manual opt-out)

For signalling writes (eventfd, sockets, IPC pipes) where immediate delivery
matters, you can force IOOpt to leave a call alone:

### Method 1 — opaque wrapper
```c
#include <unistd.h>
__attribute__((noinline, optnone))
static ssize_t write_signal(int fd, const void *buf, size_t count) {
    return write(fd, buf, count); // hidden from IOOpt
}
```

### Method 2 — inline-assembly barrier
```c
#define IO_OPT_BARRIER() __asm__ volatile("" ::: "memory")

write(fd, data1, 10);   // batched...
write(fd, data2, 10);   // ...with this.
IO_OPT_BARRIER();       // opaque memory hazard -> flush
write(ipc_fd, sig, 1);  // executes immediately, alone
IO_OPT_BARRIER();
write(fd, data3, 10);   // new batch starts here
```

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
> `LLVM_ENABLE_ABI_BREAKING_CHECKS` setting as the host tool that loads it. Keep them
> in lockstep.

```bash
git clone https://github.com/adrianjhpc/IOOptCompilerPass.git
cd IOOpt
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running the tests
```bash
make test
```

---

## Usage

### As part of an optimised / LTO build
Loading the plugin registers the function pipeline at *OptimizerLast*, so it runs at
`-O2`/`-O3`; under full LTO the interprocedural inliner runs as well.

**Makefile / C project:**
```bash
export CFLAGS="-O3 -flto"
export LDFLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so"
make
```

**CMake project:**
```bash
cmake . \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_C_FLAGS="-O3 -flto" \
  -DCMAKE_CXX_FLAGS="-O3 -flto" \
  -DCMAKE_EXE_LINKER_FLAGS="-flto -Wl,--load-pass-plugin=/path/to/libIOOpt.so"
```

### Explicit `opt` pipelines
```bash
# Function-level: hoist, batch, then (opt-in) prefetch
opt -load-pass-plugin=./libIOOpt.so -passes=io-opt in.ll -S -o out.ll

# Module-level: interprocedural wrapper inlining + full function pipeline
opt -load-pass-plugin=./libIOOpt.so -passes=io-lto-merge in.ll -S -o out.ll
```

Enable prefetch hints:
```bash
opt -load-pass-plugin=./libIOOpt.so \
    -io-opt-prefetch -io-opt-prefetch-sequential \
    -passes=io-opt in.ll -S -o out.ll
```

---

## Authors
Adrian Jackson

## License
Apache 2.0 — see [LICENSE](LICENSE).
