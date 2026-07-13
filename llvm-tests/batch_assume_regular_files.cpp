// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=MERGE
// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -io-opt-assume-regular-files=false -S | %FileCheck %s --check-prefix=NOMERGE

#include <unistd.h>

extern "C" {

// Two physically contiguous writes (buf[0..10) then buf[10..20)) with unused
// return values are a clean contiguous-merge candidate. This doubles as the
// positive batching guard: if merging ever silently stops working, MERGE fails.

// MERGE-LABEL: define {{.*}}@contig_writes
// NOMERGE-LABEL: define {{.*}}@contig_writes
void contig_writes(int fd, char* buf) {
    // Default: merged into a single 20-byte contiguous write.
    // MERGE: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}20)
    // MERGE-NOT: i64 {{.*}}10)

    // With -io-opt-assume-regular-files=false, batching is suppressed for the
    // same atomicity/message-boundary reason: both original 10-byte writes must
    // survive and no 20-byte merged write may appear.
    // NOMERGE-COUNT-2: call i64 @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    // NOMERGE-NOT: i64 {{.*}}20)
    write(fd, buf, 10);
    write(fd, buf + 10, 10);
}

}

