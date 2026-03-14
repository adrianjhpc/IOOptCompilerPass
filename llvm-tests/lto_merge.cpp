// LTO test setup
// RUN: echo '#include <unistd.h>' > %t.logger.cpp
// RUN: echo '#include <string.h>' >> %t.logger.cpp
// RUN: echo '__attribute__((always_inline)) void write_payload(int fd, const char* data) { write(fd, data, strlen(data)); write(fd, "\n", 1); }' >> %t.logger.cpp

// This generates perfectly clean IR without prematurely running any optimizations!
// RUN: %ppclang -O0 -Xclang -disable-O0-optnone -emit-llvm -c %s -o %t.main.bc
// RUN: %ppclang -O0 -Xclang -disable-O0-optnone -emit-llvm -c %t.logger.cpp -o %t.logger.bc

// RUN: %llvmlink %t.main.bc %t.logger.bc -o %t.merged.bc

// RUN: %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="default<O3>,function(io-opt)" %t.merged.bc -disable-output 2>&1 | %FileCheck %s
#include <unistd.h>
#include <string.h>

// External function defined in the dynamically generated logger.cpp
__attribute__((always_inline)) void write_payload(int fd, const char* data);

int main() {
    int fd = 1; // dummy fd (stdout)
    const char* header = "[LOG ENTRY]: ";

    // Write 1: Happens here in main.cpp
    write(fd, header, strlen(header));

    // Write 2 & 3: Happen inside logger.cpp.
    // Without LTO, the compiler cannot merge these
    write_payload(fd, "Cross-module LTO is working!");

    // Write 4: Add a footer to trigger the safe threshold (N>=4) for dynamic sizes
    const char* footer = " [DONE]\n";
    write(fd, footer, strlen(footer));

    return 0;
}

// --- VERIFICATION ---
// We expect the LTO linker to successfully inline write_payload
// and merge all 4 scattered writes into a single writev call.
// CHECK: [IOOpt] SUCCESS: N-Way converted 4 writes to writev!
