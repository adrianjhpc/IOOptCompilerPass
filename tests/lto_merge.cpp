// LTO test setup
// RUN: echo '#include <unistd.h>' > %t.logger.cpp
// RUN: echo '#include <string.h>' >> %t.logger.cpp
// RUN: echo '__attribute__((always_inline)) void write_payload(int fd, const char* data) { write(fd, data, strlen(data)); write(fd, "\n", 1); }' >> %t.logger.cpp

// This generates perfectly clean IR without prematurely running any optimizations!
// RUN: clang++-20 -O0 -Xclang -disable-O0-optnone -emit-llvm -c %s -o %t.main.bc
// RUN: clang++-20 -O0 -Xclang -disable-O0-optnone -emit-llvm -c %t.logger.cpp -o %t.logger.bc

// RUN: llvm-link-20 %t.main.bc %t.logger.bc -o %t.merged.bc

// RUN: opt-20 -load-pass-plugin=/home/adrianj/IOCompiler/build/src/libIOOpt.so -passes="default<O3>,function(io-opt)" %t.merged.bc -disable-output 2>&1 | FileCheck-20 %s
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
    // Without LTO, the compiler cannot merge these!
    write_payload(fd, "Cross-module LTO is working!");

    return 0;
}

// --- VERIFICATION ---
// We expect the LTO linker to successfully inline write_payload 
// and merge all 3 scattered writes into a single writev call.
// CHECK: [IOOpt] SUCCESS: N-Way converted 3 writes to writev!
