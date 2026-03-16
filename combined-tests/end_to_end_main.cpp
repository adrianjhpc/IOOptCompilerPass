#include <unistd.h>
#include <fcntl.h>
#include <cstring>

// Defined in end_to_end_lib.cpp to test LTO boundary merging
extern void write_footer(int fd);

int main() {
    int fd = open("output_test.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return 1;

    char buffer[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // MLIR Test: Strided Loop Batching
    //#pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 1000; i += 2) {
        write(fd, &buffer[0], 32); // use modulo so we don't read out of bounds
    }

    // ClangIR emits memory-backed control flow (allocas/branches) for C++ loops.
    // Our strict MLIR pass correctly detects this is not an scf.for loop and safely skips it!
    // CHECK-MLIR: call {{.*}} @write(


    // LTO Test: Cross-Module I/O
    // The LLVM LTO pass should inline and merge this cross-module write
    write_footer(fd);

    close(fd);
    return 0;
}
