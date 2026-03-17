#include <unistd.h>
#include <cstring>

// Manually define open so ClangIR doesn't treat it as a variadic (...) function!
extern "C" int open(const char *pathname, int flags, int mode);

extern void write_footer(int fd);

int main() {
    // 577 is the integer equivalent of O_WRONLY | O_CREAT | O_TRUNC on Linux
    int fd = open("output_test.txt", 577, 0644);
    if (fd < 0) return 1;

    char buffer[] = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

    // MLIR Test: Strided Loop Batching
    #pragma clang loop unroll(disable) vectorize(disable)
    for (int i = 0; i < 1000; i += 2) {
        write(fd, &buffer[0], 32); 
    }

    // --- FILECHECK VALIDATION ---
    // Ensure the individual writes in the loop were deleted:
    // CHECK-MLIR-NOT: call {{.*}} @write(
    // Ensure our pass injected the batched vector write:
    // CHECK-MLIR: call {{.*}} @writev(

    write_footer(fd);
    close(fd);
    return 0;
}
