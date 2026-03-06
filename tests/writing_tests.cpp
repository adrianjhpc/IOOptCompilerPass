// RUN: clang++-20 -O2 -fno-inline -emit-llvm -S -c %s -o - | opt-20 -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | FileCheck-20 %s

#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#define NOINLINE __attribute__((noinline))

// CHECK-LABEL: define {{.*}}test_contiguous_write
NOINLINE void test_contiguous_write(int fd) {
    char buffer[10] = "012345678";
    // We relax the regex to ignore 'noundef' and other attributes
    // CHECK: call {{.*}} @write(i32 {{.*}}%0, ptr {{.*}}, i64 4)
    write(fd, &buffer[0], 2);
    write(fd, &buffer[2], 2);
}

// CHECK-LABEL: define {{.*}}test_non_contiguous_write
NOINLINE void test_non_contiguous_write(int fd) {
    char buf1[50] = {0};
    char buf2[50] = {0};
    char buf3[50] = {0};

    // CHECK: %iovec.array.N = alloca [3 x { ptr, i64 }]
    // CHECK: call {{.*}} @writev(i32 {{.*}}, ptr %iovec.array.N, i32 3)
    
    // 3 writes * 50 bytes = 150 bytes total. 
    // 150 > 128, so the Dynamic Cost Model threshold drops to 3 and vectorizes
    write(fd, buf1, 50);
    write(fd, buf2, 50);
    write(fd, buf3, 50);
}

int main() {
    int fd = open("/dev/null", O_WRONLY);
    if (fd < 0) return 1;
    test_contiguous_write(fd);
    test_non_contiguous_write(fd);
    close(fd);
    return 0;
}
