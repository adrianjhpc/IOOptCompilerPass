// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

#include <unistd.h>

// CHECK-LABEL: define {{.*}}test_alias_mutation
__attribute__((noinline))
void test_alias_mutation(int fd, char* buffer) {
    
    // The memory mutation (buffer[0] = 'X') happens between the writes.
    // The compiler must recognize the hazard and abort the batch.
    
    // CHECK-NOT: call {{.*}} @writev
    
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    // CHECK: store i8 88, ptr {{.*}}
    // CHECK: call {{.*}} @write(i32 {{.*}}, ptr {{.*}}, i64 {{.*}}10)
    
    write(fd, buffer, 10);
    buffer[0] = 'X'; // 88 in ASCII
    write(fd, buffer, 10);
}
