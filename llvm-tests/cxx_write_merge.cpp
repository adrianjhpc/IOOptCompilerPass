// RUN: %ppclang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <iostream>

// CHECK-LABEL: @_Z14test_cxx_writev
void test_cxx_write() {
    char buf[21] = "01234567890123456789";
    
    // FileCheck asserts that we see one write of length 20
    // CHECK: call ptr @_ZNSo5writeEPKcl(ptr {{.*}}, ptr {{.*}}, i64 20) 

    // FileCheck asserts we do not see another write after it
    // CHECK-NOT: call {{.*}} @_ZNSo5writeEPKcl
    std::cout.write(buf, 10);
    std::cout.write(buf + 10, 10);
}
