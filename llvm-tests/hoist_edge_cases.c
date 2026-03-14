// RUN: %clang -O0 -Xclang -disable-O0-optnone -emit-llvm -S %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes="mem2reg,instcombine,io-opt" -S | %FileCheck %s

#include <stdio.h>
#include <unistd.h>

// CHECK-LABEL: @test_param_dependency_c_std
int test_param_dependency_c_std(FILE **fp_arr, int index) {
    char buf[20];
    
    // Calculate the file pointer here
    // The fread cannot be hoisted above this line because it depends on 'fp'
    FILE *fp = fp_arr[index];
    
    // CHECK: getelementptr
    // CHECK: load ptr
    // CHECK: call i64 @fread
    fread(buf, 1, 20, fp);
    return buf[0];
}

// CHECK-LABEL: @test_param_dependency_posix
int test_param_dependency_posix(int start_fd) {
    char buf[20];
    
    // The file descriptor is altered before the read
    int actual_fd = start_fd + 5;
    
    // CHECK: add nsw i32
    // CHECK: call i64 @read
    read(actual_fd, buf, 20);
    
    return buf[0];
}
