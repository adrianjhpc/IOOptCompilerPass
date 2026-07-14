// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/IOOpt%shlibext -passes=io-opt -S | %FileCheck %s --check-prefix=OFF

#include <unistd.h>
extern "C" {
// OFF-LABEL: define {{.*}}@default_off
// OFF-NOT: posix_fadvise
void default_off(int fd, char* buf) {
    for (int i = 0; i < 1000; i++) read(fd, buf, 4096);
}
}

