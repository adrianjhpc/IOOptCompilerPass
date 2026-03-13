// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

// Mock MPI Definitions to avoid requiring actual <mpi.h> during LLVM tests
typedef int MPI_File;
typedef int MPI_Datatype;
typedef long MPI_Offset;
struct MPI_Status { int count; int cancelled; int MPI_SOURCE; int MPI_TAG; int MPI_ERROR; };

extern "C" int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void *buf, int count, MPI_Datatype datatype, MPI_Status *status);

// -------------------------------------------------------------------------
// TEST: MPI CONTIGUOUS BATCHING
// -------------------------------------------------------------------------
// CHECK-LABEL: define {{.*}}test_mpi_batching
__attribute__((noinline))
void test_mpi_batching(MPI_File fh, const char* buffer, MPI_Datatype dt, MPI_Status* status, MPI_Offset base_offset) {

    // We provide 4 contiguous MPI_File_write_at calls.
    // The pass should sum the 'count' argument (10+10+10+10 = 40)
    // and emit a SINGLE MPI_File_write_at call.

    // Ensure we don't see the individual calls
    // CHECK-NOT: call {{.*}} @MPI_File_write_at(i32 {{.*}}, i64 {{.*}}, ptr {{.*}}, i32 10, i32 {{.*}}, ptr {{.*}})

    // CHECK: call {{.*}} @MPI_File_write_at(i32 {{.*}}, i64 {{.*}}, ptr {{.*}}, i32 40, i32 {{.*}}, ptr {{.*}})
    
    MPI_File_write_at(fh, base_offset + 0,  buffer + 0,  10, dt, status);
    MPI_File_write_at(fh, base_offset + 10, buffer + 10, 10, dt, status);
    MPI_File_write_at(fh, base_offset + 20, buffer + 20, 10, dt, status);
    MPI_File_write_at(fh, base_offset + 30, buffer + 30, 10, dt, status);
}
