// RUN: %ppclang -O2 -fno-inline -emit-llvm -S -c %s -o - | %opt -load-pass-plugin=%shlibdir/libIOOpt%shlibext -passes=io-opt -S | %FileCheck %s

// --- MOCK MPI TYPES ---
typedef void* MPI_File;
typedef long long MPI_Offset;
typedef int MPI_Datatype;
struct MPI_Status { int count; int cancelled; int MPI_SOURCE; int MPI_TAG; int MPI_ERROR; };
#define MPI_CHAR 1

extern "C" int MPI_File_write_at(MPI_File fh, MPI_Offset offset, const void *buf,
                                 int count, MPI_Datatype datatype, MPI_Status *status);
extern "C" int MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
// ----------------------

// CHECK-LABEL: define dso_local void @_Z20test_mpi_status_trap
void test_mpi_status_trap(MPI_File fh, char* buf1, char* buf2) {
    MPI_Status status1, status2;
    
    // The pass MUST NOT batch these because the status pointers are different!
    // We use {{.*}} to ignore the specific register names (like %4 and %5) that Clang generates.
    // CHECK: call i32 @MPI_File_write_at(ptr {{.*}}, i64 {{.*}}0, ptr {{.*}}, i32 {{.*}}10, i32 {{.*}}1, ptr {{.*}})
    // CHECK: call i32 @MPI_File_write_at(ptr {{.*}}, i64 {{.*}}10, ptr {{.*}}, i32 {{.*}}10, i32 {{.*}}1, ptr {{.*}})
    // CHECK-NOT: shadow.buf    
    MPI_File_write_at(fh, 0, buf1, 10, MPI_CHAR, &status1);
    MPI_File_write_at(fh, 10, buf2, 10, MPI_CHAR, &status2);
    
    // If the pass batched the calls, status2 would be garbage memory here.
    int count1, count2;
    MPI_Get_count(&status1, MPI_CHAR, &count1);
    MPI_Get_count(&status2, MPI_CHAR, &count2);
}
