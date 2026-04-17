#include <mpi.h>
#include <iostream>
#include <vector>
#include <chrono>

#define NUM_ITERATIONS 10000
#define CHUNK_SIZE 1024

// We use a macro to manually unroll the loop, ensuring the compiler's Phase 4 
// block-level batcher sees them as adjacent instructions and collapses them!
#define WRITE_CHUNK(i) \
    MPI_File_write_at(fh, base_offset + (i) * CHUNK_SIZE, buffer.data() + (i) * CHUNK_SIZE, CHUNK_SIZE, MPI_BYTE, &status)

__attribute__((noinline))
void write_16_chunks(MPI_File fh, MPI_Offset base_offset, const std::vector<char>& buffer) {
    MPI_Status status;
    // 16 distinct MPI calls. 
    // The IOOpt pass will mathematically prove these are contiguous and rewrite 
    // this function to contain exactly ONE call with a length of 16384 bytes!
    WRITE_CHUNK(0);  WRITE_CHUNK(1);  WRITE_CHUNK(2);  WRITE_CHUNK(3);
    WRITE_CHUNK(4);  WRITE_CHUNK(5);  WRITE_CHUNK(6);  WRITE_CHUNK(7);
    WRITE_CHUNK(8);  WRITE_CHUNK(9);  WRITE_CHUNK(10); WRITE_CHUNK(11);
    WRITE_CHUNK(12); WRITE_CHUNK(13); WRITE_CHUNK(14); WRITE_CHUNK(15);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File fh;
    const char* filename = "mpi_ioopt_benchmark.dat";
    
    // Open file for writing
    if (MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) {
        if (rank == 0) std::cerr << "Error opening file.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Create a 16KB payload buffer
    std::vector<char> buffer(16 * CHUNK_SIZE, 'A' + rank);

    MPI_Offset per_rank_bytes = (MPI_Offset)NUM_ITERATIONS * (16 * CHUNK_SIZE);
    MPI_Offset rank_base = (MPI_Offset)rank * per_rank_bytes;
    
    // Warmup
    write_16_chunks(fh, rank_base, buffer);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    // The core workload: 10,000 checkpoints
    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        MPI_Offset offset = rank_base + (MPI_Offset)i * (16 * CHUNK_SIZE);
        write_16_chunks(fh, offset, buffer);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();

    MPI_File_close(&fh);

    if (rank == 0) {
        std::chrono::duration<double> diff = end_time - start_time;
        double total_mb = (NUM_ITERATIONS * 16.0 * CHUNK_SIZE) / (1024.0 * 1024.0);
        std::cout << "========================================\n";
        std::cout << " MPI-IO Benchmark Complete\n";
        std::cout << "========================================\n";
        std::cout << " Total Data Written : " << total_mb << " MB\n";
        std::cout << " Total Time         : " << diff.count() << " seconds\n";
        std::cout << " Throughput         : " << (total_mb / diff.count()) << " MB/s\n";
        std::cout << "========================================\n";
    }

    MPI_Finalize();
    return 0;
}
