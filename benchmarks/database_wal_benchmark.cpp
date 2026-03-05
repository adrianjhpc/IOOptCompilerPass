#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// Database WAL Header (24 bytes total)
struct LogHeader {
    uint64_t timestamp;
    uint64_t tx_id;
    uint32_t payload_size;
    uint32_t checksum;
};

__attribute__((noinline))
void write_wal_record(int fd, uint64_t tx_id, const char* row_data, uint32_t size) {
    LogHeader hdr;
    hdr.timestamp = 1700000000 + tx_id; // Simulated timestamp
    hdr.tx_id = tx_id;
    hdr.payload_size = size;
    hdr.checksum = 0xDEADBEEF;          // Simulated checksum
    
    // Add a simulated trailing footer/checksum block
    uint32_t block_footer = 0xCAFEBABE;

    // The torture test: 3 scattered writes.
    // The pass will now see this is >= 3 and vectorize it!
    write(fd, &hdr, sizeof(LogHeader));
    write(fd, row_data, size);
    write(fd, &block_footer, sizeof(uint32_t)); 
}

int main() {
    const char* filename = "wal_output.log";
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Failed to open output file");
        return 1;
    }

    // Simulate a 512-byte database row
    uint32_t row_size = 512;
    char* row_data = (char*)malloc(row_size);
    memset(row_data, 'D', row_size - 1);
    row_data[row_size - 1] = '\n'; 

    // Simulate 100,000 database transactions
    int num_transactions = 100000;
    for (int i = 0; i < num_transactions; ++i) {
        write_wal_record(fd, i, row_data, row_size);
    }

    free(row_data);
    close(fd);
    return 0;
}
