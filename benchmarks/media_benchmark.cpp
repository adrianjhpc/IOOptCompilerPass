#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

// A dummy CRC function. We calculate this before the writes so 
// the compiler doesn't interleave heavy memory reads between our system calls
uint32_t calculate_crc(const char* type, const char* data, uint32_t len) {
    uint32_t crc = 0;
    for(int i = 0; i < 4; ++i) crc ^= type[i];
    if (len > 0) crc ^= data[0];
    if (len > 1) crc ^= data[len - 1];
    return crc;
}

__attribute__((noinline))
void write_chunk(int fd, const char* chunk_type, const char* chunk_data, uint32_t data_len) {
    uint32_t length = data_len;
    uint32_t crc = calculate_crc(chunk_type, chunk_data, data_len);

    // Buffer 1: Stack (length)
    // Buffer 2: Read-Only Data (chunk_type "mdat")
    // Buffer 3: Heap (chunk_data)
    // Buffer 4: Stack (crc)
    write(fd, &length, sizeof(length));
    write(fd, chunk_type, 4);
    write(fd, chunk_data, data_len);
    write(fd, &crc, sizeof(crc));
}

int main() {
    const char* filename = "streaming_output.log";
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Failed to open output file");
        return 1;
    }

    // Simulate a 10KB video frame living on the heap
    uint32_t frame_size = 10240;
    char* frame_data = (char*)malloc(frame_size);
    memset(frame_data, 'V', frame_size);

    // Simulate streaming 50,000 frames (e.g., a short video file)
    int num_frames = 50000;
    for (int i = 0; i < num_frames; ++i) {
        // "mdat" is the standard MP4 chunk type for Media Data
        write_chunk(fd, "mdat", frame_data, frame_size);
    }

    free(frame_data);
    close(fd);
    return 0;
}
