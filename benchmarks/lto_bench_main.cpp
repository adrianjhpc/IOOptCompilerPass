#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// Forward declaration of the external library function
void write_log_payload(int fd, const char* message);

__attribute__((noinline))
void process_request(int fd) {
    const char* header = "[SYS-LOG] Request Processed: ";

    // Write 1: The header (Happens here in main)
    write(fd, header, 29);

    // Write 2 & 3: The payload (Happens in the external library)
    write_log_payload(fd, "User successfully authenticated and session created.");

    // Write 4: The final footer (Triggers N>=4 batching post-LTO inlining)
    write(fd, " [DONE]\n", 8);
}

int main() {
    const char* filename = "lto_benchmark.log";
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Failed to open output file");
        return 1;
    }

    // Simulate 100,000 incoming web requests
    int num_requests = 100000;
    for (int i = 0; i < num_requests; ++i) {
        process_request(fd);
    }

    close(fd);
    return 0;
}
