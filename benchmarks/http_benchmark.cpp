#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

// We use noinline so -O3 doesn't put this into main,
// allowing the pass to easily find the distinct write calls.
__attribute__((noinline))
void handle_request(int fd, const char* payload, size_t payload_len) {
    // Stack memory (Dynamic Header)
    char header[256];
    int header_len = snprintf(header, sizeof(header),
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/html\r\n"
        "Content-Length: %zu\r\n"
        "\r\n", payload_len);

    // Read-Only memory (Static Footer)
    const char* footer = "\r\n--End-of-Response--\r\n";
    size_t footer_len = strlen(footer);

    // The Compiler Torture Test: 4 writes, dynamic sizes (N>=4 threshold)
    write(fd, header, header_len);
    write(fd, payload, payload_len);
    write(fd, footer, footer_len);
    write(fd, "\n", 1); 
}

int main() {
    // We write to a real file to verify the data integrity later
    const char* filename = "http_output.log";
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("Failed to open output file");
        return 1;
    }

    // Generate a dummy payload on the Heap
    size_t payload_size = 1024; // 1KB JSON/HTML payload
    char* payload = (char*)malloc(payload_size);
    memset(payload, 'A', payload_size - 1);
    payload[payload_size - 1] = '\n'; // Add a newline at the end

    // Simulate 100,000 incoming web requests
    int num_requests = 100000;
    for (int i = 0; i < num_requests; ++i) {
        handle_request(fd, payload, payload_size);
    }

    free(payload);
    close(fd);
    return 0;
}
