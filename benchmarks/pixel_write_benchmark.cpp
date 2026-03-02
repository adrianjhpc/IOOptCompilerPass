#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <chrono>

const int WIDTH = 4000;
const int HEIGHT = 4000;
const int TOTAL_PIXELS = WIDTH * HEIGHT;
const char* FILENAME = "output.ppm";
const char* EXPECTED_HEADER = "P6\n4000 4000\n255\n";

// -----------------------------------------------------------------------------
// Correctness verification
// -----------------------------------------------------------------------------
bool verify_output() {
    std::cout << "Verifying binary correctness...\n";
    
    std::ifstream file(FILENAME, std::ios::binary);
    if (!file) {
        std::cerr << "[FAIL] Could not open output file for verification.\n";
        return false;
    }

    // Verify header
    size_t header_len = strlen(EXPECTED_HEADER);
    std::vector<char> header_buf(header_len);
    file.read(header_buf.data(), header_len);
    
    if (std::memcmp(header_buf.data(), EXPECTED_HEADER, header_len) != 0) {
        std::cerr << "[FAIL] Header corruption detected.\n";
        return false;
    }

    // Verify pixel data (buffered for speed)
    // We read in chunks of one row (4000 pixels * 3 bytes) at a time
    std::vector<char> row_buffer(WIDTH * 3);
    int pixel_count = 0;

    for (int y = 0; y < HEIGHT; y++) {
        file.read(row_buffer.data(), row_buffer.size());
        
        if (file.gcount() != row_buffer.size()) {
            std::cerr << "[FAIL] File truncated early at row " << y << ".\n";
            return false;
        }

        for (int x = 0; x < WIDTH; x++) {
            // Recalculate the expected color mathematically
            char expected_r = (pixel_count % 255);
            char expected_g = ((pixel_count / WIDTH) % 255);
            char expected_b = 128;

            int buf_idx = x * 3;
            if (row_buffer[buf_idx] != expected_r || 
                row_buffer[buf_idx + 1] != expected_g || 
                row_buffer[buf_idx + 2] != expected_b) {
                
                std::cerr << "[FAIL] Pixel corruption at index " << pixel_count << " (X: " << x << ", Y: " << y << ").\n";
                std::cerr << "Expected: [" << (int)expected_r << ", " << (int)expected_g << ", " << (int)expected_b << "]\n";
                std::cerr << "Got:      [" << (int)row_buffer[buf_idx] << ", " << (int)row_buffer[buf_idx+1] << ", " << (int)row_buffer[buf_idx+2] << "]\n";
                return false;
            }
            pixel_count++;
        }
    }

    // Ensure there is no garbage data appended at the end
    char eof_check;
    if (file.read(&eof_check, 1)) {
        std::cerr << "[FAIL] File contains extra trailing bytes.\n";
        return false;
    }

    std::cout << "[PASS] 100% Data Integrity! (" << pixel_count << " pixels verified natively).\n";
    return true;
}

// -----------------------------------------------------------------------------
// Benchmark
// -----------------------------------------------------------------------------
int main() {
    int fd = open(FILENAME, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    write(fd, EXPECTED_HEADER, strlen(EXPECTED_HEADER));

    std::cout << "Generating 4000x4000 image...\n";
    
    auto start = std::chrono::high_resolution_clock::now();

    char pixel[3];
    
    for (int i = 0; i < TOTAL_PIXELS; i++) {
        pixel[0] = (i % 255);           // Red
        pixel[1] = ((i / WIDTH) % 255); // Green
        pixel[2] = 128;                 // Blue

        // Unoptimized: 3 system calls. Optimized: 1 system call.
        write(fd, pixel, 1);
        write(fd, pixel + 1, 1);
        write(fd, pixel + 2, 1);
    }

    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double> diff = end - start;
    std::cout << "Time taken: " << diff.count() << " seconds\n";

    // Flush and close the file to disk so we can safely verify it
    close(fd);

    // Run the correctness checker
    if (!verify_output()) {
        return 1; // Exit with error if our compiler pass corrupted the logic
    }

    return 0;
}
