#include <unistd.h>

// TEST 1: The Happy Path
// The buffer is untouched between read and write. 
// EXPECTATION: read and write are erased, sendfile is inserted.
void proxy_clean(int fd_in, int fd_out, size_t size) {
    char buffer[4096];
    read(fd_in, buffer, size);
    write(fd_out, buffer, size);
}

// TEST 2: The Hazard Path
// The buffer is mutated before being written.
// EXPECTATION: The compiler aborts the optimization to prevent data corruption. 
// read and write must remain intact.
void proxy_mutated(int fd_in, int fd_out, size_t size) {
    char buffer[4096];
    read(fd_in, buffer, size);
    
    // MEMORY HAZARD: We alter the data!
    buffer[0] = 'X'; 
    
    write(fd_out, buffer, size);
}

// TEST 3: Size Mismatch
// Reading 4096 bytes, but only writing 2048. 
// EXPECTATION: Must abort. sendfile cannot handle partial buffer writes cleanly here.
void proxy_size_mismatch(int fd_in, int fd_out) {
    char buffer[4096];
    read(fd_in, buffer, 4096);
    write(fd_out, buffer, 2048); 
}

// TEST 4: Opaque Call Hazard
// An external function is called between the read and write.
// EXPECTATION: Must abort. 'dummy_log' might have access to the buffer pointer globally, 
// or it might mutate system state. We assume the worst.
extern void dummy_log(void); 

void proxy_opaque_call(int fd_in, int fd_out, size_t size) {
    char buffer[4096];
    read(fd_in, buffer, size);
    dummy_log(); 
    write(fd_out, buffer, size);
}
