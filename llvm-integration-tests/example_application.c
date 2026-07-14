#include <stddef.h>

extern void my_fast_write(int fd, const char *buf, size_t len);

void process_data(int fd) {
    char header[10] = "HEADER";
    char body[10]   = "BODY";

    // 4 writes guarantees we trip the BatchThreshold (4) for writev!
    my_fast_write(fd, header, 10);
    my_fast_write(fd, body, 10);
    my_fast_write(fd, header, 10);
    my_fast_write(fd, body, 10);
}
