#include <unistd.h>
#include <stddef.h>

void my_fast_write(int fd, const char *buf, size_t len) {
    write(fd, buf, len);
}
