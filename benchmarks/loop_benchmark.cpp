#include <cstdlib>
#include <unistd.h>
#include <fcntl.h>
#include <cstdio>

__attribute__((noinline))
static void dump(int fd, char *buf, long blk, int n) {
    for (int i = 0; i < n; i++)
        pwrite(fd, buf + (long)i * blk, blk, (off_t)i * blk);
}

int main(int argc, char **argv) {       
    long blk = 65536;
    int  n   = 1024;
    int  reps= 1;
    char *buf = (char *)std::malloc((size_t)n * blk);
    const char *filename = "output.dat";
    if (!buf) return 1;
    for (size_t i = 0; i < (size_t)n * blk; i++) buf[i] = (char)i;
    for (int r = 0; r < reps; r++) {
        int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return 3;
        dump(fd, buf, blk, n);
        fsync(fd);
        close(fd);
    }
    std::free(buf);
    return 0;
}

