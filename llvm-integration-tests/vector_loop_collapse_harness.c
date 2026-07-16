// End-to-end execution harness for IOOpt loop-collapse transforms.
//
// Each case runs the SAME function in a baseline (plugin off) and an optimized
// (plugin on) build; the harness diffs their output files. Identical output +
// a confirmed "fired" transform == the optimization is semantically correct.
//
//   ./exe A N OUT       dynamic-N contiguous pwrite  -> one pwrite(N*BLK)
//   ./exe B N OUT IN    dynamic-N contiguous pread   -> one pread(N*BLK)
//   ./exe D OUT         const-N  scattered pwrite     -> one pwritev
//   (N==0 is exercised by calling case A with N=0.)

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>

#define BLK 4096
#define DN  64            /* Needs a constant trip count */

static void fill(char *b, size_t n) {           /* deterministic, reproducible */
    unsigned x = 0x12345678u;
    for (size_t i = 0; i < n; i++) { x = x * 1103515245u + 12345u; b[i] = (char)(x >> 16); }
}

// Dynamic-N: contiguous buffer (step==BLK) and contiguous file offset
// (step==BLK). Return unused -> collapses to one pwrite(buf, N*BLK, 0).
__attribute__((noinline))
static void caseA_pwrite(int fd, char *buf, int n) {
    for (int i = 0; i < n; i++)
        pwrite(fd, buf + (long)i * BLK, BLK, (off_t)i * BLK);
}

// Dynamic-N: contiguous pread -> one pread(buf, N*BLK, 0).
__attribute__((noinline))
static void caseB_pread(int fd, char *buf, int n) {
    for (int i = 0; i < n; i++)
        pread(fd, buf + (long)i * BLK, BLK, (off_t)i * BLK);
}

// Const-N: scattered buffers (stride 2*BLK > count BLK), contiguous file
// offsets -> one pwritev of DN iovecs.
__attribute__((noinline))
static void caseD_scatter(int fd, char *buf) {
    for (int i = 0; i < DN; i++)
        pwrite(fd, buf + (long)i * 2 * BLK, BLK, (off_t)i * BLK);
}

// Tier / CFR-loop: copy IN -> OUT via a read/write loop.
// -fno-inline keeps it intact so the copy_file_range loop promotion can fire.
__attribute__((noinline))
static void caseE_copy(int src, int dst) {
    char buf[65536];
    ssize_t n;
    while ((n = read(src, buf, sizeof buf)) > 0)
        write(dst, buf, n);
}


static int write_all(int fd, const char *b, size_t n) {   /* for the B read-back */
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, b + off, n - off);
        if (w <= 0) return -1;
        off += (size_t)w;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s A|B|D ...\n", argv[0]); return 2; }
    const char mode = argv[1][0];

    if (mode == 'A') {                       // A N OUT
        int n = atoi(argv[2]); const char *out = argv[3];
        size_t sz = (size_t)(n > 0 ? n : 0) * BLK;
        char *buf = malloc(sz ? sz : 1); fill(buf, sz);
        int fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return 3;
        caseA_pwrite(fd, buf, n);
        close(fd); free(buf); return 0;
    }
    if (mode == 'B') {                       // B N OUT IN
        int n = atoi(argv[2]); const char *out = argv[3], *in = argv[4];
        size_t sz = (size_t)n * BLK;
        char *buf = malloc(sz ? sz : 1); memset(buf, 0, sz ? sz : 1);
        int fdi = open(in, O_RDONLY); if (fdi < 0) return 4;
        caseB_pread(fdi, buf, n); close(fdi);
        int fdo = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fdo < 0) return 3;
        int rc = write_all(fdo, buf, sz); close(fdo); free(buf);
        return rc ? 5 : 0;
    }
    if (mode == 'D') {                       // D OUT
        const char *out = argv[2];
        size_t sz = (size_t)DN * 2 * BLK;
        char *buf = malloc(sz); fill(buf, sz);
        int fd = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644); if (fd < 0) return 3;
        caseD_scatter(fd, buf);
        close(fd); free(buf); return 0;
    }
    if (mode == 'E') {                       // E OUT IN
        const char *out = argv[2], *in = argv[3];
        int fdi = open(in, O_RDONLY);
        if (fdi < 0) return 4;
        int fdo = open(out, O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fdo < 0) { close(fdi); return 3; }
        caseE_copy(fdi, fdo);
        fsync(fdo);                          // force durability before hashing
        close(fdo);
        close(fdi);
        return 0;
    }
    fprintf(stderr, "unknown mode %c\n", mode); return 2;
}

