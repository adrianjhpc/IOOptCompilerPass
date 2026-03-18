#include <sys/uio.h>
#include <unistd.h>
#include <stddef.h>

// Intrinsic for merging 2 sequential writes
ssize_t ioopt_writev_2(int fd, void *b1, size_t l1, void *b2, size_t l2) {
    struct iovec iov[2] = {{b1, l1}, {b2, l2}};
    return writev(fd, iov, 2);
}

// Intrinsic for merging 3 sequential writes
ssize_t ioopt_writev_3(int fd, void *b1, size_t l1, void *b2, size_t l2, void *b3, size_t l3) {
    struct iovec iov[3] = {{b1, l1}, {b2, l2}, {b3, l3}};
    return writev(fd, iov, 3);
}

// Intrinsic for merging 4 sequential writes
ssize_t ioopt_writev_4(int fd, void *b1, size_t l1, void *b2, size_t l2, void *b3, size_t l3, void *b4, size_t l4) {
    struct iovec iov[4] = {{b1, l1}, {b2, l2}, {b3, l3}, {b4, l4}};
    return writev(fd, iov, 4);
}
