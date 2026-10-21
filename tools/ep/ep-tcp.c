// raw TCP ping-pong of a payload to N echo servers in parallel (EP gate, step 1b)
// server: ep-tcp -S port [-spin]     client: ep-tcp -n iters [-b bytes] [-spin] host:port...
#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

static int spin = 0;
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
static void rd(int fd, char * b, size_t n) {
    size_t g = 0;
    while (g < n) {
        ssize_t r = recv(fd, b + g, n - g, spin ? MSG_DONTWAIT : 0);
        if (r < 0 && (errno == EAGAIN || errno == EINTR)) continue;
        if (r <= 0) exit(0);
        g += r;
    }
}
static void wr(int fd, const char * b, size_t n) {
    size_t g = 0;
    while (g < n) { ssize_t r = send(fd, b + g, n - g, 0); if (r <= 0) exit(1); g += r; }
}
static int cmp(const void * a, const void * b) { double x = *(double *)a, y = *(double *)b; return x < y ? -1 : x > y; }

int main(int argc, char ** argv) {
    int iters = 5000, port = 0, server = 0; size_t bytes = 20480 + 48; int nc = 0; char * eps[16];
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-S")) { server = 1; port = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "-n")) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-b")) bytes = atol(argv[++i]);
        else if (!strcmp(argv[i], "-spin")) spin = 1;
        else eps[nc++] = argv[i];
    }
    char * buf = malloc(bytes + 64); memset(buf, 1, bytes);
    int one = 1;
    if (server) {
        int ls = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(port), .sin_addr.s_addr = INADDR_ANY };
        if (bind(ls, (void *)&a, sizeof a) || listen(ls, 4)) { perror("bind"); return 1; }
        for (;;) {
            int fd = accept(ls, 0, 0);
            if (fork() == 0) {
                setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
                for (;;) { rd(fd, buf, bytes); wr(fd, buf, bytes); }
            }
            close(fd);
        }
    }
    int fds[16];
    for (int i = 0; i < nc; i++) {
        char * c = strchr(eps[i], ':'); *c = 0;
        struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(atoi(c + 1)) };
        inet_pton(AF_INET, eps[i], &a.sin_addr);
        fds[i] = socket(AF_INET, SOCK_STREAM, 0);
        setsockopt(fds[i], IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        if (connect(fds[i], (void *)&a, sizeof a)) { perror("connect"); return 1; }
    }
    double * t = malloc(sizeof(double) * iters);
    for (int it = -300; it < iters; it++) {
        double t0 = now_us();
        for (int i = 0; i < nc; i++) wr(fds[i], buf, bytes);
        if (spin) {  // poll all sockets round-robin so replies are taken as they arrive
            size_t got[16] = {0}; int left = nc;
            while (left) for (int i = 0; i < nc; i++) if (got[i] < bytes) {
                ssize_t r = recv(fds[i], buf, bytes - got[i], MSG_DONTWAIT);
                if (r > 0) { got[i] += r; if (got[i] == bytes) left--; }
            }
        } else for (int i = 0; i < nc; i++) rd(fds[i], buf, bytes);
        if (it >= 0) t[it] = now_us() - t0;
    }
    qsort(t, iters, sizeof(double), cmp);
    double s = 0; for (int i = 0; i < iters; i++) s += t[i];
    printf("tcp peers=%d bytes=%zu %s: min %.0f median %.0f mean %.0f p90 %.0f p99 %.0f max %.0f us | x40 (median) %.2f ms\n", nc, bytes, spin ? "spin" : "block",
           t[0], t[iters / 2], s / iters, t[(int)(iters * 0.9)], t[(int)(iters * 0.99)], t[iters - 1], t[iters / 2] * 40 / 1000);
    return 0;
}
