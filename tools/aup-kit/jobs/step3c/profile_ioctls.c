#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/ioctl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "amdxdna_accel.h"

static FILE *out;
static int xdna_fds[8], xdna_fd_count;
static int is_xdna(int fd) { for (int i = 0; i < xdna_fd_count; ++i) if (xdna_fds[i] == fd) return 1; return 0; }
static int (*next_ioctl)(int, unsigned long, ...);
static uint64_t now_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec * 1000000000ULL + t.tv_nsec;
}
__attribute__((constructor)) static void setup(void) {
    next_ioctl = dlsym(RTLD_NEXT, "ioctl");
    const char *path = getenv("HETEROMEGA_IOCTL_LOG");
    if (path) { char name[4096]; snprintf(name, sizeof name, "%s.%d.csv", path, (int)getpid()); out = fopen(name, "w"); }
    if (out) fprintf(out, "start_ns,end_ns,tid,operation,ret,errno,ctx,seq\n");
}
int ioctl(int fd, unsigned long req, ...) {
    va_list ap; va_start(ap, req); void *arg = va_arg(ap, void *); va_end(ap);
    if (!next_ioctl) next_ioctl = dlsym(RTLD_NEXT, "ioctl");
    const char *op = NULL;
    if (req == DRM_IOCTL_AMDXDNA_CREATE_CTX) op = "context";
    else if (is_xdna(fd)) {
        if (req == DRM_IOCTL_AMDXDNA_EXEC_CMD) op = "submit";
        else if (req == DRM_IOCTL_AMDXDNA_WAIT_CMD) op = "legacy_wait";
        else if (req == DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT) op = "timeline_wait";
        else if (req == DRM_IOCTL_AMDXDNA_CREATE_BO) op = "create_bo";
        else if (req == DRM_IOCTL_AMDXDNA_GET_BO_INFO) op = "bo_info";
        else if (req == DRM_IOCTL_GEM_CLOSE) op = "close_bo";
    }
    uint64_t t0 = out && op ? now_ns() : 0;
    int ret = next_ioctl(fd, req, arg); int e = ret < 0 ? errno : 0;
    uint64_t t1 = t0 ? now_ns() : 0;
    if (req == DRM_IOCTL_AMDXDNA_CREATE_CTX && ret == 0 && !is_xdna(fd) && xdna_fd_count < 8) xdna_fds[xdna_fd_count++] = fd;
    if (t0) {
        unsigned ctx = 0; unsigned long long seq = 0;
        if (req == DRM_IOCTL_AMDXDNA_EXEC_CMD) {
            struct amdxdna_drm_exec_cmd *a = arg; ctx = a->ctx; seq = a->seq;
        }
        fprintf(out, "%llu,%llu,%ld,%s,%d,%d,%u,%llu\n",
                (unsigned long long)t0, (unsigned long long)t1,
                syscall(SYS_gettid), op, ret, e, ctx, seq);
    }
    if (ret < 0) errno = e;
    return ret;
}
__attribute__((destructor)) static void finish(void) { if (out) fclose(out); }
