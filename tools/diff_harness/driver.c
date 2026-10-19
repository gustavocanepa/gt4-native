/* Native side of the differential harness: runs one decomp function (linked in as func_ADDR) on
 * trials read from a file and writes what it returned and what it left in guest memory.
 *
 * 32-bit process. The game image is mapped at the addresses the original uses (code and .rodata
 * read only in practice, .data/.bss writable), and so is a shared random block, so raw pointers
 * in the decomp sources mean what they meant on the PS2. One fork per trial: a fault is a result.
 *
 * argv: trials.in results.out image.bin
 * trials.in : u32 base, u32 size, u32 ntrials, then per trial u32 args[4] + size bytes of the block
 * image.bin : two regions, each u32 address, u32 length, bytes: the code section, the data block
 * results.out, per trial: u32 status (1 ran, 0 faulted), u32 ret_lo, u32 ret_hi, the block (size
 *             bytes), u32 n, then n pairs (u32 address, u32 word) of the words of the data
 *             block the run changed
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdint.h>
#include <xmmintrin.h>

/* RET_T is the function's return type (-DRET_T=float for f32); default: a 32-bit integer. */
#ifndef RET_T
#define RET_T int
#endif
extern RET_T func_UNDER_TEST(unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                             unsigned a4, unsigned a5, unsigned a6, unsigned a7);

static uint32_t bits_of(RET_T v)
{
    uint32_t b = 0;
    memcpy(&b, &v, sizeof v < 4 ? sizeof v : 4);
    return b;
}

static void *map_at(uint32_t base, uint32_t size)
{
    void *m = mmap((void *)(uintptr_t)base, size, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
    if (m != (void *)(uintptr_t)base) { fprintf(stderr, "cannot map guest memory at %#x\n", base); exit(3); }
    return m;
}

int main(int argc, char **argv)
{
    if (argc < 4) return 2;
    /* The PS2 FPU has no denormals: flush them (FTZ and DAZ) so SSE arithmetic agrees. */
    _mm_setcsr(_mm_getcsr() | 0x8040);
    FILE *in = fopen(argv[1], "rb"), *out = fopen(argv[2], "wb"), *img = fopen(argv[3], "rb");
    uint32_t base, size, n;
    if (!in || !out || !img || fread(&base, 4, 1, in) != 1 || fread(&size, 4, 1, in) != 1 || fread(&n, 4, 1, in) != 1) return 2;
    uint8_t *data = NULL, *pristine = NULL;
    uint32_t data_lo = 0, data_len = 0;
    for (int region = 0; region < 2; region++) {
        uint32_t lo, len;
        if (fread(&lo, 4, 1, img) != 1 || fread(&len, 4, 1, img) != 1) return 2;
        uint32_t span = (len + 0xFFF) & ~0xFFFu;
        uint8_t *m = map_at(lo & ~0xFFFu, span + (lo & 0xFFF ? 0x1000 : 0));
        if (fread(m + (lo & 0xFFF), 1, len, img) != len) return 2;
        if (region == 1) {
            data = m; data_lo = lo; data_len = len;
            pristine = malloc(len);
            memcpy(pristine, data, len);
        }
    }
    void *mem = map_at(base, size);
    uint32_t args[8];
    uint8_t *buf = malloc(size);
    for (uint32_t t = 0; t < n; t++) {
        if (fread(args, 4, 8, in) != 8 || fread(buf, 1, size, in) != size) return 2;
        memcpy(mem, buf, size);
        int fd[2];
        if (pipe(fd)) return 2;
        fflush(out);
        pid_t pid = fork();
        if (pid == 0) {
            close(fd[0]);
            alarm(2);
            RET_T r = func_UNDER_TEST(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]);
            uint32_t hdr[3] = {1, bits_of(r), 0};
            FILE *w = fdopen(fd[1], "wb");
            fwrite(hdr, 4, 3, w);
            fwrite(mem, 1, size, w);
            uint32_t count = 0;
            for (uint32_t i = 0; i + 4 <= data_len; i += 4)
                if (memcmp(data + i, pristine + i, 4)) count++;
            fwrite(&count, 4, 1, w);
            for (uint32_t i = 0; i + 4 <= data_len; i += 4)
                if (memcmp(data + i, pristine + i, 4)) {
                    uint32_t pair[2] = {data_lo + i, *(uint32_t *)(data + i)};
                    fwrite(pair, 4, 2, w);
                }
            fclose(w);
            _exit(0);
        }
        close(fd[1]);
        size_t cap = 1 << 16, got = 0;
        uint8_t *res = malloc(cap);
        ssize_t r;
        while ((r = read(fd[0], res + got, cap - got)) > 0) {
            got += r;
            if (got == cap) { cap *= 2; res = realloc(res, cap); }
        }
        close(fd[0]);
        int st;
        waitpid(pid, &st, 0);
        if (WIFEXITED(st) && WEXITSTATUS(st) == 0 && got >= 16 + (size_t)size) fwrite(res, 1, got, out);
        else {
            uint32_t hdr[3] = {0, 0, 0}, zero = 0;
            fwrite(hdr, 4, 3, out);
            memset(res, 0, size);
            fwrite(res, 1, size, out);
            fwrite(&zero, 4, 1, out);
        }
        free(res);
    }
    fclose(out);
    return 0;
}
