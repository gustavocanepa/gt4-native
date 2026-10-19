#!/usr/bin/env python3
"""Differential test: a decomp function compiled for the host against the original MIPS code.

For one function of a gt4-decomp checkout:
  * the original side runs the function's bytes from the game image in a MIPS emulator (Unicorn),
    the whole image mapped (code, .rodata, .data, .bss), so calls and jump tables work;
  * the native side compiles the decomp source for a 32-bit host (the EE `long` spelled long long,
    128-bit registers as a struct; see compile_census.py --guest32), links the decomp sources of
    every function it calls, and runs it with the same image mapped at the same addresses;
  * both get the same trials (4 arguments, some pointing into a shared random memory block) and
    must return the same value and leave the same memory (the block and every changed word of
    the game's .data/.bss).

    diff_test.py --decomp /mnt/e/PROJECTS/GT4 ADDR [ADDR ...] [--trials 300] [--seed 1]
                 [--args p,i,i,i] [--source FILE]
    diff_test.py --decomp ... all [--sample N]

Argument kinds: p pointer into the memory block, i small int, w random 32-bit, c byte, f float.
A function is skipped when something it calls has no decomp source. Needs the decomp's venv
(rabbitizer, via tools/match.py) with unicorn installed, gcc-multilib, and the game image.
"""
import argparse
import os
import shutil
import random
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NATIVE = os.path.dirname(os.path.dirname(HERE))
BASE, SIZE = 0x10000000, 0x4000          # the shared memory block
STACK, STACK_SIZE = 0x7F000000, 0x10000
SENTINEL = 0x7E000000
ERRORS = {}
NARGS = 8         # EABI: ints in a0-a3 then t0-t3, floats in f12-f19; on the host everything is on the stack in order
BITS = 32           # host pointer width of the native side (--m64: the 64-bit identity core, guest memory mapped at its low addresses)
ADAPT = False       # --adapt: run every compiled source through tools/u32_adapt.py (pointer members become 32-bit guest pointers)
WORK = []        # scratch directories of the function being tested, removed before the next one
MAX_CALLEES = 60
HOST_LIBC = {"strlen", "strcmp", "strncmp", "strcpy", "strncpy", "strcat", "strchr", "strrchr", "strstr",
             "memcpy", "memmove", "memset", "memcmp", "memchr", "abs", "labs", "toupper", "tolower"}

sys.path.insert(0, os.path.join(NATIVE, "tools"))
from compile_census import guest_long  # noqa: E402

project = symbols = None
IMAGE = {}       # filled by load_game(): code section, data section, bss range


def load_game(decomp):
    """Import the decomp's own modules and read the game image (once)."""
    global project, symbols
    sys.path.insert(0, os.path.join(decomp, "tools"))
    os.chdir(decomp)
    import project as _p
    import symbols as _s
    import report
    project, symbols = _p, _s
    entry, sections = project.load_image()
    code = next((a, b) for a, b in sections if a <= entry < a + len(b))
    data = next((a, b) for a, b in sections if (a, b) != code)
    fbss, end = report.bss_range()
    hi = (end + 0xFFF) & ~0xFFF
    lo = data[0] & ~0xFFF
    blob = bytearray(hi - lo)
    tail = code[0] + len(code[1]) - lo                      # the code section's last bytes share the data's first page
    if tail > 0:
        blob[:tail] = code[1][len(code[1]) - tail:]
        code = (code[0], code[1][:len(code[1]) - tail])
    blob[data[0] - lo:data[0] - lo + len(data[1])] = data[1]
    IMAGE.update(code=code, data_lo=lo, data=bytes(blob), entry=entry)


def source_for_addr(addr):
    p = project.source_for(addr)
    return str(p) if p else None


def arg_kinds(path, addr):
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(rf"\bfunc_{addr:08X}\s*\(([^)]*)\)", text)
    if not m:
        return ["i"] * NARGS
    params = [p for p in m.group(1).split(",") if p.strip() and p.strip() != "void"]

    def used_as_address(p):
        name = re.findall(r"\w+", p)[-1] if re.findall(r"\w+", p) else ""
        return bool(name) and bool(re.search(rf"\*\s*\(\s*[\w\s]+\*\s*\)\s*\(?\s*{name}\b|\(\s*[\w\s]+\*\s*\)\s*\(?\s*{name}\b|M2C_FIELD\(\s*{name}\b", text))

    kinds = ["p" if "*" in p or used_as_address(p) else ("f" if re.search(r"\b(f32|float)\b", p) else
                                   "c" if re.search(r"\b(char|s8|u8)\b", p) else "i") for p in params]
    return (kinds + ["i"] * NARGS)[:NARGS]


def return_kind(path, addr):
    """('void'|'f32'|'int'|'unsupported', width in bits, C type for the native driver)."""
    text = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(rf"([\w\s\*]+?)\bfunc_{addr:08X}\s*\(", text)
    head = (m.group(1) if m else "").replace('extern "C"', "").strip().split("\n")[-1]
    if re.search(r"\bvoid\b(?!\s*\*)", head):
        return "void", 32, "int"
    body = text[m.end():] if m else ""
    if "{" in body and not re.search(r"\breturn\b", body[body.index("{"):]):
        return "void", 32, "int"      # falls off its end: the original leaves whatever was in v0
    if re.search(r"\b(f32|float)\b(?!\s*\*)", head):
        return "f32", 32, "float"
    if "*" not in head and re.search(r"\b(s16|u16|short)\b", head):
        return "int", 16, "short"
    if "*" not in head and re.search(r"\b(s8|u8|char)\b", head):
        return "int", 8, "char"
    if "*" not in head and re.search(r"\b(f64|double|s64|u64|long long)\b", head):
        return "unsupported", 64, "int"
    return "int", 32, "int"


def make_memory(rng, floaty=False):
    """Random words: some small ints, some pointers into the block, some wild. Small integers read
    as floats are denormals, which the PS2 FPU flushes and Unicorn's generic MIPS FPU does not, so
    float functions get normal floats instead."""
    words = []
    for _ in range(SIZE // 4):
        r = rng.random()
        if r < 0.45:
            words.append(rng.randrange(0, 64) if not floaty else
                         struct.unpack("<I", struct.pack("<f", rng.uniform(-1000, 1000)))[0])
        elif r < 0.85:
            words.append(BASE + rng.randrange(0, SIZE // 4) * 4)
        else:
            words.append(rng.randrange(0, 1 << 32) if not floaty else
                         struct.unpack("<I", struct.pack("<f", rng.uniform(-1e6, 1e6)))[0])
    return struct.pack(f"<{SIZE // 4}I", *words)


def make_args(kinds, rng):
    out = []
    for k in kinds:
        if k == "p":
            out.append(BASE + rng.randrange(0, SIZE // 64) * 16)
        elif k == "c":
            out.append(rng.randrange(0, 256))
        elif k == "f":
            out.append(struct.unpack("<I", struct.pack("<f", rng.choice(
                [0.0, 1.0, -1.0, 0.5, 2.5, 100.0, 3.14159, -7.25, rng.uniform(-1000, 1000)])))[0])
        elif k == "w":
            out.append(rng.randrange(0, 1 << 32))
        else:
            out.append(rng.choice([0, 1, 2, 3, 4, 7, 8, 15, 16, 31, 100, 255, 256, 0x7FFF, 0xFFFF,
                                   rng.randrange(0, 1 << 20)]))
    return out


ULPS = 4


def near_float(a, b):
    """Two 32-bit words that are finite floats within a few ulp of each other. The PS2 FPU rounds
    toward zero and flushes denormals; an x86 rounds to nearest, so float results differ in the last
    bit or two by design: that is an accuracy difference of the platform, not a decomp bug."""
    if a == b:
        return True
    nan = lambda x: (x & 0x7F800000) == 0x7F800000 and (x & 0x7FFFFF)
    if nan(a) and nan(b):
        return True          # the EE has no NaNs: any out-of-domain result is undefined there
    if (a & 0x7F800000) == 0x7F800000 or (b & 0x7F800000) == 0x7F800000:
        return False
    ordered = lambda x: (x ^ 0x7FFFFFFF) if x >> 31 else x
    return abs(ordered(a) - ordered(b)) <= ULPS


def mem_close(a, b):
    if a == b:
        return True
    return all(near_float(*struct.unpack_from("<2I", a[i:i + 4] + b[i:i + 4]))
               for i in range(0, len(a), 4) if a[i:i + 4] != b[i:i + 4])


def same_words(native, orig):
    """Equal dictionaries of changed words, except that a pointer to constant data (a string literal)
    has the host's address on one side and the game's on the other: only that both are such
    pointers is compared."""
    if native.keys() != orig.keys():
        return False
    return all(v == orig[k] or near_float(v, orig[k]) or (0x08000000 <= v < 0x0C000000 and 0x100000 <= orig[k] < 0x6D5DFC)
               for k, v in native.items())


def changed_words(now, was, lo):
    """{address: word} of the words that differ, comparing page by page first."""
    out = {}
    for off in range(0, len(was), 0x1000):
        if now[off:off + 0x1000] != was[off:off + 0x1000]:
            for i in range(off, off + 0x1000, 4):
                if now[i:i + 4] != was[i:i + 4]:
                    out[lo + i] = struct.unpack_from("<I", now, i)[0]
    return out


def run_original(addr, kinds, ret, trials):
    from unicorn import Uc, UC_ARCH_MIPS, UC_MODE_MIPS64, UC_MODE_LITTLE_ENDIAN, UcError
    from unicorn.mips_const import (UC_CPU_MIPS64_5KF, UC_MIPS_REG_A0, UC_MIPS_REG_V0, UC_MIPS_REG_RA,
                                    UC_MIPS_REG_SP, UC_MIPS_REG_F0, UC_MIPS_REG_F12, UC_MIPS_REG_T0)
    code_addr, code = IMAGE["code"]
    lo, data = IMAGE["data_lo"], IMAGE["data"]
    results = []
    for args, mem in trials:
        mu = Uc(UC_ARCH_MIPS, UC_MODE_MIPS64 | UC_MODE_LITTLE_ENDIAN)
        mu.ctl_set_cpu_model(UC_CPU_MIPS64_5KF)
        mu.mem_map(code_addr & ~0xFFF, (len(code) + 0xFFF) & ~0xFFF)
        mu.mem_write(code_addr, code)
        mu.mem_map(lo, len(data))
        mu.mem_write(lo, data)
        mu.mem_map(BASE, SIZE)
        mu.mem_write(BASE, mem)
        mu.mem_map(STACK, STACK_SIZE)
        mu.mem_map(SENTINEL, 0x1000)
        ints = floats = 0
        for k, a in zip(kinds, args):
            if k == "f":
                mu.reg_write(UC_MIPS_REG_F12 + floats, a)
                floats += 1
            else:
                reg = UC_MIPS_REG_A0 + ints if ints < 4 else UC_MIPS_REG_T0 + (ints - 4)
                mu.reg_write(reg, a if a < 0x80000000 else a - (1 << 32) + (1 << 64))
                ints += 1
        mu.reg_write(UC_MIPS_REG_SP, STACK + STACK_SIZE - 0x100)
        mu.reg_write(UC_MIPS_REG_RA, SENTINEL)
        why = ""
        try:
            mu.emu_start(addr, SENTINEL, timeout=2_000_000, count=400000)
            v0 = (mu.reg_read(UC_MIPS_REG_F0) if ret[0] == "f32" else mu.reg_read(UC_MIPS_REG_V0)) & 0xFFFFFFFF
            ok = True
        except UcError as e:
            ok, v0, why = False, 0, str(e)
            ERRORS.setdefault(addr, str(e))
        if ok:
            results.append((True, v0, bytes(mu.mem_read(BASE, SIZE)),
                            changed_words(bytes(mu.mem_read(lo, len(data))), data, lo), ""))
        else:
            results.append((False, 0, b"", {}, why))
    return results


def compile_one(src, addr, work, tag):
    ext = os.path.splitext(src)[1]
    decomp = os.getcwd()
    where = work
    orig_dir = os.path.dirname(src)
    if ADAPT:
        sys.path.insert(0, os.path.join(NATIVE, "tools"))
        import u32_adapt
        where = os.path.join(work, "ad_" + tag)
        ad, _ = u32_adapt.adapt_file(src, where, tag, decomp)
        if ad:
            src, ext = ad, ".cpp"
    text = guest_long(open(src, encoding="utf-8", errors="replace").read())
    text = re.sub(r"^\s*/\*\s*compiler:[^*]*\*/", "", text, count=1)
    srcf = os.path.join(where, f"{tag}{ext}")
    open(srcf, "w", encoding="utf-8").write(text)
    inc = ["-I", os.path.join(NATIVE, "compat", "host32"), "-I", os.path.join(decomp, "include"),
           "-I", os.path.join(decomp, "include", "shim"), "-I", os.path.join(decomp, "include", "stl"),
           "-I", orig_dir]
    obj = os.path.join(work, f"{tag}.o")
    cc = "g++" if ext == ".cpp" else "gcc"
    r = subprocess.run([cc, f"-m{BITS}", "-malign-double", "-msse2", "-mfpmath=sse", "-O1", "-w", "-fpermissive",
                        "-fno-strict-aliasing", "-fPIC", "-c", srcf, "-o", obj] + inc, capture_output=True, text=True,
                       timeout=180)
    if r.returncode:
        return None, "compile: " + next((l for l in r.stderr.splitlines() if "error" in l), r.stderr[:160])
    return obj, None


def undefined_symbols(obj):
    out = subprocess.run(["nm", "-u", obj], capture_output=True, text=True).stdout.split()
    return [u for u in out if u != "U" and not u.startswith(("_GLOBAL_OFFSET", "__stack_chk", "__x86"))]


def defined_functions(obj):
    out = subprocess.run(["nm", "--defined-only", "-g", obj], capture_output=True, text=True).stdout.splitlines()
    return [l.split()[-1] for l in out if len(l.split()) == 3 and l.split()[1] in "TtWw"
            and not l.split()[-1].startswith(("_ZTV", "_ZTI", "_ZTS"))]


def build_native(addr, src, ctype="int"):
    """Compile the function and, transitively, the decomp sources of everything it calls."""
    work = tempfile.mkdtemp(prefix=f"difft_{addr:08x}_")
    WORK.append(work)
    objs, queue, seen, defsyms, aliases = [], [(addr, src)], {addr}, {}, {}
    while queue:
        a, s = queue.pop()
        obj, err = compile_one(s, a, work, f"f_{a:08x}")
        if err:
            return None, err
        objs.append(obj)
        defined = defined_functions(obj)
        want = f"func_{a:08X}"
        if want not in defined and defined:
            real = symbols.name_of(a)
            pick = next((d for d in defined if real and (d == real or real in d)), defined[0] if len(defined) == 1 else None)
            if pick:
                aliases[want] = pick
        for name in undefined_symbols(obj):
            mangled = re.match(r"_Z\d+func_([0-9A-Fa-f]{8})", name)
            target = int(mangled.group(1), 16) if mangled else symbols.address_of(name)
            if target is None:
                return None, f"undefined: {name} (unknown address)"
            if mangled:                           # a C++ caller that did not declare it extern "C"
                aliases[name] = f"func_{target:08X}"
            if target >= 0x617A80:
                defsyms[name] = target            # a game global: it lives where it always did
                continue
            callee = source_for_addr(target)
            libc = symbols.name_of(target)
            if not callee and libc in HOST_LIBC:
                aliases.setdefault(name, libc)    # the C library routine the game linked: use the host's
                continue
            if not callee:
                return None, f"undefined: {name} (no decomp source at {target:#x})"
            if target not in seen:
                if len(seen) > MAX_CALLEES:
                    return None, "undefined: callee closure too large"
                seen.add(target)
                queue.append((target, callee))
    drv = os.path.join(work, "driver.o")
    subprocess.run(["gcc", f"-m{BITS}", "-malign-double", "-msse2", "-O1", "-fPIC", "-c", os.path.join(HERE, "driver.c"), "-o", drv,
                    f"-Dfunc_UNDER_TEST=func_{addr:08X}", f"-DRET_T={ctype}"], check=True)
    exe = os.path.join(work, "run")
    if aliases:
        asm = os.path.join(work, "alias.s")
        with open(asm, "w") as f:
            for n, v in aliases.items():
                f.write(f".globl {n}\n.type {n},@function\n{n}:\n\tjmp {v}{'@PLT' if BITS == 64 else ''}\n")
        subprocess.run(["gcc", f"-m{BITS}", "-c", asm, "-o", asm[:-2] + ".o"], check=True)
        objs.append(asm[:-2] + ".o")
    defsym_flags = [f"-Wl,--defsym,{n}={v:#x}" for n, v in defsyms.items()]
    if BITS == 64 and defsyms:
        # ld --defsym symbols get the load bias added in a PIE (their GOT entries are relocated); symbols defined by the
        # assembler (.set) stay absolute, which is what a guest address is
        asm = os.path.join(work, "abs.s")
        with open(asm, "w") as f:
            for n, v in defsyms.items():
                f.write(f".weak {n}\n.set {n}, {v:#x}\n")
        subprocess.run(["gcc", "-m64", "-c", asm, "-o", asm[:-2] + ".o"], check=True)
        objs.append(asm[:-2] + ".o")
        defsym_flags = []
    r = subprocess.run(["gcc", f"-m{BITS}", "-no-pie" if BITS == 32 else "-pie", *objs, drv, "-o", exe] + defsym_flags,
                       capture_output=True, text=True)
    if r.returncode:
        return None, "link: " + next((l for l in r.stderr.splitlines() if "undefined" in l or "multiple" in l), r.stderr[:160])
    code_addr, code = IMAGE["code"]
    with open(os.path.join(work, "image.bin"), "wb") as f:      # what the driver maps before the trials
        f.write(struct.pack("<II", code_addr, len(code)) + code)
        f.write(struct.pack("<II", IMAGE["data_lo"], len(IMAGE["data"])) + IMAGE["data"])
    return exe, work


def run_native(exe, work, trials):
    inp, out = os.path.join(work, "in.bin"), os.path.join(work, "out.bin")
    with open(inp, "wb") as f:
        f.write(struct.pack("<3I", BASE, SIZE, len(trials)))
        for args, mem in trials:
            f.write(struct.pack("<%dI" % NARGS, *args) + mem)
    r = subprocess.run([exe, inp, out, os.path.join(work, "image.bin")], capture_output=True, text=True, timeout=240)
    if r.returncode:
        raise RuntimeError("native runner: " + r.stderr[:200])
    data = open(out, "rb").read()
    pos, res = 0, []
    for _ in trials:
        st, lo, hi = struct.unpack_from("<3I", data, pos)
        mem = data[pos + 12:pos + 12 + SIZE]
        n = struct.unpack_from("<I", data, pos + 12 + SIZE)[0]
        words = {}
        for j in range(n):
            a, v = struct.unpack_from("<II", data, pos + 16 + SIZE + j * 8)
            words[a] = v
        pos += 16 + SIZE + n * 8
        res.append((bool(st), lo, mem, words))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", required=True)
    ap.add_argument("addrs", nargs="+")
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--args")
    ap.add_argument("--source", help="test this file instead of the decomp's own source (a negative control)")
    ap.add_argument("--sample", type=int, default=0, help="with `all`: a random sample of that many sources")
    ap.add_argument("--from", dest="from_addr", help="with `all`: start at this address (resume)")
    ap.add_argument("--m64", action="store_true", help="native side compiled -m64 (a PIE; the game image is still mapped at its low addresses)")
    ap.add_argument("--adapt", action="store_true", help="with --m64: adapt every compiled source (tools/u32_adapt.py) so pointer members are 32-bit")
    ap.add_argument("--shard", help="with `all`: I/N, the I-th of N interleaved parts (run N processes in parallel)")
    a = ap.parse_args()
    a.decomp = os.path.abspath(a.decomp)
    global BITS, ADAPT
    if a.m64 or a.adapt:
        BITS = 64
    ADAPT = a.adapt
    load_game(a.decomp)
    addrs = a.addrs
    if addrs == ["all"]:
        found = []
        for root, _, names in os.walk(os.path.join(a.decomp, "src")):
            for n in names:
                m = re.fullmatch(r"func_([0-9A-Fa-f]{8})\.(c|cpp)", n)
                if m:
                    found.append(m.group(1).lower())
        found.sort()
        if a.shard:
            i, n = (int(x) for x in a.shard.split("/"))
            found = found[i::n]
        if a.from_addr:
            found = [f for f in found if int(f, 16) >= int(a.from_addr, 16)]
        if a.sample:
            random.Random(7).shuffle(found)
            found = found[:a.sample]
        addrs = found
    totals = {}

    def tally(k):
        totals[k] = totals.get(k, 0) + 1

    for s in addrs:
        for w in WORK:                            # /tmp is memory on some systems: never leave these behind
            shutil.rmtree(w, ignore_errors=True)
        WORK.clear()
        addr = int(s, 16)
        src = a.source or source_for_addr(addr)
        if not src:
            print(f"{addr:08X}: no source", flush=True)
            tally("no source")
            continue
        ret = return_kind(src, addr)
        if ret[0] == "unsupported":
            print(f"{addr:08X}: skipped (64-bit or double return)", flush=True)
            tally("skipped: wide return")
            continue
        try:
            exe, work = build_native(addr, src, ret[2])
        except subprocess.TimeoutExpired:
            exe, work = None, "compile: timed out"
        if not exe:
            print(f"{addr:08X}: skipped ({work})", flush=True)
            tally("skipped: " + work.split(":")[0])
            continue
        kinds = (a.args.split(",") + ["i"] * NARGS)[:NARGS] if a.args else arg_kinds(src, addr)
        rng = random.Random(a.seed)
        src_text = open(src, encoding="utf-8", errors="replace").read()
        floaty = ret[0] == "f32" or "f" in kinds or bool(re.search(r"\b(f32|float)\b", src_text))
        trials = [(make_args(kinds, rng), make_memory(rng, floaty)) for _ in range(a.trials)]
        try:
            nat = run_native(exe, work, trials)
            org = run_original(addr, kinds, ret, trials)
        except Exception as e:                       # one function must not end a long run
            print(f"{addr:08X}: harness error ({type(e).__name__}: {str(e)[:80]})", flush=True)
            tally("harness error")
            continue
        check_ret = ret[0] != "void"
        mask = (1 << ret[1]) - 1

        def canon(v):
            return 0x7FC00000 if ret[0] == "f32" and (v & 0x7F800000) == 0x7F800000 and (v & 0x7FFFFF) else v

        same = diff = both_fault = 0
        first = None
        for i, ((nok, nret, nmem, nwords), (ook, oret, omem, owords, owhy)) in enumerate(zip(nat, org)):
            nret, oret = canon(nret), canon(oret)
            if (not nok and not ook) or "UNALIGNED" in owhy:
                both_fault += 1
            elif nok and ook and ((nret & mask) == (oret & mask) or not check_ret or (ret[0] == "f32" and near_float(nret, oret))) and mem_close(nmem, omem) and same_words(nwords, owords):
                same += 1
            else:
                diff += 1
                if first is None:
                    why = ("native faulted" if not nok else "original faulted" if not ook else
                           "return value" if ((nret & mask) != (oret & mask) and check_ret and not (ret[0] == "f32" and near_float(nret, oret))) else
                           "memory block" if not mem_close(nmem, omem) else "game data")
                    first = (i, why, trials[i][0], nret, oret)
        oracle_stuck = "EXCEPTION" in ERRORS.get(addr, "") and all(not o[0] and "UNALIGNED" not in o[4] for o in org)
        verdict = ("ORACLE-UNSUPPORTED" if oracle_stuck else
                   "AGREE" if diff == 0 and same else "DIFFER" if diff else "no usable trial")
        line = f"{addr:08X} {kinds}: {verdict}  same={same} differ={diff} both-fault={both_fault}"
        if first:
            line += (f"  [original: {ERRORS[addr]}]" if addr in ERRORS else "") + \
                f"  first: trial {first[0]} {first[1]} args={[hex(x) for x in first[2]]} native={first[3]:#x} orig={first[4]:#x}"
        print(line, flush=True)
        tally(verdict)
    for w in WORK:
        shutil.rmtree(w, ignore_errors=True)
    print("SUMMARY", totals)


if __name__ == "__main__":
    main()
