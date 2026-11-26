#!/usr/bin/env python3
"""Lists the title's 256-bit vector stores for `VICTORY_PATCH_WIDE`.

Rosetta 2 stops ("unexpectedly need to EmulateForward on a synchronous
exception") when a 32-byte store begins in writable memory and faults in the
next page, which fault tracking makes happen (docs/macos.md, "Masked stores
into protected pages hang Rosetta"). 128-bit stores, `vextractf128` to memory
and smaller stores fault and resume. `victory-loader/src/wide.rs` rewrites
each listed store as a jump to a trampoline that stores the two 128-bit
halves. This script finds the stores with `llvm-objdump` and writes the list;
nothing of the title is in it but offsets and lengths.

    tools/macos/wide-stores.py <cached title image> > wide-sites.txt

The image is the mapped copy VIctory writes to its cache
(`<cache_dir>/images/victory-image-<hash>.dll`), whose bytes are the title's.

A store of 5 bytes or more becomes a 5-byte jump in place. A 4-byte store has
no room for one: its run takes in the instructions after it (or before it)
that can be moved, until 5 bytes are covered, and only where no branch lands
inside the run. A line: RVA (hex), bytes replaced, then the pieces of the run:
`sN` a 256-bit store, `cN` bytes copied, `rN:K:T` bytes copied with their
RIP-relative displacement at K redone for RVA T, `kN:CC:T` a conditional jump
to T, `jN:T` a jump to T that ends the run.
"""

import collections
import re
import shutil
import struct
import subprocess
import sys

STORE = re.compile(r"^v\w+\s+ymmword ptr \[[^]]*\], ymm\d+$")
MASKED = re.compile(r"^v(p?maskmov)")
# What cannot be moved out of place (or is patched by `VICTORY_PATCH_MASKED`).
FIXED = re.compile(
    r"^(vextract|vcvtps2ph|vpmaskmov|vmaskmov|call|ret|loop|bnd|int3|int\b|syscall|xbegin|xend|"
    r"push|pop|enter|leave|rep|lock|j\w+)"
)
BRANCH = re.compile(r"^(j\w+|call|loop\w*|xbegin)\s+0x([0-9a-f]+)")
COND = re.compile(r"^(j(?!mp)\w+)\s+0x([0-9a-f]+)")
JUMP = re.compile(r"^jmp\s+0x([0-9a-f]+)")
LINE = re.compile(r"^\s*([0-9a-f]+):\s+((?:[0-9a-f]{2} )+)\s*\t(\S+)(?:\t(.*))?$")


def sections(data):
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    count = struct.unpack_from("<H", data, pe + 6)[0]
    optional = struct.unpack_from("<H", data, pe + 20)[0]
    base = struct.unpack_from("<Q", data, pe + 24 + 24)[0]
    found = {}
    for i in range(count):
        at = pe + 24 + optional + 40 * i
        name = data[at : at + 8].rstrip(b"\0").decode()
        vsize, va, rsize, raw = struct.unpack_from("<IIII", data, at + 8)
        found[name] = (va, vsize, raw, rsize)
    return base, found


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    image = sys.argv[1]
    objdump = shutil.which("llvm-objdump") or "/opt/homebrew/opt/llvm/bin/llvm-objdump"
    data = open(image, "rb").read()
    base, secs = sections(data)
    tva, tsize, traw, _ = secs[".text"]

    def code(rva, n):
        return data[traw + rva - tva : traw + rva - tva + n]

    # Where branches land: direct targets, function starts and ends, and
    # anything in the data that looks like an address in the code (switch tables).
    targets = set()
    va, vsize, raw, rsize = secs[".pdata"]
    for k in range(min(vsize, rsize) // 12):
        begin, end, _ = struct.unpack_from("<III", data, raw + 12 * k)
        targets.update((begin, end))
    for name in (".rdata", ".data"):
        va, vsize, raw, rsize = secs[name]
        for (value,) in struct.iter_unpack("<I", data[raw : raw + rsize - rsize % 4]):
            if tva <= value < tva + tsize:
                targets.add(value)

    # One pass: every 256-bit store, with the three instructions before it and
    # the four after.
    stores = []  # [rva, length, text, before, after]
    recent = collections.deque(maxlen=3)
    waiting = []
    proc = subprocess.Popen(
        [objdump, "-d", "--x86-asm-syntax=intel", "-j", ".text", image],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    for line in proc.stdout:
        m = LINE.match(line)
        if not m:
            if not line.strip() or line.endswith(":\n"):
                recent.clear()
            continue
        rva = int(m.group(1), 16) - base
        n = len(m.group(2).split())
        text = re.sub(r"\s+", " ", f"{m.group(3)} {m.group(4) or ''}").strip()
        b = BRANCH.match(text)
        if b:
            targets.add(int(b.group(2), 16) - base)
        insn = (rva, n, text)
        for w in waiting:
            w[4].append(insn)
        waiting = [w for w in waiting if len(w[4]) < 4]
        if STORE.match(text.split(" #")[0]):
            entry = [rva, n, text, list(recent)[::-1], []]
            stores.append(entry)
            waiting.append(entry)
        recent.append(insn)
    proc.wait()

    def piece(a, m, tt):
        t = tt.split(" #")[0]
        if STORE.match(t):
            return None if ("rip" in t or MASKED.match(t)) else f"s{m}"
        c = COND.match(t)
        if c:
            b = code(a, m)
            cc = (b[0] if m == 2 else b[1]) & 0xF
            return f"k{m}:{cc:x}:{int(c.group(2), 16) - base:x}"
        if "rip" in t:
            if FIXED.match(t):
                return None
            target = re.search(r"# 0x([0-9a-f]+)", tt)
            if not target:
                return None
            target = int(target.group(1), 16) - base
            b = code(a, m)
            for k in range(1, m - 3):
                if struct.unpack_from("<i", b, k)[0] == target - (a + m):
                    return f"r{m}:{k}:{target:x}"
            return None
        if FIXED.match(t) or not t:
            return None
        return f"c{m}"

    sites = {}
    stats = collections.Counter()
    short = []
    for rva, n, text, before, after in stores:
        t = text.split(" #")[0]
        if MASKED.match(t):
            continue
        if n >= 5:
            sites.setdefault(rva, (n, [f"s{n}"]))
            continue
        if "rip" in t:
            stats["4-byte RIP-relative, left"] += 1
            continue
        short.append((rva, before, after))
    # 4-byte stores: the instructions after first, then those before.
    left = []
    taken = set()  # stores already inside a run made for an earlier one
    for rva, before, after in short:
        if rva in taken:
            continue
        pieces, cur = ["s4"], rva + 4
        for a, m, tt in after:
            if a != cur or a in targets:
                break
            j = JUMP.match(tt.split(" #")[0])
            if j:
                pieces.append(f"j{m}:{int(j.group(1), 16) - base:x}")
                cur += m
                break
            p = piece(a, m, tt)
            if p is None:
                break
            pieces.append(p)
            cur += m
            if cur - rva >= 5:
                break
        if cur - rva >= 5:
            for k in range(rva + 1, cur):
                sites.pop(k, None)
                taken.add(k)
            sites[rva] = (cur - rva, pieces)
        else:
            left.append((rva, before))
    def free(lo, hi):
        return not any(r < hi and r + t > lo for r, (t, _) in sites.items() if r < hi and r > lo - 64)
    ends = {r + t: r for r, (t, p) in sites.items() if not any(x.startswith("j") for x in p)}
    unpatched = []
    for rva, before in left:
        # A site that ends where this store begins takes it in.
        if rva not in targets and rva in ends:
            r0 = ends.pop(rva)
            t0, p0 = sites[r0]
            sites[r0] = (t0 + 4, p0 + ["s4"])
            ends[rva + 4] = r0
            continue
        run, start = [], rva
        for a, m, tt in before:
            if a + m != start or start in targets:
                break
            p = piece(a, m, tt)
            if p is None:
                break
            run.insert(0, p)
            start = a
            if rva + 4 - start >= 5:
                break
        if rva + 4 - start >= 5 and free(start, rva + 4):
            sites[start] = (rva + 4 - start, run + ["s4"])
        else:
            unpatched.append(rva)
    last = 0
    out = []
    for rva in sorted(sites):
        total, pieces = sites[rva]
        if rva < last:
            continue
        out.append(f"{rva:x} {total} {' '.join(pieces)}")
        last = rva + total
    print("\n".join(out))
    print(
        f"{len(stores)} 256-bit stores; {len(out)} sites listed; {len(unpatched)} 4-byte stores left "
        f"(no room and nothing that can be moved)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
