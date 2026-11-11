#!/usr/bin/env python3
# Rewrites a translated shader's raw global memory read helper so that a read inside the range chosen last costs one compare and one add
# (the range held in registers instead of re-read from the pool table per word), and the hint plus search are one out-of-line function
# instead of being inlined at every read. Only the exact helper shape is touched; anything else is reported and left alone.
# Usage: rawfast.py <cache dir b-chain> <out dir> <key>...
import os, re, subprocess, sys, tempfile

PRELUDE = """
struct dm_range { ulong s; ulong l; ulong d; };
static __attribute__((noinline)) uint dm_raw_slow(ulong a, const device uint* t, thread dm_range& r)
{
    uint hi = uint(a >> 32), lo = uint(a);
    uint key = (hi << 11) | (lo >> 21);
    uint w = t[(key & 8191u) * 8u + 6u];
    uint i = w & 8191u;
    if (!(i != 0u && (w >> 13) == key && hi < 256u))
    {
        i = 0u;
        for (uint step = 4096u; step != 0u; step >>= 1)
        {
            uint n = i + step;
            ulong start = ulong(t[n * 8u]) | (ulong(t[n * 8u + 1u]) << 32);
            if (start <= a) i = n;
        }
    }
    ulong start = ulong(t[i * 8u]) | (ulong(t[i * 8u + 1u]) << 32);
    ulong end = ulong(t[i * 8u + 2u]) | (ulong(t[i * 8u + 3u]) << 32);
    if (a >= end) return 0u;
    ulong delta = ulong(t[i * 8u + 4u]) | (ulong(t[i * 8u + 5u]) << 32);
    r.s = start; r.l = end - start; r.d = delta;
    return *(reinterpret_cast<const device uint*>(a + delta));
}
"""

HELPER = re.compile(r'static inline __attribute__\(\(always_inline\)\)\nint (_\d+)\(thread const int& (_\d+), thread const int& (_\d+), (?:const )?device (_\d+)& (cp_s\d+|vp_s\d+|fp_s\d+), thread uint& (_\d+)\)\n\{\n.*?\n\}\n', re.S)


def rewrite(src):
    found = [m for m in HELPER.finditer(src) if '13' and 'uint(_' in m.group(0) and '>> uint(21)' in m.group(0)]
    if len(found) != 1: return None, f'{len(found)} helpers of the known shape'
    m = found[0]
    body = m.group(0)
    if len(re.findall(r'reinterpret_cast<(?:const )?device uint\*>', body)) != 1 or re.search(r'reinterpret_cast<[^>]*>\([^;]*\)\) = ', body): return None, 'helper is not a plain load'
    if len(re.findall(r'thread uint& _\d+\)\n\{', src)) != 1: return None, 'other helpers share the cache word'
    const = 'const ' if 'const device ' + m.group(4) in m.group(0).split('\n')[1] else ''
    if len(re.findall(r'\+ 4096;', body)) != 1 or body.count('>> uint(21)') != 1: return None, 'helper search differs'
    name, lo, hi, ty, buf, cache = m.groups()
    fast = (f'static inline __attribute__((always_inline))\nint {name}(thread const int& {lo}, thread const int& {hi}, {const}device {ty}& {buf}, thread dm_range& {cache})\n{{\n'
            f'    ulong a = ulong(uint({lo})) | (ulong(uint({hi})) << 32);\n'
            f'    if (a - {cache}.s < {cache}.l) return as_type<int>(*(reinterpret_cast<const device uint*>(a + {cache}.d)));\n'
            f'    return as_type<int>(dm_raw_slow(a, &{buf}._m0[0], {cache}));\n}}\n')
    out = src[:m.start()] + PRELUDE + fast + src[m.end():]
    others = len(re.findall(rf'\b{cache}\b', out))
    out, n1 = re.subn(rf'\buint {cache} = 0u;', f'dm_range {cache} = {{0ul, 0ul, 0ul}};', out)
    out, n2 = re.subn(rf'thread uint& {cache}\b', f'thread dm_range& {cache}', out)
    if n1 < 1: return None, f'{n1} declarations of the cache word'
    # every other use of the word must be passing it on to the helper or a function that does
    for line in out.splitlines():
        if re.search(rf'\b{cache}\b', line) and not re.search(rf'dm_range|{name}\(|, {cache}\)|{cache}\.', line): return None, f'other use of the cache word: {line.strip()[:80]}'
    return out, f'helper {name}, {out.count(name + "(") - 1} reads'


def main():
    cache, out = sys.argv[1:3]
    os.makedirs(out, exist_ok=True)
    for key in sys.argv[3:]:
        src = open(f'{cache}/{key}_b.metal').read()
        new, note = rewrite(src)
        if new is None: print(key, 'LEFT ALONE:', note); continue
        msl = f'{out}/{key}_b.metal'; open(msl, 'w').write(new)
        with tempfile.TemporaryDirectory() as td:
            air = f'{td}/x.air'
            subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-c', '-O1', '-fpreserve-invariance', msl, '-o', air], check=True, capture_output=True)
            lib = f'{out}/{key}_b.metallib'
            subprocess.run(['xcrun', '-sdk', 'macosx', 'metallib', air, '-o', lib], check=True, capture_output=True)
        print(key, note, os.path.getsize(f'{cache}/{key}_b.metallib'), '->', os.path.getsize(lib), flush=True)


main()
