#!/usr/bin/env python3
# On a shader already rewritten by rawfast.py (dm_range {s, l, d} in 64 bits): the range kept as its part inside one 4 GB window
# {hi, lo, len, p}, so a read inside it costs a 32-bit subtract, two compares and a pointer add instead of 64-bit arithmetic.
# Usage: raw32.py <src dir> <out dir> <key>...
import re, sys, os

OLD_STRUCT = 'struct dm_range { ulong s; ulong l; ulong d; };'
NEW_STRUCT = 'struct dm_range { uint hi; uint lo; uint len; const device uchar* p; };'
OLD_SET = '    r.s = start; r.l = end - start; r.d = delta;\n'
NEW_SET = ('    ulong ws = max(start, ulong(hi) << 32), we = min(end, (ulong(hi) + 1ul) << 32);\n'
           '    r.hi = hi; r.lo = uint(ws); r.len = uint(min(we - ws, 0xfffffffful)); r.p = reinterpret_cast<const device uchar*>(ws + delta);\n')
HELPER = re.compile(r'(int _\d+\(thread const int& (_\d+), thread const int& (_\d+), (?:const )?device _\d+& (?:cp|vp|fp)_s\d+, thread dm_range& (_\d+)\)\n\{\n)'
                    r'    ulong a = ulong\(uint\(\2\)\) \| \(ulong\(uint\(\3\)\) << 32\);\n'
                    r'    if \(a - \4\.s < \4\.l\) return as_type<int>\(\*\(reinterpret_cast<const device uint\*>\(a \+ \4\.d\)\)\);\n')

def xform(src):
    if OLD_STRUCT not in src or OLD_SET not in src: return None, 'not a rawfast shader'
    n = [0]
    def h(m):
        n[0] += 1; lo, hi, r = m.group(2), m.group(3), m.group(4)
        return (m.group(1) + f'    uint off = uint({lo}) - {r}.lo;\n'
                f'    if (uint({hi}) == {r}.hi && off < {r}.len) return as_type<int>(*(reinterpret_cast<const device uint*>({r}.p + off)));\n'
                f'    ulong a = ulong(uint({lo})) | (ulong(uint({hi})) << 32);\n')
    out = HELPER.sub(h, src)
    if not n[0]: return None, 'no helper of the known shape'
    out = out.replace(OLD_STRUCT, NEW_STRUCT).replace(OLD_SET, NEW_SET)
    out, d = re.subn(r'dm_range (_\d+) = \{0ul, 0ul, 0ul\};', r'dm_range \1 = {0u, 0u, 0u, nullptr};', out)
    if re.search(r'\.s\b|\.l\b|\.d\b', '\n'.join(l for l in out.splitlines() if 'dm_range' in l or '_m0' not in l and re.search(r'_\d+\.(s|l|d)\b', l))):
        pass
    if re.search(r'\b_\d+\.(s|l|d)\b', out): return None, 'a use of the old range fields is left'
    return out, f'{n[0]} helpers, {d} declarations'

if __name__ == '__main__':
    sd, od = sys.argv[1:3]; os.makedirs(od, exist_ok=True)
    for k in sys.argv[3:]:
        out, why = xform(open(f'{sd}/{k}_b.metal').read())
        print(k, why)
        if out: open(f'{od}/{k}_b.metal', 'w').write(out)
