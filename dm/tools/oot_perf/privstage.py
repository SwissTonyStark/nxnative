#!/usr/bin/env python3
# privstage.py <src dir (meshpd.py output)> <out dir> <key>...
# Mesh shaders whose body writes every attribute word of a vertex from one thread at byte (base_s + 4 E + 128 k), base_s one of a few per-thread bases (the grass of Ocarina
# of Time: 3 vertices a thread, 27 / 21 words each): the vertex staging leaves the threadgroup array for thread-private registers dm_pv[s][k], written at the same sites, and the
# entries emit each vertex from them (its index n from the base: base / 4 = (n / 32) S + n % 32). The array keeps the system words, the primitive-count word and the index bytes
# (~80 words instead of ~2,700), so a GPU core's threadgroup memory holds many more workgroups. Unwritten vertices are zero, as before (a zero pass over the thread's own slots first).
import re, sys, os

def xform(src):
    m = re.search(r'threadgroup spvUnsafeArray<uint, (\d+)> (_\d+);', src)
    if not m: return None, 'no array'
    N, V = int(m.group(1)), m.group(2)
    hm = re.search(r'static inline uint dm_shi\(uint u\) \{ return u >= (\d+)u \? u - (\d+)u : u; \}\n', src)
    if not hm: return None, 'no meshpd helpers'
    E, P = int(hm.group(1)), int(hm.group(2))
    mnp = re.search(r'uint np = min\(' + V + r'\[(\d+)\] >> 24, (\d+)u\);', src)
    mb = re.search(r'uint b = (\d+)u \+ \(n / 32u\) \* (\d+)u \+ n % 32u;', src)
    ma = re.search(r'uint at = (\d+)u \+ n;', src)
    if not (mnp and mb and ma): return None, 'no entry loops'
    NP, S, A = int(mnp.group(1)), int(mb.group(2)), int(ma.group(1)); K = S // 32
    SHIFT = NP - 4; N2 = N - SHIFT + 1; TRAP = N2 - 1
    # the body
    bm = re.search(r'static inline __attribute__\(\(always_inline\)\) void dm_stage_body\(([^)]*)\)\n\{\n', src)
    if not bm: return None, 'no body'
    bstart = bm.start(); bend = src.index('\n}\n', bm.end()) + 3
    head, body, tail = src[:bstart], src[bstart:bend], src[bend:]
    defs = {d.group(1): d.group(2) for d in re.finditer(r'^    (?:int|uint) (_\d+) = (.*);$', body, re.M)}
    bases = []; bad = []
    def st(mm):
        a = defs.get(mm.group(1), ''); m1 = re.fullmatch(r'uint\(int\(uint\((_\d+)\) >> uint\(2\)\)\)', a)
        b = defs.get(m1.group(1), '') if m1 else ''; m2 = re.fullmatch(r'(_\d+) \+ (\d+)', b)
        if not m2: bad.append(mm.group(0)[:60]); return mm.group(0)
        base, c = m2.group(1), int(m2.group(2)) - 4 * E
        if c < 0 or c % 128 or c // 128 >= K: bad.append(mm.group(0)[:60]); return mm.group(0)
        if base not in bases: bases.append(base)
        return f'    dm_pv[{bases.index(base)}][{c // 128}] = {mm.group(2)};'
    body = re.sub(r'^    ' + V + r'\[dm_shw\(int\((_\d+)\)\)\] = (.*);$', st, body, flags=re.M)
    if bad: return None, 'stores of another shape: ' + '; '.join(bad[:3])
    NS = len(bases)
    if not NS or NS * 32 > 256: return None, f'{NS} bases'
    for s, base in enumerate(bases):
        body, n = re.subn(r'^(    int ' + base + r' = .*;)$', r'\1 dm_vb[%d] = uint(%s);' % (s, base), body, count=1, flags=re.M)
        if n != 1: return None, f'base {base} not a top-level definition'
    body = body.replace(', uint dm_tg)\n{\n', f', uint dm_tg, thread uint (&dm_pv)[{NS}][{K}], thread uint (&dm_vb)[{NS}])\n{{\n', 1)
    if 'dm_vb)' not in body: return None, 'body signature not as meshpd leaves it'
    # helpers: the array keeps [0, 4), the count word at 4, the index bytes after it, a trap slot last for anything else
    head = head.replace(hm.group(0), f'static inline uint dm_shi(uint u) {{ u = u >= {E}u ? u - {P}u : u; return u >= {NP}u ? u - {SHIFT}u : u < 4u ? u : {TRAP}u; }}\n')
    hb = re.search(r'static inline uint dm_shb\(uint b\) \{ return b >= (\d+)u \? b - (\d+)u : b; \}\n', head)
    if not hb: return None, 'no byte helper'
    head = head.replace(hb.group(0), f'static inline uint dm_shb(uint b) {{ b = b >= {hb.group(1)}u ? b - {hb.group(2)}u : b; return b >= {4 * NP}u ? b - {4 * SHIFT}u : b < 16u ? b : {4 * TRAP}u; }}\n')
    out = head + body + tail
    # entries: declarations, the call, the loops
    out = out.replace(f'    for (uint i = dm_lid; i < {N}u; i += 32u) {V}[i] = 0u;\n',
                      f'    for (uint i = dm_lid; i < {N2}u; i += 32u) {V}[i] = 0u;\n    uint dm_pv[{NS}][{K}] = {{}}; uint dm_vb[{NS}] = {{{", ".join(["0xffffffffu"] * NS)}}};\n')
    out = out.replace(f'dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, {V}, dm_payload, dm_tg);', f'dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, {V}, dm_payload, dm_tg, dm_pv, dm_vb);')
    out = out.replace(f'uint np = min({V}[{NP}] >> 24,', 'uint np = min(%s[4] >> 24,' % V)
    out = out.replace(f'uint at = {A}u + n;', f'uint at = {A - 4 * SHIFT}u + n;')
    nloops = [0]
    def loop(mm):
        nloops[0] += 1; maxv, asm = mm.group(1), mm.group(2)
        asm = re.sub(V + r'\[b \+ (\d+)u\]', lambda x: 'dm_pv[DM_S][%d]' % (int(x.group(1)) // 32), asm)
        if V + '[' in asm: raise ValueError('vertex assembly reads the array in another way')
        o = f'    for (uint n = dm_lid; n < {maxv}u; n += 32u) {{ dm_vertex_t v = {{}}; dm_out.set_vertex(n, v); }}\n    threadgroup_barrier(mem_flags::mem_threadgroup);\n'
        for s in range(NS):
            o += (f'    if (dm_vb[{s}] != 0xffffffffu)\n    {{\n        uint w = dm_vb[{s}] >> 2u, n = (w / {S}u) * 32u + (w % {S}u);\n        dm_vertex_t v;\n'
                  + asm.replace('DM_S', str(s)) + f'        if (n < {maxv}u) dm_out.set_vertex(n, v);\n    }}\n')
        return o
    out = re.sub(r'    for \(uint n = dm_lid; n < (\d+)u; n \+= 32u\)\n    \{\n        uint b = \d+u \+ \(n / 32u\) \* \d+u \+ n % 32u;\n        dm_vertex_t v;\n(.*?)        dm_out\.set_vertex\(n, v\);\n    \}\n', loop, out, flags=re.S)
    out = out.replace(f'spvUnsafeArray<uint, {N}>', f'spvUnsafeArray<uint, {N2}>')
    if f'{V}[{NP}]' in out or f'{A}u + n' in out: return None, 'an old constant is left'
    return out, f'N {N} -> {N2}, {NS} vertices a thread x {K} words, {nloops[0]} entry loops'

if __name__ == '__main__':
    sd, od = sys.argv[1:3]; os.makedirs(od, exist_ok=True)
    for k in sys.argv[3:]:
        try: out, why = xform(open(f'{sd}/{k}_b.metal').read())
        except ValueError as e: out, why = None, str(e)
        print(k, why)
        if out: open(f'{od}/{k}_b.metal', 'w').write(out)
