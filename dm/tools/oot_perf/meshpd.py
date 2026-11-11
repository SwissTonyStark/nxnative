#!/usr/bin/env python3
# Mesh shaders fed by a task shader: read the task payload straight from object_data memory instead of copying it into the
# workgroup's shared-memory array. The array keeps the system words [0, 4) and everything after the payload, shifted down by
# the payload's size, so each workgroup holds 3.2 KB less threadgroup memory (more workgroups resident per GPU core).
# Reads of the payload range go to dm_payload (word 4 is the workgroup index, as the entry used to write it); writes and
# byte stores are shifted. Usage: meshpd.py <src dir> <out dir> <key>...   (keys with no payload copy are reported and skipped)
import re, sys, os

def xform(src):
    m = re.search(r'\[\[mesh\]\] void main0\([^{]*\{\n    threadgroup spvUnsafeArray<uint, (\d+)> (_\d+);\n', src)
    if not m: return None, 'no mesh entry'
    N, V = int(m.group(1)), m.group(2)
    pc = re.search(r'    for \(uint i = dm_lid; i < (\d+)u; i \+= 32u\) ' + V + r'\[4u \+ i\] = dm_payload\.w\[i\];\n', src)
    if not pc: return None, 'no payload copy'
    P = int(pc.group(1)); M = N - P; E = 4 + P   # first word after the payload
    pt = re.search(r'struct dm_payload_t \{ uint w\[' + str(P) + r'\]; \};\n', src)
    if not pt: return None, 'no payload type'
    body = re.search(r'static inline __attribute__\(\(always_inline\)\) void dm_stage_body\(([^)]*)\)\n\{\n', src)
    if not body: return None, 'no body'
    bstart = body.start()
    bend = src.index('\n}\n', body.end()) + 3
    # helper: the byte store, shifted
    s8 = re.search(r'\(\(threadgroup uchar\*\)&' + V + r'\[0\]\)\[uint\((_\d+)\)\]', src)
    head, btxt, tail = src[:bstart], src[bstart:bend], src[bend:]
    if s8 and s8.start() < bstart:
        head = head.replace(s8.group(0), '((threadgroup uchar*)&%s[0])[dm_shb(uint(%s))]' % (V, s8.group(1)))
    elif s8: return None, 'byte store not before body'
    # body: writes then reads
    nw = [0]; nr = [0]
    def wsub(mm): nw[0] += 1; return '%s[dm_shw(%s)] =' % (V, mm.group(1))
    btxt = re.sub(re.escape(V) + r'\[([^\[\]]*)\] =(?!=)', wsub, btxt)
    def rsub(mm):
        if mm.group(1).startswith('dm_shw('): return mm.group(0)
        nr[0] += 1; return 'dm_shr(%s, dm_payload, dm_tg, %s)' % (V, mm.group(1))
    btxt = re.sub(re.escape(V) + r'\[([^\[\]]*)\]', rsub, btxt)
    # the write rule must not be undone by the read rule: dm_shw(...) indexes stay as they are (handled above)
    btxt = btxt.replace('threadgroup spvUnsafeArray<uint, %d>& %s)' % (N, V), 'threadgroup spvUnsafeArray<uint, %d>& %s, const object_data dm_payload_t& dm_payload, uint dm_tg)' % (N, V), 1)
    helpers = (pt.group(0) +
        'static inline uint dm_shi(uint u) { return u >= %du ? u - %du : u; }\n' % (E, P) +
        'static inline uint dm_shw(int i) { return dm_shi(uint(i)); }\n' +
        'static inline uint dm_shb(uint b) { return b >= %du ? b - %du : b; }\n' % (4 * E, 4 * P) +
        'static inline __attribute__((always_inline)) uint dm_shr(threadgroup spvUnsafeArray<uint, %d>& s, const object_data dm_payload_t& p, uint tg, int i) { uint u = uint(i); if (u - 4u < %du) return u == 4u ? tg : p.w[u - 4u]; return s[dm_shi(u)]; }\n' % (N, P))
    tail = tail.replace(pt.group(0), '', 1)
    # helpers go before the first function that takes the array (the byte store helper, or the body)
    first = min([x for x in [head.find('static inline __attribute__((always_inline))\nvoid dmsh_shared_store8'), len(head)] if x >= 0])
    head = head[:first] + helpers + head[first:]
    out = head + btxt + tail
    # entries
    out = out.replace(pc.group(0), '')
    out = out.replace('{ %s[0] = dm_tg; %s[4] = dm_tg; }' % (V, V), '{ %s[0] = dm_tg; }' % V)
    out = out.replace('dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, %s);' % V, 'dm_stage_body(spvDescriptorSet0, gl_SubgroupInvocationID, %s, dm_payload, dm_tg);' % V)
    out = out.replace('for (uint i = dm_lid; i < %du; i += 32u) %s[i] = 0u;' % (N, V), 'for (uint i = dm_lid; i < %du; i += 32u) %s[i] = 0u;' % (M, V))
    out = re.sub(re.escape(V) + r'\[(\d+)\] >> 24', lambda mm: '%s[%d] >> 24' % (V, int(mm.group(1)) - P if int(mm.group(1)) >= E else int(mm.group(1))), out)
    out = re.sub(r'uint b = (\d+)u \+', lambda mm: 'uint b = %du +' % (int(mm.group(1)) - P), out)
    out = re.sub(r'uint at = (\d+)u \+ n;', lambda mm: 'uint at = %du + n;' % (int(mm.group(1)) - 4 * P), out)
    out = out.replace('spvUnsafeArray<uint, %d>' % N, 'spvUnsafeArray<uint, %d>' % M)
    return out, 'N %d -> %d, payload %d, body writes %d reads %d' % (N, M, P, nw[0], nr[0])

if __name__ == '__main__':
    sd, od = sys.argv[1:3]; os.makedirs(od, exist_ok=True)
    for k in sys.argv[3:]:
        out, why = xform(open(f'{sd}/{k}_b.metal').read())
        print(k, why)
        if out: open(f'{od}/{k}_b.metal', 'w').write(out)
