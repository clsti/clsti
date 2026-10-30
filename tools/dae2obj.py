"""Convert simple COLLADA meshes (triangles with positions and normals; unit scale and node matrices applied,
Z-up) to Wavefront OBJ, which MuJoCo can load.
python tools/dae2obj.py <dir>   (writes <name>.obj next to each .dae)"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np


def _tag(e):
    return e.tag.split("}")[-1]


def convert(path):
    root = ET.parse(path).getroot()
    find = lambda e, name: [x for x in e.iter() if _tag(x) == name]
    unit = find(root, "unit")
    scale = float(unit[0].get("meter", 1.0)) if unit else 1.0
    sources = {s.get("id"): np.array(find(s, "float_array")[0].text.split(), float).reshape(-1, 3)
               for s in find(root, "source") if find(s, "float_array")}
    verts_src = {v.get("id"): find(v, "input")[0].get("source")[1:] for v in find(root, "vertices")}
    meshes = {}
    for g in find(root, "geometry"):
        parts = []
        for tri in find(g, "triangles"):
            inputs = [(i.get("semantic"), i.get("source")[1:], int(i.get("offset"))) for i in find(tri, "input")]
            stride = max(o for _, _, o in inputs) + 1
            idx = np.array(find(tri, "p")[0].text.split(), int).reshape(-1, stride)
            pos = nrm = None
            for sem, src, off in inputs:
                if sem == "VERTEX":
                    pos = sources[verts_src[src]][idx[:, off]]
                elif sem == "NORMAL":
                    nrm = sources[src][idx[:, off]]
            parts.append((pos, nrm))
        meshes[g.get("id")] = parts
    out_v, out_n = [], []
    for node in find(root, "node"):
        inst = [x for x in node if _tag(x) == "instance_geometry"]
        if not inst:
            continue
        mat = [x for x in node if _tag(x) == "matrix"]
        M = np.array(mat[0].text.split(), float).reshape(4, 4) if mat else np.eye(4)
        for pos, nrm in meshes[inst[0].get("url")[1:]]:
            out_v.append((pos @ M[:3, :3].T + M[:3, 3]) * scale)
            n = nrm @ np.linalg.inv(M[:3, :3]) if nrm is not None else np.zeros_like(pos)
            out_n.append(n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12))
    v, n = np.vstack(out_v), np.vstack(out_n)
    obj = os.path.splitext(path)[0] + ".obj"
    with open(obj, "w") as f:
        f.writelines(f"v {a:.6f} {b:.6f} {c:.6f}\n" for a, b, c in v)
        f.writelines(f"vn {a:.5f} {b:.5f} {c:.5f}\n" for a, b, c in n)
        f.writelines(f"f {k}//{k} {k + 1}//{k + 1} {k + 2}//{k + 2}\n" for k in range(1, len(v) + 1, 3))
    return obj, len(v) // 3


if __name__ == "__main__":
    for name in sorted(os.listdir(sys.argv[1])):
        if name.endswith(".dae"):
            print(*convert(os.path.join(sys.argv[1], name)))
