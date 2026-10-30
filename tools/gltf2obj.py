"""Convert simple glTF 2.0 meshes (one buffer, triangles; node transforms applied) to Wavefront OBJ with
normals and UVs, which MuJoCo can load.  glTF is Y-up; the output is Z-up like URDF / MJCF.
python tools/gltf2obj.py <dir>   (writes <name>.obj next to each .gltf)"""
import json
import os
import struct
import sys

import numpy as np

COMP = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(g, buf, i):
    a = g["accessors"][i]
    bv = g["bufferViews"][a["bufferView"]]
    n, k, c = a["count"], NCOMP[a["type"]], COMP[a["componentType"]]
    size = struct.calcsize(c)
    start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride", size * k)
    out = np.empty((n, k), np.float64)
    for j in range(n):
        out[j] = struct.unpack_from("<" + c * k, buf, start + j * stride)
    return out


def _node_matrix(node):
    if "matrix" in node:
        return np.array(node["matrix"], float).reshape(4, 4).T
    M = np.eye(4)
    if "scale" in node:
        M = np.diag([*node["scale"], 1.0]) @ M
    if "rotation" in node:
        x, y, z, w = node["rotation"]
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        T = np.eye(4)
        T[:3, :3] = R
        M = T @ M
    if "translation" in node:
        T = np.eye(4)
        T[:3, 3] = node["translation"]
        M = T @ M
    return M


def convert(path):
    g = json.load(open(path))
    buf = open(os.path.join(os.path.dirname(path), g["buffers"][0]["uri"]), "rb").read()
    lines, base = [], 0
    yup = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # Y-up -> Z-up
    for node in g["nodes"]:
        if "mesh" not in node:
            continue
        M = yup @ _node_matrix(node)
        for prim in g["meshes"][node["mesh"]]["primitives"]:
            at = prim["attributes"]
            P = _accessor(g, buf, at["POSITION"])
            P = (np.c_[P, np.ones(len(P))] @ M.T)[:, :3]
            N = _accessor(g, buf, at["NORMAL"]) @ M[:3, :3].T if "NORMAL" in at else None
            T = _accessor(g, buf, at["TEXCOORD_0"]) if "TEXCOORD_0" in at else None
            idx = _accessor(g, buf, prim["indices"]).astype(int).ravel() if "indices" in prim else np.arange(len(P))
            lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in P]
            if N is not None:
                lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in N]
            if T is not None:
                lines += [f"vt {u:.6f} {1 - v:.6f}" for u, v in T]
            for a, b, c in idx.reshape(-1, 3) + base + 1:
                if N is not None and T is not None:
                    lines.append(f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}")
                elif N is not None:
                    lines.append(f"f {a}//{a} {b}//{b} {c}//{c}")
                else:
                    lines.append(f"f {a} {b} {c}")
            base += len(P)
    out = path[:-5] + ".obj"
    open(out, "w").write("\n".join(lines) + "\n")
    return out


if __name__ == "__main__":
    d = sys.argv[1]
    for f in sorted(os.listdir(d)):
        if f.endswith(".gltf"):
            print(convert(os.path.join(d, f)))
