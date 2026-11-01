"""Render-only appearance of the produce (used by the media scripts, never by the benchmarks).

The contact geometry stays the primitive of twin_core.OBJECTS; the meshes and sepals below have no mass and no contacts,
so the physics is identical with or without them.  The strawberry body is a surface of revolution that follows the
collision ellipsoid from the calyx to the middle and tapers toward the tip, never outside the ellipsoid (so a finger
that touches the fruit never sinks into the drawn surface; near the tip the drawn surface lies at most ~1.5 mm inside);
achenes are in the texture, sepals and stem are thin geoms.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ASSETS = Path(__file__).resolve().parent / "assets"


def _profile(u):
    """radius factor of the strawberry along its axis, u in [-1 (tip), 1 (calyx)]"""
    g = np.sqrt(np.clip(1 - u ** 2, 0, None))
    return g * (1 - 0.20 * np.clip(-u, 0, None) ** 1.3)        # never outside the collision ellipsoid


def strawberry_mesh(a, b, c, nu=64, nphi=48, path=None):
    path = path or ASSETS / "strawberry.obj"
    us = -np.cos(np.linspace(0, math.pi, nu))                           # denser near both ends
    lines = []
    for i, u in enumerate(us):
        g = _profile(u)
        for j in range(nphi + 1):
            ph = 2 * math.pi * j / nphi
            lines.append(f"v {a*u:.6f} {b*g*math.cos(ph):.6f} {c*g*math.sin(ph):.6f}")
    for i, u in enumerate(us):
        for j in range(nphi + 1):
            lines.append(f"vt {j/nphi:.5f} {i/(nu-1):.5f}")
    idx = lambda i, j: i * (nphi + 1) + j + 1
    for i in range(nu - 1):
        for j in range(nphi):
            p, q, r, s = idx(i, j), idx(i, j + 1), idx(i + 1, j + 1), idx(i + 1, j)
            lines.append(f"f {p}/{p} {s}/{s} {r}/{r}")
            lines.append(f"f {p}/{p} {r}/{r} {q}/{q}")
    path.write_text("\n".join(lines) + "\n")
    return path


def strawberry_texture(path=None, w=1024, h=512, seed=3):
    """u (around) -> x, axis (tip -> calyx) -> y; red skin, pale shoulder, achenes in a staggered lattice"""
    path = path or ASSETS / "strawberry.png"
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h)[:, None]
    base = np.zeros((h, w, 3))
    red = np.array([214, 30, 38]) / 255
    deep = np.array([168, 14, 26]) / 255
    pale = np.array([236, 120, 92]) / 255
    t = np.clip((y - 0.86) / 0.14, 0, 1)[:, :, None]                     # pale band under the calyx
    yy3 = y[:, :, None]
    base[:] = deep * (1 - yy3) * 0.35 + red * (0.65 + 0.35 * yy3)
    base = base * (1 - t) + pale * t
    base += rng.normal(0, 0.012, base.shape)
    img = Image.fromarray(np.clip(base * 255, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    nx = 17
    for r in range(3, 32):
        v = r / 33
        if v > 0.93:
            break
        for k in range(nx):
            x = (k + 0.5 * (r % 2) + rng.normal(0, 0.08)) / nx * w
            yy = v * h + rng.normal(0, 1.2)
            sx = 6.5 * (0.6 + 0.6 * math.sin(math.pi * v) + 0.2)          # wider cells where the fruit is wide
            d.ellipse((x - sx - 2, yy - 6, x + sx + 2, yy + 6), fill=(170, 16, 26))   # dimple
            d.ellipse((x - sx * 0.40, yy - 3.0, x + sx * 0.40, yy + 3.0), fill=(232, 205, 110))  # achene
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    img.save(path)
    return path


def tomato_texture(path=None, w=512, h=256, seed=4):
    path = path or ASSETS / "tomato.png"
    rng = np.random.default_rng(seed)
    base = np.array([214, 58, 28]) / 255
    y = np.linspace(0, 1, h)[:, None, None]
    img = base * (0.92 + 0.08 * np.cos(np.linspace(0, 12 * math.pi, w))[None, :, None] * (1 - abs(2 * y - 1))) * np.ones((h, w, 3))
    img += rng.normal(0, 0.01, img.shape)
    Image.fromarray(np.clip(img * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0)).save(path)
    return path


def assets_xml():
    ASSETS.mkdir(exist_ok=True)
    from twin_core import OBJECTS
    a, b, c = OBJECTS["strawberry"].size
    sm = strawberry_mesh(a, b, c)
    st = strawberry_texture()
    return (f'<mesh name="strawberry_vis" file="{sm}" inertia="shell"/>'
            f'<texture name="strawberry_tex" type="2d" file="{st}"/>'
            f'<material name="strawberry_mat" texture="strawberry_tex" specular="0.22" shininess="0.25" reflectance="0"/>'
            f'<material name="tomato_mat" rgba="0.55 0.045 0.03 1" specular="0.35" shininess="0.6" reflectance="0"/>'
            f'<material name="sepal" rgba="0.24 0.46 0.16 1" specular="0.2" shininess="0.3"/>')


def _sepals(center, axis, n, length, width, back_deg, radius0, var=8):
    """thin flattened ellipsoids arranged around `axis` at `center`, tilted back by back_deg"""
    ax = np.asarray(axis, float) / np.linalg.norm(axis)
    tmp = np.array([0, 0, 1.0]) if abs(ax[2]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(ax, tmp); e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1)
    out = []
    for k in range(n):
        ph = 2 * math.pi * (k + 0.3) / n
        rad = math.cos(ph) * e1 + math.sin(ph) * e2                    # radial direction
        tilt = math.radians(back_deg + var * math.sin(3 * ph))
        dirv = math.cos(tilt) * rad - math.sin(tilt) * ax               # leaf lies back over the shoulder
        nrm = np.cross(dirv, np.cross(ax, rad)); nrm /= np.linalg.norm(nrm)
        side = np.cross(nrm, dirv)
        pos = np.asarray(center) + (radius0 + length * 0.5) * dirv
        R = np.stack([dirv, side, nrm], axis=1)                         # columns: local x, y, z
        q = _quat(R)
        out.append(f'<geom type="ellipsoid" size="{length/2:.4f} {width/2:.4f} 0.0005" pos="{pos[0]:.4f} {pos[1]:.4f} {pos[2]:.4f}" '
                   f'quat="{q}" material="sepal" contype="0" conaffinity="0" mass="0" group="1"/>')
    return "".join(out)


def _quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[i + 1] = 0.25 * s
        q[j + 1] = (R[j, i] + R[i, j]) / s
        q[k + 1] = (R[k, i] + R[i, k]) / s
        w, x, y, z = q
    return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"


def object_visuals(base, size):
    """extra render-only geoms for one object body, in the body frame"""
    if base == "strawberry":
        a = size[0]
        return ('<geom type="mesh" mesh="strawberry_vis" material="strawberry_mat" contype="0" conaffinity="0" mass="0" group="1"/>'
                + _sepals((a * 0.93, 0, 0), (1, 0, 0), 9, 0.0115, 0.0042, 62, 0.0015)
                + f'<geom type="capsule" fromto="{a*0.93:.4f} 0 0 {a+0.0045:.4f} 0 0.0012" size="0.0010" material="sepal" contype="0" conaffinity="0" mass="0" group="1"/>')
    if base == "tomato":
        r = size[0]
        return (_sepals((0, 0, r * 1.012), (0, 0, 1), 6, 0.0095, 0.0038, 13, 0.0008, var=1.5)
                + f'<geom type="capsule" fromto="0 0 {r*0.98:.4f} 0.0006 0 {r+0.0035:.4f}" size="0.0011" material="sepal" contype="0" conaffinity="0" mass="0" group="1"/>')
    return ""
