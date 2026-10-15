"""Instance Island — copy one retopo island's topology onto every congruent island.

A garment often has the same pattern piece several times (sleeve gores, skirt
panels, collar halves), and the retopo of each copy should share one topology.
Replace Twin Island covers a left/right reflected PAIR with a rigid copy; this
covers the general case:

  * any number of targets, same-handed or reflected
  * outlines that are only nearly identical (a few mm off) — the source's
    boundary vertices are re-laid on each target's OWN outline, segment by
    segment (corner to corner), at the same arc-length fractions, so the
    target's pattern outline is kept exactly
  * the target's boundary vertex count may differ from the source's; the
    result always takes the source's divisions

Interior vertices follow a rigid fit of the source onto the target plus an
inverse-distance blend of the boundary residuals. Every shape key is carried
over by interpolating the target's old geometry (boundary along its old
outline, interior barycentrically), and when the active object is the Retopo
with a Guide set, the CLO Projector re-projects the new vertices afterwards.
Optionally, seam vertices shared with neighbouring islands are snapped to the
same 3D position (AC9_3D_Project) so Finalize can weld them.

Destructive on the targets: every vertex and face except the outline corners
is deleted and rebuilt. Ctrl+Z reverts.
"""

import math

import bmesh
import bpy
from bpy.props import BoolProperty, FloatProperty
from mathutils import Vector
from mathutils.geometry import barycentric_transform, intersect_point_tri_2d


PROJECT_SK = "AC9_3D_Project"
_RESAMPLE_N = 64


# ---------------------------------------------------------------- islands ---

def _islands(bm):
    seen = set()
    out = []
    for f in bm.faces:
        if f in seen:
            continue
        stack = [f]
        seen.add(f)
        comp = []
        while stack:
            x = stack.pop()
            comp.append(x)
            for e in x.edges:
                for g in e.link_faces:
                    if g not in seen:
                        seen.add(g)
                        stack.append(g)
        out.append(comp)
    return out


def _boundary_loop(faces):
    """Ordered outer boundary vertices of an island, counter-clockwise.

    Returns None for islands whose boundary is not a single simple loop
    (holes, bow-ties) — those are skipped rather than guessed at.
    """
    fs = set(faces)
    adj = {}
    for f in faces:
        for e in f.edges:
            if sum(1 for g in e.link_faces if g in fs) == 1:
                a, b = e.verts
                adj.setdefault(a, []).append(b)
                adj.setdefault(b, []).append(a)
    if not adj or any(len(n) != 2 for n in adj.values()):
        return None
    start = next(iter(adj))
    seq = [start]
    prev, cur = None, start
    while True:
        nxt = adj[cur][0] if adj[cur][0] is not prev else adj[cur][1]
        if nxt is start:
            break
        seq.append(nxt)
        prev, cur = cur, nxt
        if len(seq) > len(adj):
            return None
    if len(seq) != len(adj):
        return None
    if _signed_area([v.co.to_2d() for v in seq]) < 0:
        seq.reverse()
    return seq


def _signed_area(pts):
    n = len(pts)
    return sum(pts[i].x * pts[(i + 1) % n].y - pts[(i + 1) % n].x * pts[i].y
               for i in range(n)) / 2


# ------------------------------------------------------------ 2D helpers ---

def _corners(pts, angle_deg):
    """Indices of loop points whose turning angle exceeds angle_deg."""
    n = len(pts)
    out = []
    for i in range(n):
        a = pts[i] - pts[i - 1]
        b = pts[(i + 1) % n] - pts[i]
        if a.length == 0 or b.length == 0:
            continue
        t = math.degrees(math.atan2(a.x * b.y - a.y * b.x, a.dot(b)))
        if abs(t) > angle_deg:
            out.append(i)
    return out


def _resample(pts, n):
    """n points evenly spaced by arc length along a closed polyline."""
    m = len(pts)
    seg = [(pts[(i + 1) % m] - pts[i]).length for i in range(m)]
    total = sum(seg)
    out = []
    i, acc = 0, 0.0
    for k in range(n):
        target = total * k / n
        while acc + seg[i] < target and i < m - 1:
            acc += seg[i]
            i += 1
        w = (target - acc) / seg[i] if seg[i] else 0.0
        out.append(pts[i].lerp(pts[(i + 1) % m], w))
    return out, total


def _rigid_fit(src, dst):
    """Best rotation+translation (no scale, no reflection) mapping src to dst.

    Returns (apply, rms).
    """
    n = len(src)
    cs = sum(src, Vector((0, 0))) / n
    cd = sum(dst, Vector((0, 0))) / n
    sxx = sxy = 0.0
    for a, b in zip(src, dst):
        a = a - cs
        b = b - cd
        sxx += a.x * b.x + a.y * b.y
        sxy += a.x * b.y - a.y * b.x
    th = math.atan2(sxy, sxx)
    c, s = math.cos(th), math.sin(th)

    def apply(p):
        q = p - cs
        return Vector((c * q.x - s * q.y + cd.x, s * q.x + c * q.y + cd.y))

    rms = math.sqrt(sum((apply(a) - b).length_squared for a, b in zip(src, dst)) / n)
    return apply, rms


def _mirror2(p):
    return Vector((-p.x, p.y))


def _outline_match(src_pts, dst_pts):
    """Compare two closed outlines; return (rms/size, mirrored) of the best fit
    over every cyclic start, with and without reflection."""
    a, la = _resample(src_pts, _RESAMPLE_N)
    b, lb = _resample(dst_pts, _RESAMPLE_N)
    size = max(la, lb) / 4.0
    best = (float("inf"), False)
    for mirrored in (False, True):
        aa = [_mirror2(p) for p in reversed(a)] if mirrored else a
        if mirrored:
            aa, _ = _resample(aa, _RESAMPLE_N)
        for s in range(_RESAMPLE_N):
            bb = b[s:] + b[:s]
            _, rms = _rigid_fit(aa, bb)
            if rms / size < best[0]:
                best = (rms / size, mirrored)
    return best


# --------------------------------------------------------------- template ---

class _Template:
    """The source island as plain data: 2D positions, faces, CCW boundary
    loop split at its corners. Built either as-is or reflected, so a mirrored
    target is just another same-handed target of the reflected template."""

    def __init__(self, pos, faces, loop, angle_deg):
        self.pos = pos
        self.faces = faces
        self.loop = loop
        lp = [pos[i] for i in loop]
        cidx = _corners(lp, angle_deg)
        if len(cidx) < 2:
            raise RuntimeError("source outline has fewer than 2 corners")
        self.corners = [loop[i] for i in cidx]
        n = len(loop)
        # role[i] = ('c', k) for corner k, ('b', k, u) for a vertex on segment k
        self.role = {}
        for k, i in enumerate(cidx):
            j = cidx[(k + 1) % len(cidx)]
            span = [loop[(i + t) % n] for t in range(((j - i) % n) + 1)]
            d = [0.0]
            for p, q in zip(span, span[1:]):
                d.append(d[-1] + (pos[p] - pos[q]).length)
            self.role[span[0]] = ("c", k)
            for t, vi in enumerate(span[1:-1], 1):
                self.role[vi] = ("b", k, d[t] / d[-1])

    @classmethod
    def from_island(cls, faces, loop_verts, angle_deg, mirrored):
        verts = list({v for f in faces for v in f.verts})
        idx = {v: i for i, v in enumerate(verts)}
        pos = [v.co.to_2d() for v in verts]
        fl = [[idx[v] for v in f.verts] for f in faces]
        loop = [idx[v] for v in loop_verts]
        if mirrored:
            pos = [_mirror2(p) for p in pos]
            fl = [list(reversed(f)) for f in fl]
            loop = list(reversed(loop))
        return cls(pos, fl, loop, angle_deg)


# ---------------------------------------------------------------- rebuild ---

def _segments(loop_verts, angle_deg):
    pts = [v.co.to_2d() for v in loop_verts]
    cidx = _corners(pts, angle_deg)
    n = len(loop_verts)
    segs = []
    for k, i in enumerate(cidx):
        j = cidx[(k + 1) % len(cidx)]
        segs.append([loop_verts[(i + t) % n] for t in range(((j - i) % n) + 1)])
    return segs


def _polyline_at(cos, layers, params, u):
    """Point (2D co, {layer: value}) at arc fraction u along a sampled span."""
    for i in range(len(params) - 1):
        if params[i] - 1e-9 <= u <= params[i + 1] + 1e-9:
            w = (u - params[i]) / (params[i + 1] - params[i]) if params[i + 1] > params[i] else 0.0
            return (cos[i].lerp(cos[i + 1], w),
                    {k: vals[i].lerp(vals[i + 1], w) for k, vals in layers.items()})
    return cos[-1], {k: vals[-1] for k, vals in layers.items()}


def _rebuild_target(bm, tpl, faces, loop_verts, angle_deg, sk_layers, uv_layers):
    """Replace one target island with the template. Returns a stats dict."""
    segs = _segments(loop_verts, angle_deg)
    K = len(tpl.corners)
    if len(segs) != K:
        raise RuntimeError(f"outline has {len(segs)} corners, source has {K}")

    # Which target corner is template corner 0: best rigid fit over shifts.
    tc = [tpl.pos[i] for i in tpl.corners]
    best = None
    for shift in range(K):
        dc = [segs[(k + shift) % K][0].co.to_2d() for k in range(K)]
        _, rms = _rigid_fit(tc, dc)
        if best is None or rms < best[0]:
            best = (rms, shift)
    shift = best[1]
    segs = segs[shift:] + segs[:shift]

    # Old geometry, sampled before anything is deleted.
    seg_data = []
    for s in segs:
        d = [0.0]
        for p, q in zip(s, s[1:]):
            d.append(d[-1] + (p.co.to_2d() - q.co.to_2d()).length)
        seg_data.append(([v.co.copy() for v in s],
                         {k: [v[l].copy() for v in s] for k, l in sk_layers.items()},
                         [x / d[-1] for x in d]))
    tris = []
    for f in faces:
        loops = list(f.loops)
        for k in range(1, len(loops) - 1):
            tl = (loops[0], loops[k], loops[k + 1])
            tris.append(([l.vert.co.copy() for l in tl],
                         {k2: [l.vert[l2].copy() for l in tl] for k2, l2 in sk_layers.items()},
                         {k2: [l[l2].uv.copy() for l in tl] for k2, l2 in uv_layers.items()}))
    zavg = sum(v.co.z for v in loop_verts) / len(loop_verts)

    def find_tri(p2):
        for t in tris:
            co = t[0]
            if intersect_point_tri_2d(p2, co[0].to_2d(), co[1].to_2d(), co[2].to_2d()):
                return t
        return min(tris, key=lambda t: (p2 - ((t[0][0] + t[0][1] + t[0][2]) / 3).to_2d()).length)

    def bary(t, P, vals):
        co = t[0]
        return barycentric_transform(P, co[0], co[1], co[2], vals[0], vals[1], vals[2])

    def bary_uv(t, P, uvs):
        co = t[0]
        r = barycentric_transform(P, co[0], co[1], co[2],
                                  uvs[0].to_3d(), uvs[1].to_3d(), uvs[2].to_3d())
        return r.to_2d()

    # Face attributes to carry over: the majority value on the old island.
    def majority(values):
        cnt = {}
        for x in values:
            cnt[x] = cnt.get(x, 0) + 1
        return max(cnt, key=cnt.get)

    mat = majority([f.material_index for f in faces])
    smooth = majority([f.smooth for f in faces])
    int_layers = {k: majority([f[l] for f in faces]) for k, l in bm.faces.layers.int.items()}

    corners = [s[0] for s in segs]
    n = len(tpl.pos)
    new_pos = [None] * n           # ('corner', vert) | (co, {sk: val})
    for i in range(n):
        r = tpl.role.get(i)
        if r is None:
            continue
        if r[0] == "c":
            new_pos[i] = ("corner", corners[r[1]])
        else:
            co, vals = _polyline_at(*seg_data[r[1]], r[2])
            new_pos[i] = (co, vals)

    bnd = [i for i in range(n) if new_pos[i] is not None]
    src = [tpl.pos[i] for i in bnd]
    dst = [(new_pos[i][1].co.to_2d() if new_pos[i][0] == "corner" else new_pos[i][0].to_2d())
           for i in bnd]
    fit, _ = _rigid_fit(src, dst)
    resid = [(fit(a), b - fit(a)) for a, b in zip(src, dst)]
    for i in range(n):
        if new_pos[i] is not None:
            continue
        q = fit(tpl.pos[i])
        ws = 0.0
        dv = Vector((0, 0))
        for p, d in resid:
            w = 1.0 / max((q - p).length, 1e-6) ** 4
            ws += w
            dv += d * w
        q = q + dv / ws
        P = Vector((q.x, q.y, zavg))
        t = find_tri(q)
        new_pos[i] = (P, {k: bary(t, P, t[1][k]) for k in sk_layers})

    # Delete the old island except its corners.
    keep = set(corners)
    old_faces = len(faces)
    old_verts = list({v for f in faces for v in f.verts if v not in keep})
    bmesh.ops.delete(bm, geom=list(faces), context='FACES_ONLY')
    bmesh.ops.delete(bm, geom=old_verts, context='VERTS')
    for cv in corners:
        for e in list(cv.link_edges):
            if not e.link_faces:
                bm.edges.remove(e)

    verts = []
    for np_ in new_pos:
        if np_[0] == "corner":
            verts.append(np_[1])
            continue
        co, vals = np_
        v = bm.verts.new(co)
        for k, l in sk_layers.items():
            v[l] = vals[k]
        verts.append(v)
    basis = bm.verts.layers.shape.get("Basis")

    new_faces = []
    for f in tpl.faces:
        nf = bm.faces.new([verts[i] for i in f])
        nf.material_index = mat
        nf.smooth = smooth
        for k, val in int_layers.items():
            nf[bm.faces.layers.int[k]] = val
        for k, l in uv_layers.items():
            for lp in nf.loops:
                t = find_tri(lp.vert.co.to_2d())
                lp[l].uv = bary_uv(t, lp.vert.co, t[2][k])
        new_faces.append(nf)
    if basis is not None:
        for v in verts:
            if v not in keep:
                v[basis] = v.co.copy()
    for nf in new_faces:
        nf.normal_update()
    return {"deleted_faces": old_faces, "created_faces": len(new_faces),
            "created_verts": n - len(corners),
            "new_verts": [v for v in verts if v not in keep]}


def sync_seams(bm, islands_faces, angle_deg, touched, tol=1e-4):
    """Average the AC9_3D_Project position of seam vertices shared between
    islands: two outline segments (of different islands) whose end corners
    coincide in 3D and that have the same vertex count are one seam.

    Only seams with at least one side in `touched` (vertices this run
    created) are changed, so the rest of the garment is left as it was.
    Returns (seams_synced, max_gap_before)."""
    lay = bm.verts.layers.shape.get(PROJECT_SK)
    if lay is None:
        return 0, 0.0
    sides = []
    for k, faces in enumerate(islands_faces):
        loop = _boundary_loop(faces)
        if loop is None:
            continue
        for s in _segments(loop, angle_deg):
            sides.append((k, s))
    done = set()
    n_synced = 0
    max_gap = 0.0
    dist = lambda a, b: (a[lay] - b[lay]).length
    for a, (ka, sa) in enumerate(sides):
        for b, (kb, sb) in enumerate(sides):
            if b <= a or ka == kb or len(sa) != len(sb) or (a, b) in done:
                continue
            if not any(v in touched for v in sa[1:-1]) and \
                    not any(v in touched for v in sb[1:-1]):
                continue
            if dist(sa[0], sb[0]) < tol and dist(sa[-1], sb[-1]) < tol:
                pair = sb
            elif dist(sa[0], sb[-1]) < tol and dist(sa[-1], sb[0]) < tol:
                pair = sb[::-1]
            else:
                continue
            done.add((a, b))
            for x, y in zip(sa, pair):
                max_gap = max(max_gap, dist(x, y))
                m = (x[lay] + y[lay]) / 2
                x[lay] = m
                y[lay] = m.copy()
            n_synced += 1
    return n_synced, max_gap


# --------------------------------------------------------------- operator ---

class AC9_OT_InstanceIsland(bpy.types.Operator):
    """Copy the active island's topology onto every island with the same
    outline (or onto the other selected islands).

    Select a vertex on the island you trust and run. With one island selected,
    every congruent island (same-handed or reflected) in the mesh becomes a
    target; with several islands selected, only those are, and the active
    vertex's island is the source. Each target keeps its own outline: the
    source's boundary vertices are re-laid along it corner to corner.

    Destructive on the targets. Ctrl+Z reverts.
    """

    bl_idname = "ac9_cloth.instance_island"
    bl_label = "Instance Island"
    bl_description = (
        "Copy the selected island's topology onto every island with the same "
        "outline (or onto the other selected islands). Each target keeps its "
        "own outline; the source's edge divisions are re-laid along it. "
        "Destructive on the targets. Undo (Ctrl+Z) to revert"
    )
    bl_options = {'REGISTER', 'UNDO'}

    tolerance: FloatProperty(
        name="Outline Tolerance",
        description="How different an outline may be and still count as the "
                    "same piece, as RMS distance relative to the piece size",
        default=0.03, min=0.001, max=0.5)
    include_mirrored: BoolProperty(
        name="Include Mirrored",
        description="Also copy onto reflected (left/right) copies of the piece",
        default=True)
    corner_angle: FloatProperty(
        name="Corner Angle",
        description="Outline turns sharper than this split the outline into "
                    "segments that are matched corner to corner",
        default=45.0, min=10.0, max=150.0)
    sync_seams: BoolProperty(
        name="Sync Seams",
        description="Snap the 3D position (AC9_3D_Project) of seam vertices "
                    "shared with neighbouring islands so Finalize can weld them",
        default=True)

    @classmethod
    def poll(cls, context):
        ob = context.active_object
        if ob is None or ob.type != 'MESH' or ob.mode != 'EDIT':
            cls.poll_message_set("Edit Mode on a mesh, with a vertex selected on the source island.")
            return False
        return True

    def execute(self, context):
        ob = context.active_object
        me = ob.data
        bm = bmesh.from_edit_mesh(me)
        bm.verts.ensure_lookup_table()

        islands = _islands(bm)
        owner = {}
        for k, faces in enumerate(islands):
            for f in faces:
                for v in f.verts:
                    owner[v] = k
        sel_islands = {owner[v] for v in bm.verts if v.select and v in owner}
        if not sel_islands:
            self.report({'ERROR'}, "Select a vertex on the source island first.")
            return {'CANCELLED'}
        active = bm.select_history.active
        if isinstance(active, bmesh.types.BMVert) and active in owner:
            src_k = owner[active]
        elif len(sel_islands) == 1:
            src_k = next(iter(sel_islands))
        else:
            self.report({'ERROR'}, "Several islands are selected — make a vertex "
                                   "on the source island the active one.")
            return {'CANCELLED'}

        src_faces = islands[src_k]
        src_loop = _boundary_loop(src_faces)
        if src_loop is None:
            self.report({'ERROR'}, "The source island's outline is not a single loop.")
            return {'CANCELLED'}
        src_pts = [v.co.to_2d() for v in src_loop]
        src_area = sum(f.calc_area() for f in src_faces)

        if len(sel_islands) > 1:
            cand = sorted(sel_islands - {src_k})
        else:
            cand = [k for k in range(len(islands)) if k != src_k]

        targets = []
        rejected = 0
        for k in cand:
            area = sum(f.calc_area() for f in islands[k])
            if len(sel_islands) == 1 and abs(area - src_area) > src_area * 0.15:
                continue
            loop = _boundary_loop(islands[k])
            if loop is None:
                rejected += 1
                continue
            err, mirrored = _outline_match(src_pts, [v.co.to_2d() for v in loop])
            if err > self.tolerance or (mirrored and not self.include_mirrored):
                rejected += 1
                continue
            targets.append((k, loop, mirrored))
        if not targets:
            self.report({'WARNING'}, "No island with the same outline was found "
                                     f"(tolerance {self.tolerance:.3f}).")
            return {'CANCELLED'}

        try:
            tpls = {m: _Template.from_island(src_faces, src_loop, self.corner_angle, m)
                    for m in {t[2] for t in targets}}
        except RuntimeError as e:
            self.report({'ERROR'}, f"Source: {e}")
            return {'CANCELLED'}

        sk_layers = {k: l for k, l in bm.verts.layers.shape.items() if k != "Basis"}
        uv_layers = dict(bm.loops.layers.uv.items())
        done, failed = 0, []
        touched_idx = set()
        created = []
        for k, loop, mirrored in targets:
            try:
                st = _rebuild_target(bm, tpls[mirrored], islands[k], loop,
                                     self.corner_angle, sk_layers, uv_layers)
                created += st["new_verts"]
                done += 1
            except RuntimeError as e:
                failed.append(str(e))
        bm.normal_update()
        bm.verts.index_update()
        touched_idx = {v.index for v in created}
        bmesh.update_edit_mesh(me, loop_triangles=True, destructive=True)

        proj_msg = ""
        top = getattr(context.scene, "ac9_cloth_retopo", None)
        if (top is not None and top.retopo_obj is ob and top.guide_obj is not None
                and top.guide_flat_shapekey):
            from ..clo_projector import core as projector_core
            bpy.ops.object.mode_set(mode='OBJECT')
            try:
                proj = projector_core.run_forward_projection(
                    context, ob, top.guide_obj, top.guide_flat_shapekey, incremental=True)
                proj_msg = f" Projected {proj.new_verts_computed} vert(s), {proj.failed} failed."
            finally:
                bpy.ops.object.mode_set(mode='EDIT')

        seam_msg = ""
        if self.sync_seams:
            bm = bmesh.from_edit_mesh(ob.data)
            bm.verts.ensure_lookup_table()
            touched = {bm.verts[i] for i in touched_idx if i < len(bm.verts)}
            n_sync, gap = sync_seams(bm, _islands(bm), self.corner_angle, touched)
            bmesh.update_edit_mesh(ob.data)
            if n_sync:
                seam_msg = f" Synced {n_sync} seam(s) (max gap {gap * 1000:.1f} mm)."

        msg = f"Instanced onto {done} island(s)."
        if failed:
            msg += f" {len(failed)} skipped: {failed[0]}."
        msg += proj_msg + seam_msg
        if hasattr(top, "status_faces"):
            top.status_faces = msg
        self.report({'WARNING'} if failed else {'INFO'}, msg)
        return {'FINISHED'}
