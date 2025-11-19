import os
import math
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pybullet as p
import pybullet_data
import trimesh
import open3d as o3d


# ----------------------------
# 基础工具
# ----------------------------

def set_seed(seed: int):
    np.random.seed(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def aabb_center_radius_from_vertices(verts: np.ndarray) -> Tuple[np.ndarray, float]:
    if verts.size == 0:
        return np.zeros(3, dtype=np.float32), 1.0
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    ctr = (vmin + vmax) * 0.5
    rad = float(np.max(vmax - vmin) * 0.5)
    return ctr.astype(np.float32), max(rad, 1e-6)


# ----------------------------
# PyBullet & 几何转换
# ----------------------------

def setup_bullet(additional_paths: List[Path]):
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    for ap in additional_paths:
        p.setAdditionalSearchPath(str(ap))


def quat_to_mat(q: Tuple[float, float, float, float]) -> np.ndarray:
    x, y, z, w = q
    x2, y2, z2 = x + x, y + y, z + z
    xx = x * x2; yy = y * y2; zz = z * z2
    xy = x * y2; xz = x * z2; yz = z * z2
    wx = w * x2; wy = w * y2; wz = w * z2
    return np.array([
        [1.0 - (yy + zz), xy - wz,        xz + wy],
        [xy + wz,         1.0 - (xx + zz),yz - wx],
        [xz - wy,         yz + wx,        1.0 - (xx + yy)]
    ], dtype=np.float32)


def pose_to_mat(pos, orn) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = quat_to_mat(orn)
    M[:3, 3] = np.array(pos, dtype=np.float32)
    return M


def look_at_cam2world(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """相机->世界，坐标系 +Z 前, +X 右, +Y 上"""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-12)
    r = np.cross(f, up); r = r / (np.linalg.norm(r) + 1e-12)
    u = np.cross(r, f)
    T = np.eye(4, dtype=np.float32)
    T[:3, 0] = r; T[:3, 1] = u; T[:3, 2] = f; T[:3, 3] = eye
    return T


# ----------------------------
# 关键：URDF 目录 + mesh 路径解析
# ----------------------------

def auto_find_urdf(urdf_dir: Path) -> Path:
    """优先 mobility.urdf；否则取该目录下第一份 *.urdf"""
    cand = urdf_dir / "mobility.urdf"
    if cand.is_file():
        return cand.resolve()
    urdfs = sorted(urdf_dir.glob("*.urdf"))
    if not urdfs:
        raise FileNotFoundError(f"在 {urdf_dir} 未找到 URDF 文件")
    return urdfs[0].resolve()


def resolve_mesh_path(fname: str, urdf_dir: Path) -> Optional[Path]:
    """
    逐级尝试把 URDF 返回的 meshAssetFileName 解析为真实文件：
    1) 原样（可能是绝对/相对）；
    2) urdf_dir / fname；
    3) 逐段后缀：urdf_dir / parts[i:]；（应对重复前缀如 dataset\\102074\\dataset\\102074\\...）
    4) 在 urdf_dir 递归按 basename 搜索；若多于一个，再按“路径后缀匹配”精确筛选。
    """
    try_paths = []

    p0 = Path(fname)
    try_paths.append(p0)
    try_paths.append((urdf_dir / p0))

    parts = p0.parts
    for i in range(len(parts)):
        try_paths.append(urdf_dir.joinpath(*parts[i:]))

    for c in try_paths:
        if c.is_file():
            return c.resolve()

    # 最后兜底：按 basename 搜索
    hits = list(urdf_dir.rglob(p0.name))
    if len(hits) == 1:
        return hits[0].resolve()
    if len(hits) > 1:
        suffix = p0.as_posix()
        for h in hits:
            if h.as_posix().endswith(suffix):
                return h.resolve()
    return None


# ----------------------------
# 加载刚体 & 拼装当前姿态网格
# ----------------------------

def load_body_from_urdf_dir(urdf_dir: Path) -> Tuple[int, Path]:
    """只需给 URDF 所在文件夹；自动找到 URDF 并加载。"""
    urdf_path = auto_find_urdf(urdf_dir)
    body_id = p.loadURDF(
        str(urdf_path),
        basePosition=[0, 0, 0],
        useFixedBase=True,
        flags=p.URDF_MERGE_FIXED_LINKS | p.URDF_USE_SELF_COLLISION
    )
    return body_id, urdf_path


def get_dof_and_limits(body_id: int):
    dof, joint_indices, limits = 0, [], []
    n = p.getNumJoints(body_id)
    for j in range(n):
        info = p.getJointInfo(body_id, j)
        jtype = info[2]
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            dof += 1
            joint_indices.append(j)
            limits.append((jtype, float(info[8]), float(info[9])))
    return dof, joint_indices, limits


def sample_and_apply_joint_state(body_id: int, joint_indices, limits, rng: np.random.Generator):
    q_out = []
    for (jid, (jtype, ll, ul)) in zip(joint_indices, limits):
        q = float(rng.uniform(ll, ul))
        p.resetJointState(body_id, jid, targetValue=q)
        q_out.append(q)
    return q_out


def trimesh_from_visual_shape(geom_type, dims, filename: Optional[Path]) -> Optional[trimesh.Trimesh]:
    if geom_type == p.GEOM_MESH and filename is not None:
        try:
            mesh = trimesh.load(str(filename), force='mesh', process=False)
        except Exception:
            return None
        if mesh is not None and mesh.vertices.size > 0 and dims is not None:
            try:
                sx, sy, sz = float(dims[0]), float(dims[1]), float(dims[2])
                mesh.apply_scale([sx, sy, sz])
            except Exception:
                pass
        return mesh
    return None


def build_world_mesh_from_bullet(body_id: int, urdf_dir: Path) -> trimesh.Trimesh:
    """把当前关节状态下所有可视几何（世界位姿）拼成一个 Trimesh。"""
    base_pos, base_orn = p.getBasePositionAndOrientation(body_id)
    T_base = pose_to_mat(base_pos, base_orn)

    link_T = {}
    for j in range(p.getNumJoints(body_id)):
        st = p.getLinkState(body_id, j, computeForwardKinematics=True)
        link_T[j] = pose_to_mat(st[4], st[5])

    meshes = []
    vs_list = p.getVisualShapeData(body_id) or []
    for vs in vs_list:
        link_idx = vs[1]
        geom_type = vs[2]
        dims = vs[3]
        raw_name = vs[4].decode("utf-8") if isinstance(vs[4], (bytes, bytearray)) else vs[4]
        lpos = vs[5]
        lorn = vs[6]

        filename = None
        if raw_name:
            filename = resolve_mesh_path(raw_name, urdf_dir)
            if filename is None:
                continue

        tri = trimesh_from_visual_shape(geom_type, dims, filename)
        if tri is None or tri.vertices.size == 0:
            continue

        T_local = pose_to_mat(lpos, lorn)
        T_world = (T_base if link_idx == -1 else link_T.get(link_idx, np.eye(4, dtype=np.float32))) @ T_local
        tri = tri.copy()
        tri.apply_transform(T_world)
        meshes.append(tri)

    if len(meshes) == 0:
        return trimesh.Trimesh(vertices=np.zeros((0, 3), dtype=np.float32),
                               faces=np.zeros((0, 3), dtype=np.int64), process=False)

    merged = trimesh.util.concatenate(meshes)
    merged.remove_unreferenced_vertices()
    merged.remove_degenerate_faces()
    return merged


# ----------------------------
# 视角生成 & Raycasting → 外观点 + “指向相机”的法向
# ----------------------------

def safe_up_from_dir(dir_vec: np.ndarray) -> np.ndarray:
    """给定视线方向 dir（指向物体），构造一个稳定的 up 向量（与 dir 正交）。"""
    a = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(dir_vec, a))) > 0.95:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    r = np.cross(a, dir_vec)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        r = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        r = r / r_norm
    up = np.cross(dir_vec, r)
    up = up / (np.linalg.norm(up) + 1e-12)
    return up.astype(np.float32)


def build_views(center: np.ndarray,
                cam_R: float,
                extra_views: int,
                rng: np.random.Generator) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    返回 [(eye, up), ...]；前 6 个为 ±X/±Y/±Z，其后为随机视角。
    """
    ctr = np.asarray(center, dtype=np.float32)

    base_dirs = [
        np.array([+1, 0, 0], dtype=np.float32),
        np.array([-1, 0, 0], dtype=np.float32),
        np.array([0, +1, 0], dtype=np.float32),
        np.array([0, -1, 0], dtype=np.float32),
        np.array([0, 0, +1], dtype=np.float32),
        np.array([0, 0, -1], dtype=np.float32),
    ]

    views = []
    for d in base_dirs:
        eye = ctr + d * cam_R
        up = safe_up_from_dir(-d)  # 视线指向物体中心 => -d
        views.append((eye, up))

    if extra_views > 0:
        # 高斯归一化法：对单位球面均匀
        dirs = rng.normal(size=(extra_views, 3)).astype(np.float32)
        dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12)
        for d in dirs:
            eye = ctr + d * cam_R
            up = safe_up_from_dir(-d)
            views.append((eye, up))

    return views


def o3d_points_normals_from_multiview(mesh_legacy: o3d.geometry.TriangleMesh,
                                      center: np.ndarray, radius: float,
                                      width: int, height: int, fov_deg: float,
                                      cam_dist_mul: float,
                                      near: float, far: float,
                                      extra_views: int,
                                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 (points, normals)，法向统一为“指向相机”。
    """
    # 预计算三角面法向（兜底）
    mesh_legacy.compute_triangle_normals()
    tri_normals = np.asarray(mesh_legacy.triangle_normals, dtype=np.float32)

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(tmesh)

    ctr = np.asarray(center, dtype=np.float32)
    R = float(max(radius, 1e-6))
    cam_R = cam_dist_mul * R

    # 生成视角（固定 6 + 随机 extra）
    view_list = build_views(ctr, cam_R, extra_views, rng)

    all_pts = []
    all_nrm = []
    for eye, up_vec in view_list:
        rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            fov_deg=float(fov_deg),
            center=o3d.core.Tensor(ctr, o3d.core.Dtype.Float32),
            eye=o3d.core.Tensor(eye, o3d.core.Dtype.Float32),
            up=o3d.core.Tensor(up_vec, o3d.core.Dtype.Float32),
            width_px=int(width), height_px=int(height)
        )
        hits = scene.cast_rays(rays)
        t_hit = hits['t_hit'].numpy().reshape(-1)
        rays_np = rays.numpy().reshape(-1, 6)  # [ox,oy,oz, dx,dy,dz]

        valid = np.isfinite(t_hit) & (t_hit > 0)
        if near > 0:
            valid &= (t_hit >= near)
        if far < 1e8:
            valid &= (t_hit <= far)
        if not np.any(valid):
            continue

        o = rays_np[valid, 0:3]  # 相机位置
        d = rays_np[valid, 3:6]  # 光线方向（单位）
        t = t_hit[valid][:, None]
        pts = o + d * t

        # 法向：优先 primitive_normals / primitive_ids，兜底 -d
        normals = None
        try:
            if 'primitive_normals' in hits:
                n = hits['primitive_normals'].numpy().reshape(-1, 3)[valid]
                normals = n.astype(np.float32)
            elif 'primitive_ids' in hits:
                pid = hits['primitive_ids'].numpy().reshape(-1)[valid]
                pid = np.clip(pid, 0, len(tri_normals)-1).astype(np.int64)
                normals = tri_normals[pid]
        except Exception:
            normals = None

        if normals is None:
            normals = -d.astype(np.float32)

        # 统一指向相机
        to_cam = eye[None, :] - pts
        to_cam /= (np.linalg.norm(to_cam, axis=1, keepdims=True) + 1e-12)
        flip = (np.sum(normals * to_cam, axis=1) < 0)
        normals[flip] *= -1.0
        normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        all_pts.append(pts.astype(np.float32))
        all_nrm.append(normals.astype(np.float32))

    if len(all_pts) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    P = np.concatenate(all_pts, axis=0).astype(np.float32)
    N = np.concatenate(all_nrm, axis=0).astype(np.float32)
    return P, N


# ----------------------------
# 主流程
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    # 仅需 URDF 所在的文件夹
    ap.add_argument("--urdf-dir", type=str, required=True, help="包含 URDF 的文件夹路径（例如含 mobility.urdf）")

    # 输出 & 数量
    ap.add_argument("--out-dir", type=str, required=True, help="输出目录")
    ap.add_argument("--num", type=int, default=500, help="帧数/样本数")
    ap.add_argument("--seed", type=int, default=1)

    # Mesh 保存 & 缩放
    ap.add_argument("--mesh-format", type=str, default="obj", choices=["obj", "ply"], help="mesh 输出格式")
    ap.add_argument("--scale-factor", type=float, required=True,
                    help=">1：定义为 原始尺寸/缩放后尺寸 = scale-factor（即 mesh 等比缩小）")

    # 点云（带法向）
    ap.add_argument("--pcd-format", type=str, default="ply", choices=["ply", "xyz"], help="点云输出格式（ply 将包含 nx,ny,nz）")
    ap.add_argument("--pcd-points", type=int, default=20000, help="每帧点云点数")
    ap.add_argument("--pcd-ascii", action="store_true", help="PLY 以 ASCII 写出（默认二进制）")

    # 相机（多视角）
    ap.add_argument("--depth-width", type=int, default=320)
    ap.add_argument("--depth-height", type=int, default=240)
    ap.add_argument("--depth-fov", type=float, default=60.0)
    ap.add_argument("--cam-dist-mul", type=float, default=2.2, help="相机距离 = 此倍数 * 物体半径")
    ap.add_argument("--near-mul", type=float, default=0.05)
    ap.add_argument("--far-mul", type=float, default=6.0)
    ap.add_argument("--extra-views", type=int, default=0, help="额外随机相机数量（在球面上均匀随机），默认 0")

    args = ap.parse_args()
    if args.scale_factor <= 1.0:
        print("--scale-factor < 1; Mesh将被放大")

    set_seed(args.seed)
    urdf_dir = Path(args.urdf_dir).resolve()
    out_dir = ensure_dir(Path(args.out_dir).resolve())

    # 连接仿真并加载
    setup_bullet([urdf_dir])
    body_id, urdf_path = load_body_from_urdf_dir(urdf_dir)

    # DOF 信息
    dof, joint_indices, limits = get_dof_and_limits(body_id)

    # 元信息（便于复现）
    meta = {
        "urdf_dir": str(urdf_dir),
        "urdf_path": str(urdf_path),
        "dof": int(dof),
        "depth": {
            "views": int(6 + max(0, int(args.extra_views))),
            "fixed_views": 6,
            "extra_views": int(max(0, int(args.extra_views))),
            "width": int(args.depth_width),
            "height": int(args.depth_height),
            "fov_deg": float(args.depth_fov),
            "cam_dist_mul": float(args.cam_dist_mul),
            "near_mul": float(args.near_mul),
            "far_mul": float(args.far_mul),
        },
        "pipeline": "pybullet_visual_mesh + open3d_multiview_raycast (pcd with camera-oriented normals)",
        "scale_factor": float(args.scale_factor),
        "pcd_points": int(args.pcd_points),
        "pcd_format": args.pcd_format,
        "mesh_format": args.mesh_format,
        "seed": int(args.seed),
        "pcd_normals": "camera_oriented (raycast primitive normal if available else view_dir)"
    }
    robot_state = {}

    rng = np.random.default_rng(args.seed)

    for i in range(int(args.num)):
        # 1) 随机关节
        q = sample_and_apply_joint_state(body_id, joint_indices, limits, rng)
        robot_state[str(i)] = [[float(val), 0.0] for val in q]

        # 2) 拼装世界网格
        mesh_world = build_world_mesh_from_bullet(body_id, urdf_dir)
        if mesh_world.vertices.size == 0:
            # 占位
            pcd_path = out_dir / f"cloud_{i}.{args.pcd_format}"
            if args.pcd_format == "ply":
                o3d.io.write_point_cloud(str(pcd_path), o3d.geometry.PointCloud(), write_ascii=bool(args.pcd_ascii))
            else:
                np.savetxt(pcd_path, np.zeros((0, 3), dtype=np.float32), fmt="%.8f")
            print(f"[{i+1}/{args.num}] empty mesh, wrote empty cloud to {pcd_path.name}")
            continue

        # 3) 缩放（保存缩放后的 mesh）
        s = float(args.scale_factor)
        mesh_world_scaled = mesh_world.copy()
        mesh_world_scaled.apply_scale(1.0 / s)

        mesh_path = out_dir / f"mesh_{i}.{args.mesh_format}"
        mesh_world_scaled.export(mesh_path)

        # 4) 多视角射线 → 点 + 法向（指向相机）
        verts = mesh_world_scaled.vertices.view(np.ndarray).astype(np.float32)
        ctr, rad = aabb_center_radius_from_vertices(verts)

        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(verts.astype(np.float64)),
            triangles=o3d.utility.Vector3iVector(mesh_world_scaled.faces.astype(np.int32))
        )
        o3d_mesh.compute_vertex_normals()

        near = float(args.near_mul * rad)
        far = float(args.far_mul * rad)

        pts, nrms = o3d_points_normals_from_multiview(
            mesh_legacy=o3d_mesh,
            center=ctr, radius=rad,
            width=int(args.depth_width), height=int(args.depth_height),
            fov_deg=float(args.depth_fov),
            cam_dist_mul=float(args.cam_dist_mul),
            near=near, far=far,
            extra_views=int(max(0, int(args.extra_views))),
            rng=rng
        )

        # 5) 精确到指定点数（点与法向保持一一对应）
        N = int(args.pcd_points)
        if pts.shape[0] == 0:
            pts_fixed = np.zeros((0, 3), dtype=np.float32)
            nrm_fixed = np.zeros((0, 3), dtype=np.float32)
        elif pts.shape[0] >= N:
            idx = np.random.choice(pts.shape[0], size=N, replace=False)
            pts_fixed = pts[idx]
            nrm_fixed = nrms[idx]
        else:
            idx = np.random.choice(pts.shape[0], size=N, replace=True)
            pts_fixed = pts[idx]
            nrm_fixed = nrms[idx]

        # 6) 写出（PLY 写 nx,ny,nz；若选择 xyz 则仅写坐标）
        pcd_path = out_dir / f"cloud_{i}.{args.pcd_format}"
        if args.pcd_format == "ply":
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_fixed.astype(np.float64))
            pcd.normals = o3d.utility.Vector3dVector(nrm_fixed.astype(np.float64))
            o3d.io.write_point_cloud(str(pcd_path), pcd, write_ascii=bool(args.pcd_ascii))
        else:
            np.savetxt(pcd_path, pts_fixed.astype(np.float32), fmt="%.8f")

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[{i+1}/{args.num}] verts={len(mesh_world_scaled.vertices)} | pcd={pts_fixed.shape} | saved={mesh_path.name}, {pcd_path.name}")

    # 7) meta / robot_state
    save_json(out_dir / "meta.json", meta)
    save_json(out_dir / "robot_state.json", robot_state)

    # 8) 数据划分（列出点云文件名）
    ids = list(range(int(args.num)))
    rng.shuffle(ids)
    n_train = int(0.8 * len(ids))
    n_val = int(0.1 * len(ids))
    split = {
        "train": [f"cloud_{k}.{args.pcd_format}" for k in ids[:n_train]],
        "val":   [f"cloud_{k}.{args.pcd_format}" for k in ids[n_train:n_train+n_val]],
        "test":  [f"cloud_{k}.{args.pcd_format}" for k in ids[n_train+n_val:]],
    }
    split_path = ensure_dir(out_dir.parent / "assets" / "datainfo") / f"multiple_models_data_split_dict_{int(args.seed)}.json"
    save_json(split_path, split)

    print(f"\n[Done] 输出目录: {out_dir}")
    print(f"meta.json / robot_state.json 已写入；数据划分: {split_path}")


if __name__ == "__main__":
    main()
'''

python make_partnet_dataset_for_vsm.py \
  --urdf-dir data/102074 \
  --out-dir data/pliers_2074 \
  --num 500 \
  --scale-factor 1.6780842542648315 \
  --mesh-format obj \
  --pcd-format ply \
  --pcd-points 400000 \
  --depth-width 640 --depth-height 480 --depth-fov 75 \
  --cam-dist-mul 3.2 \
  --near-mul 0.05 --far-mul 6.0 \
  --extra-views 24 \
  --seed 1

python make_partnet_dataset_for_vsm.py \
  --urdf-dir data/10893 \
  --out-dir data/scissors_10893 \
  --num 500 \
  --scale-factor 0.9951514005661011 \
  --mesh-format obj \
  --pcd-format ply \
  --pcd-points 4000000 \
  --depth-width 1280 --depth-height 720 --depth-fov 75 \
  --cam-dist-mul 3.2 \
  --near-mul 0.05 --far-mul 6.0 \
  --extra-views 24 \
  --seed 1

python make_partnet_dataset_for_vsm.py \
  --urdf-dir data/101863 \
  --out-dir data/eyeglasses_101863 \
  --num 500 \
  --scale-factor 1.0354602336883545 \
  --mesh-format obj \
  --pcd-format ply \
  --pcd-points 4000000 \
  --depth-width 1280 --depth-height 720 --depth-fov 75 \
  --cam-dist-mul 3.2 \
  --near-mul 0.05 --far-mul 6.0 \
  --extra-views 24 \
  --seed 1


'''