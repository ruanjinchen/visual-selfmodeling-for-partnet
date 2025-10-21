"""
- 指定 --anno-id，导出同一物体（单一外观）的多姿态 mesh 与对应 joint。
- 不导出颜色；mesh 仅几何。支持 obj 或 ply。
- 自动检测 DOF（revolute/prismatic），robot_state.json 写入“全部维度”，每维为 [pos, 0.0]（速度置 0）。
- 默认至少 1000 姿态（--num 可调）；每个姿态另存 angles_{i}.json 便于快速检查。
- 可选 --emit-xyzn：同时导出 mesh_{i}.xyzn，内容为均匀表面采样点+法向（6 列），可直接给原仓库训练。

兼容性说明：
- 如果后续要直接用 visual-self-modeling-main 训练，请：
  在 configs 里设 dof=你的真实 DOF；
"""

import os, re, csv, json, math, argparse, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import pybullet as p
import pybullet_data

# ----------------- 工具函数（CSV / anno_id） -----------------

def read_index_csv(index_csv: Path) -> List[Dict[str, str]]:
    rows = []
    with index_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def choose_anno_id_from_row(row: Dict[str, str]) -> str:
    # 尽量复用你现有脚本的判定逻辑习惯
    cand = [row.get("anno_id") or row.get("meta.anno_id"),
            row.get("model_id"), row.get("id")]
    md = row.get("model_dir")
    if md:
        try: cand.append(Path(md).name)
        except Exception: pass
    ur = row.get("urdf_relpath")
    if ur:
        try: cand.append(Path(ur).parts[0])
        except Exception: pass
    for c in cand:
        if c and str(c).strip():
            return str(c).strip()
    return "unknown"

def find_row_by_anno_id(rows: List[Dict[str,str]], anno_id: str) -> Optional[Dict[str,str]]:
    for r in rows:
        rid = choose_anno_id_from_row(r)
        if str(rid) == str(anno_id):
            return r
    return None

# ----------------- 关节/姿态采样 -----------------

def joint_type_name(jtype: int) -> str:
    return {
        p.JOINT_REVOLUTE: "revolute",
        p.JOINT_PRISMATIC: "prismatic",
        p.JOINT_PLANAR: "planar",
        p.JOINT_FIXED: "fixed"
    }.get(jtype, f"type_{jtype}")

def find_joints(body_id: int,
                allow_types=("revolute","prismatic"),
                name_regex: Optional[str]=None) -> List[Dict[str,object]]:
    allow = set(t.strip().lower() for t in allow_types)
    pattern = re.compile(name_regex) if name_regex else None
    out = []
    for j in range(p.getNumJoints(body_id)):
        info = p.getJointInfo(body_id, j)
        jtype = info[2]
        name  = info[1].decode('utf-8','ignore')
        tname = joint_type_name(jtype).lower()
        if tname not in allow: continue
        if pattern and not pattern.fullmatch(name): continue
        lower, upper = float(info[8]), float(info[9])
        if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
            if jtype == p.JOINT_REVOLUTE:  lower, upper = -math.pi, math.pi
            elif jtype == p.JOINT_PRISMATIC: lower, upper = -0.5, 0.5
            else: continue
        out.append({"index": int(j), "name": name, "type": tname, "lower": float(lower), "upper": float(upper)})
    return out

def sample_joint_targets_uniform(joints: List[Dict[str,object]], num: int, seed: int) -> List[List[float]]:
    rng = random.Random(seed)
    picks = []
    for _ in range(num):
        picks.append([rng.uniform(j["lower"], j["upper"]) for j in joints])
    return picks

# ----------------- Mesh 组装（无颜色） -----------------

def quaternion_to_matrix(q):
    R = np.array(p.getMatrixFromQuaternion(q), dtype=np.float64).reshape(3,3)
    T = np.eye(4); T[:3,:3] = R; return T

def pose_to_matrix(pos, orn):
    T = quaternion_to_matrix(orn); T[:3,3] = np.array(pos, dtype=np.float64); return T

def load_visual_mesh(shape, urdf_dir: Path) -> Optional[trimesh.Trimesh]:
    """
    只加载与变换视觉几何，不处理颜色/纹理。
    shape: getVisualShapeData() 的返回一项
    """
    body_uid, link_index, geom_type, dims, filename, lv_pos, lv_orn = shape[:7]
    mesh = None
    if geom_type == p.GEOM_MESH and filename:
        raw = filename.decode('utf-8','ignore') if isinstance(filename,(bytes,bytearray)) else str(filename)
        pth = Path(raw)
        candidates = [pth if pth.is_absolute() else urdf_dir / pth, urdf_dir.parent / pth]
        try: candidates.append(Path(pybullet_data.getDataPath()) / pth.name)
        except Exception: pass
        found = None
        for c in candidates:
            if c.exists(): found = c; break
        if found is None: return None
        mesh = trimesh.load(found, force='mesh', skip_missing=True, process=False)
        # URDF 里可能有 dimensions 缩放
        d = np.array(dims, dtype=float).reshape(-1)
        try:
            if d.size == 3 and not np.allclose(d, 1.0):
                mesh.apply_scale(d)
            elif d.size == 1 and not np.isclose(d[0], 1.0):
                mesh.apply_scale([d[0], d[0], d[0]])
        except Exception:
            pass
    elif geom_type == p.GEOM_BOX:
        d = np.array(dims, dtype=float).reshape(-1)
        if d.size >= 3: mesh = trimesh.creation.box(extents=2.0*d[:3])
    elif geom_type == p.GEOM_SPHERE:
        r = float(dims[0]) if len(dims)>=1 else 0.05
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=r)
    elif geom_type == p.GEOM_CYLINDER:
        if len(dims)>=2:
            r,h = float(dims[0]), float(dims[1])
            mesh = trimesh.creation.cylinder(radius=r, height=h, sections=48)
    elif geom_type == p.GEOM_CAPSULE:
        if len(dims)>=2:
            r,h = float(dims[0]), float(dims[1])
            mesh = trimesh.creation.capsule(radius=r, height=h, count=[24,24])

    if mesh is None or mesh.is_empty:
        return None

    # 应用本地视觉变换
    T_local = quaternion_to_matrix(lv_orn); T_local[:3,3] = np.array(lv_pos, dtype=np.float64)
    mesh.apply_transform(T_local)
    return mesh

def get_link_world_T(body_id: int) -> Dict[int, np.ndarray]:
    link_world_T = {}
    bpos, born = p.getBasePositionAndOrientation(body_id)
    link_world_T[-1] = pose_to_matrix(bpos, born)
    for li in range(p.getNumJoints(body_id)):
        st = p.getLinkState(body_id, li, computeForwardKinematics=1)
        if len(st)>=6 and st[4] is not None and st[5] is not None:
            pos, orn = st[4], st[5]
        else:
            pos, orn = st[0], st[1]
        link_world_T[li] = pose_to_matrix(pos, orn)
    return link_world_T

def world_mesh_pieces(body_id: int, urdf_dir: Path) -> List[trimesh.Trimesh]:
    pieces: List[trimesh.Trimesh] = []
    vdata = p.getVisualShapeData(body_id) or []
    link_world_T = get_link_world_T(body_id)
    for shape in vdata:
        link_idx = int(shape[1])
        m = load_visual_mesh(shape, urdf_dir)
        if m is None or m.is_empty: continue
        T_world = link_world_T.get(link_idx, np.eye(4))
        m.apply_transform(T_world)
        # 去掉颜色/纹理：强制只保留几何
        m = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, process=False)
        pieces.append(m)
    return pieces

def concatenate_mesh(pieces: List[trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
    if not pieces: return None
    try:
        cat = trimesh.util.concatenate(pieces)
        # 再次确保无视觉属性
        return trimesh.Trimesh(vertices=cat.vertices, faces=cat.faces, process=False)
    except Exception:
        # 兜底：手动拼接
        verts = []; faces = []; base = 0
        for m in pieces:
            if m.is_empty: continue
            verts.append(m.vertices)
            faces.append(m.faces + base)
            base += len(m.vertices)
        if not verts: return None
        V = np.vstack(verts); F = np.vstack(faces)
        return trimesh.Trimesh(vertices=V, faces=F, process=False)

# ----------------- 姿态执行与稳定 -----------------

def settle_to_targets(body_id: int, joints: List[Dict[str,object]],
                      targets: List[float],
                      timestep=1.0/240.0, motor_force=50.0,
                      pos_tol=1e-4, vel_eps=1e-3,
                      stable_hold_steps=30, max_steps=2400):
    # 清零所有关节速度控制
    for j in range(p.getNumJoints(body_id)):
        p.setJointMotorControl2(body_id, j, controlMode=p.VELOCITY_CONTROL, force=0.0)
    # 位置控制到目标
    for j, tgt in zip(joints, targets):
        p.setJointMotorControl2(body_id, int(j["index"]),
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=float(tgt),
                                force=motor_force)
    p.setTimeStep(timestep)
    stable = 0
    for _ in range(int(max_steps)):
        p.stepSimulation()
        ok = True
        for j, tgt in zip(joints, targets):
            pos, vel, *_ = p.getJointState(body_id, int(j["index"]))
            if abs(pos - tgt) > pos_tol or abs(vel) > vel_eps:
                ok = False; break
        if ok:
            stable += 1
            if stable >= stable_hold_steps:
                break
        else:
            stable = 0

# ----------------- xyzn 导出（可选） -----------------

def sample_xyzn_on_surface(mesh: trimesh.Trimesh, n_points: int = 120_000) -> np.ndarray:
    """
    采样 n_points 个表面点，并用面法向作为点法向。返回 (N,6) 数组。
    """
    if mesh.is_empty: return np.zeros((0,6), dtype=np.float32)
    import trimesh.sample as ts
    pts, fidx = ts.sample_surface(mesh, n_points)
    norms = mesh.face_normals[fidx]
    xyzn = np.hstack([pts.astype(np.float32), norms.astype(np.float32)])
    return xyzn

def write_xyzn(path: Path, xyzn: np.ndarray):
    path.write_text("\n".join(" ".join(map(str, row.tolist())) for row in xyzn), encoding="utf-8")

# ----------------- 主流程 -----------------

def main():
    ap = argparse.ArgumentParser(description="Export mesh + full-dof joints for one anno_id (no colors).")
    ap.add_argument("--index", type=Path, required=True, help="CSV 索引，需含 urdf_relpath 等列（与你现有流程一致）")
    ap.add_argument("--dataset-dir", type=Path, required=True, help="PartNet-Mobility 根目录（包含 URDF 与网格）")
    ap.add_argument("--anno-id", type=str, required=True, help="只导出该 anno_id 的模型（同一外观）")
    ap.add_argument("--out-dir", type=Path, default=Path("./saved_meshes"), help="输出目录（兼容原仓库命名）")
    ap.add_argument("--num", type=int, default=1000, help="生成姿态数量（至少 1000）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--joint-types", type=str, default="revolute,prismatic", help="允许的关节类型")
    ap.add_argument("--joint-regex", type=str, default=r".*", help="关节名正则（默认全部）")
    ap.add_argument("--mesh-format", type=str, default="obj", choices=["obj","ply"], help="mesh 输出格式（不含颜色）")
    ap.add_argument("--normalize", type=str, default="unit_sphere", choices=["unit_sphere","none"],
                    help="mesh 顶点是否平移到质心并等比缩放到单位球")
    ap.add_argument("--emit-xyzn", action="store_true", help="同时导出 mesh_{i}.xyzn（表面采样点+法向）")
    ap.add_argument("--xyzn-points", type=int, default=120_000, help="xyzn 采样点数（emit-xyzn 时有效）")
    ap.add_argument("--scale-prismatic-to-pi", action="store_true",
                    help="把平移关节按其 [min,max] 映射到 [-π,π]（便于原仓库里后续除以 π 到 [-1,1]）")
    args = ap.parse_args()

    rows = read_index_csv(args.index)
    row = find_row_by_anno_id(rows, args.anno_id)
    if row is None:
        raise SystemExit(f"[ERROR] anno_id={args.anno_id} 未在 CSV 中找到。")

    urdf_rel = row.get("urdf_relpath") or row.get("urdf_path")
    if not urdf_rel:
        raise SystemExit("[ERROR] CSV 里没有 urdf_relpath 列（或为空）。")
    urdf_path = (args.dataset_dir / urdf_rel).resolve()
    if not urdf_path.exists():
        raise SystemExit(f"[ERROR] URDF 不存在：{urdf_path}")

    out_dir: Path = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 连接物理引擎并加载模型
    if not p.isConnected(): p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setAdditionalSearchPath(str(args.dataset_dir))
    p.setGravity(0,0,0); p.setTimeStep(1.0/240.0)

    try:
        flags = p.URDF_USE_INERTIA_FROM_FILE
        bid = p.loadURDF(str(urdf_path), useFixedBase=1, flags=flags)
    except Exception as e:
        raise SystemExit(f"[ERROR] 加载 URDF 失败：{e}")

    joints = find_joints(bid,
                         allow_types=[t.strip() for t in args.joint_types.split(",") if t.strip()],
                         name_regex=args.joint_regex)
    if not joints:
        raise SystemExit("[ERROR] 未发现可动关节（revolute/prismatic）。")

    dof = len(joints)
    print(f"[INFO] anno_id={args.anno_id}, DOF={dof}, joints={[j['name'] for j in joints]}")

    rng = random.Random(args.seed)
    combos = sample_joint_targets_uniform(joints, max(1, int(args.num)), seed=args.seed)

    # 元信息+robot_state 汇总
    meta = {
        "anno_id": args.anno_id,
        "urdf": str(urdf_path),
        "dof": dof,
        "joints": [
            {"index": int(j["index"]), "name": j["name"], "type": j["type"],
             "limit_lower": float(j["lower"]), "limit_upper": float(j["upper"])}
            for j in joints
        ],
        "num_poses": len(combos),
        "normalize": args.normalize,
        "mesh_format": args.mesh_format,
        "emit_xyzn": bool(args.emit_xyzn),
        "xyzn_points": int(args.xyzn_points)
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    robot_state: Dict[str, List[List[float]]] = {}

    urdf_dir = urdf_path.parent

    ok = 0
    for i, angles in enumerate(combos):
        # 1) 让模型到位并稳定
        settle_to_targets(bid, joints, angles,
                          timestep=1.0/240.0, motor_force=50.0,
                          pos_tol=1e-4, vel_eps=1e-3,
                          stable_hold_steps=30, max_steps=2400)

        # 2) 组装世界坐标 mesh（无颜色），可选归一化
        pieces = world_mesh_pieces(bid, urdf_dir)
        if not pieces:
            print(f"[WARN] 姿态 {i} 未得到 mesh，跳过。"); continue
        mesh = concatenate_mesh(pieces)
        if mesh is None or mesh.is_empty:
            print(f"[WARN] 姿态 {i} mesh 为空，跳过。"); continue

        if args.normalize == "unit_sphere":
            ctr = mesh.vertices.mean(axis=0)
            V = mesh.vertices - ctr
            radius = np.max(np.linalg.norm(V, axis=1))
            if radius > 1e-9:
                mesh = trimesh.Trimesh(vertices=(V / radius), faces=mesh.faces, process=False)

        # 3) 导出几何 mesh（无颜色/无材质）
        if args.mesh_format == "obj":
            # trimesh 默认可能写出 mtl，这里强制纯几何：
            exported = trimesh.exchange.obj.export_obj(
                mesh,
                include_normals=True, include_texture=False, return_texture=False
            )
            (out_dir / f"mesh_{i}.obj").write_text(exported, encoding="utf-8")
        else:  # ply
            mesh.export(out_dir / f"mesh_{i}.ply")  # 默认几何-only

        # 4) 可选导出 xyzn（表面点+法向）
        if args.emit_xyzn:
            xyzn = sample_xyzn_on_surface(mesh, n_points=int(args.xyzn_points))
            write_xyzn(out_dir / f"mesh_{i}.xyzn", xyzn)

        # 5) 保存该姿态的角度（便于快速检查）
        (out_dir / f"angles_{i}.json").write_text(json.dumps({"angles": [float(a) for a in angles]}, indent=2, ensure_ascii=False), encoding="utf-8")

        # 6) 汇总到 robot_state（全维）
        full_q = []
        for j, q in zip(joints, angles):
            if j["type"] == "prismatic" and args.scale_prismatic_to_pi:
                # 把平移从 [min,max] 线性映射到 [-π, π]
                l, u = float(j["lower"]), float(j["upper"])
                t = 0.0 if (u <= l) else ((float(q) - l) / (u - l) * 2.0 - 1.0)
                q_scaled = float(t * math.pi)
                full_q.append([q_scaled, 0.0])
            else:
                full_q.append([float(q), 0.0])
        robot_state[str(i)] = full_q
        ok += 1

    (out_dir / "robot_state.json").write_text(json.dumps(robot_state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] 有效姿态 {ok}/{len(combos)} 已写入：{out_dir}")

    # 额外：给原仓库的 train/test 划分（按 9:1 随机）
    ids = list(map(int, [k for k in robot_state.keys()]))
    ids.sort()
    rng.shuffle(ids)
    n = len(ids); n_tr = int(0.9 * n)
    split = {"train": ids[:n_tr], "test": ids[n_tr:]}
    assets_dir = (out_dir.parent / "assets" / "datainfo")
    try:
        assets_dir.mkdir(parents=True, exist_ok=True)
        (assets_dir / "multiple_models_data_split_dict_1.json").write_text(
            json.dumps(split, indent=2), encoding="utf-8"
        )
        print(f"[INFO] 训练/测试划分已写入：{assets_dir/'multiple_models_data_split_dict_1.json'}")
    except Exception as e:
        print(f"[WARN] 划分文件写入失败：{e}")

    if p.isConnected():
        try: p.disconnect()
        except Exception: pass

if __name__ == "__main__":
    main()
'''
[用CSV索引和数据集根目录，选定一个 anno_id，导出至少 1200 姿态]
python make_partnet_dataset_for_vsm.py `
  --index partnet_index.csv `
  --dataset-dir D:/Dataset/PartNet/dataset `
  --anno-id 2612 `
  --out-dir ./VSM/eyeglasses_2612 `
  --num 1200 `
  --mesh-format obj `
  --normalize unit_sphere `
  --emit-xyzn `
  --xyzn-points 120000 `
  --scale-prismatic-to-pi

'''