#!/usr/bin/env python3
"""Compose a Franka arm URDF with a dexterous hand URDF for Isaac Gym.

The repo keeps generated robot assets out of git. This script builds a local
combined asset directory from an existing DOMINO/RoboTwin Franka tree and a
local hand URDF tree, for example a BrainCo Revo2 URDF.
"""

from __future__ import annotations

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable


def _parse_vec3(values: Iterable[str], name: str) -> str:
    vals = [float(v) for v in values]
    if len(vals) != 3:
        raise ValueError(f"{name} must contain exactly 3 values")
    return " ".join(f"{v:.9g}" for v in vals)


def _parse_color(values: Iterable[str] | None, name: str) -> tuple[float, float, float, float] | None:
    if values is None:
        return None
    vals = [float(v) for v in values]
    if len(vals) != 4:
        raise ValueError(f"{name} must contain exactly 4 rgba values")
    return tuple(vals)


def _format_color(color: tuple[float, float, float, float]) -> str:
    return " ".join(f"{max(0.0, min(1.0, v)):.4g}" for v in color)


def _package_name(package_root: Path) -> str | None:
    package_xml = package_root / "package.xml"
    if not package_xml.exists():
        return None
    package = ET.parse(package_xml).getroot()
    name_elem = package.find("name")
    return name_elem.text.strip() if name_elem is not None and name_elem.text else None


def _infer_package_root(urdf_path: Path, package_root_arg: str | None) -> Path:
    if package_root_arg:
        return Path(package_root_arg).expanduser().resolve()
    for parent in [urdf_path.parent, *urdf_path.parents]:
        if (parent / "package.xml").exists():
            return parent
    return urdf_path.parent


def _rewrite_mesh_filenames(
    root: ET.Element,
    prefix_dir: str,
    package_name: str | None = None,
) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith("package://"):
            package_path = filename[len("package://") :]
            if "/" in package_path:
                mesh_package, package_rel_path = package_path.split("/", 1)
                if package_name is None or mesh_package == package_name:
                    mesh.set("filename", f"{prefix_dir}/{package_rel_path}")
                    continue
            raise ValueError(
                f"Unsupported package mesh path {filename}; expected package "
                f"{package_name!r}. Pass --hand-package-root if needed."
            )
        if filename.startswith("file://") or Path(filename).is_absolute():
            continue
        mesh.set("filename", f"{prefix_dir}/{filename}")


def _set_visual_material_colors(
    root: ET.Element,
    main_color: tuple[float, float, float, float] | None,
    touch_color: tuple[float, float, float, float] | None,
) -> None:
    if main_color is None and touch_color is None:
        return
    for link in root.findall("link"):
        link_name = link.attrib.get("name", "").lower()
        color = touch_color if touch_color is not None and "touch" in link_name else main_color
        if color is None:
            continue
        for visual in link.findall("visual"):
            material = visual.find("material")
            if material is None:
                material = ET.SubElement(visual, "material", name="")
            color_elem = material.find("color")
            if color_elem is None:
                color_elem = ET.SubElement(material, "color")
            color_elem.set("rgba", _format_color(color))


def _remove_links_and_descendants(root: ET.Element, link_names: set[str]) -> None:
    parent_to_child_links: dict[str, list[str]] = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue
        parent_name = parent.attrib.get("link")
        child_name = child.attrib.get("link")
        if parent_name and child_name:
            parent_to_child_links.setdefault(parent_name, []).append(child_name)

    to_remove = set(link_names)
    stack = list(link_names)
    while stack:
        parent = stack.pop()
        for child in parent_to_child_links.get(parent, []):
            if child not in to_remove:
                to_remove.add(child)
                stack.append(child)

    for elem in list(root):
        if elem.tag == "link" and elem.attrib.get("name") in to_remove:
            root.remove(elem)
        elif elem.tag == "joint":
            parent = elem.find("parent")
            child = elem.find("child")
            parent_name = parent.attrib.get("link") if parent is not None else None
            child_name = child.attrib.get("link") if child is not None else None
            if parent_name in to_remove or child_name in to_remove:
                root.remove(elem)


def _clear_link_visual_collision(root: ET.Element, link_names: set[str]) -> None:
    for link in root.findall("link"):
        if link.attrib.get("name") not in link_names:
            continue
        for child in list(link):
            if child.tag in {"visual", "collision"}:
                link.remove(child)


def _ensure_material(root: ET.Element, name: str, rgba: str) -> None:
    for material in root.findall("material"):
        if material.attrib.get("name") == name:
            return
    material = ET.Element("material", name=name)
    ET.SubElement(material, "color", rgba=rgba)
    root.insert(0, material)


def _replace_panda_hand_with_inspire_adapter(root: ET.Element) -> None:
    link = root.find("./link[@name='panda_hand']")
    if link is None:
        raise ValueError("Cannot add adapter: link 'panda_hand' was not found")

    _ensure_material(root, "franka_custom_adapter_gray", "0.35 0.35 0.35 1.0")

    for child in list(link):
        if child.tag in {"inertial", "visual", "collision"}:
            link.remove(child)

    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0.0292", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.035")
    ET.SubElement(
        inertial,
        "inertia",
        ixx="0.000006",
        ixy="0",
        ixz="0",
        iyy="0.000006",
        iyz="0",
        izz="0.000008",
    )

    adapter_cylinders = [
        ("0 0 0.004", "0.008", "0.034"),
        ("0 0 0.0292", "0.0424", "0.018"),
        ("0 0 0.0544", "0.008", "0.028"),
    ]
    for xyz, length, radius in adapter_cylinders:
        visual = ET.SubElement(link, "visual")
        ET.SubElement(visual, "origin", xyz=xyz, rpy="0 0 0")
        geometry = ET.SubElement(visual, "geometry")
        ET.SubElement(geometry, "cylinder", length=length, radius=radius)
        ET.SubElement(visual, "material", name="franka_custom_adapter_gray")

    for xyz, length, radius in adapter_cylinders:
        collision = ET.SubElement(link, "collision")
        ET.SubElement(collision, "origin", xyz=xyz, rpy="0 0 0")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "cylinder", length=length, radius=radius)


def _prefix_hand_names(root: ET.Element, prefix: str) -> None:
    link_names = [link.attrib["name"] for link in root.findall("link")]
    joint_names = [joint.attrib["name"] for joint in root.findall("joint")]
    link_map = {name: f"{prefix}{name}" for name in link_names}
    joint_map = {name: f"{prefix}{name}" for name in joint_names}

    for link in root.findall("link"):
        link.set("name", link_map[link.attrib["name"]])
    for joint in root.findall("joint"):
        joint.set("name", joint_map[joint.attrib["name"]])
        parent = joint.find("parent")
        child = joint.find("child")
        mimic = joint.find("mimic")
        if parent is not None and parent.attrib.get("link") in link_map:
            parent.set("link", link_map[parent.attrib["link"]])
        if child is not None and child.attrib.get("link") in link_map:
            child.set("link", link_map[child.attrib["link"]])
        if mimic is not None and mimic.attrib.get("joint") in joint_map:
            mimic.set("joint", joint_map[mimic.attrib["joint"]])


def _root_link_name(root: ET.Element) -> str:
    links = {link.attrib["name"] for link in root.findall("link")}
    child_links = set()
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None and child.attrib.get("link"):
            child_links.add(child.attrib["link"])
    roots = sorted(links - child_links)
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one hand root link, got {roots}")
    return roots[0]


def build_asset(args: argparse.Namespace) -> None:
    franka_urdf = Path(args.franka_urdf).expanduser().resolve()
    hand_urdf = Path(args.hand_urdf).expanduser().resolve()
    hand_package_root = _infer_package_root(hand_urdf, args.hand_package_root)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_urdf = output_dir / args.output_name

    if not franka_urdf.exists():
        raise FileNotFoundError(franka_urdf)
    if not hand_urdf.exists():
        raise FileNotFoundError(hand_urdf)
    if not hand_package_root.exists():
        raise FileNotFoundError(hand_package_root)
    try:
        hand_urdf.relative_to(hand_package_root)
    except ValueError as exc:
        raise ValueError(
            f"hand URDF {hand_urdf} must be inside hand package root "
            f"{hand_package_root}"
        ) from exc
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shutil.copytree(franka_urdf.parent, output_dir / "franka")
    shutil.copytree(hand_package_root, output_dir / "hand")

    franka_root = ET.parse(franka_urdf).getroot()
    hand_root = ET.parse(hand_urdf).getroot()
    franka_root.set("name", args.robot_name)

    if args.strip_franka_fingers:
        _remove_links_and_descendants(
            franka_root,
            {"panda_leftfinger", "panda_rightfinger"},
        )
    if args.franka_hand_adapter_style == "inspire_cylinders":
        _replace_panda_hand_with_inspire_adapter(franka_root)
    elif args.strip_franka_hand_geometry:
        _clear_link_visual_collision(franka_root, {"panda_hand"})
    if args.strip_franka_wrist_camera_geometry:
        _clear_link_visual_collision(franka_root, {"camera_base", "camera"})

    hand_package_name = _package_name(hand_package_root)
    _rewrite_mesh_filenames(franka_root, "franka")
    _rewrite_mesh_filenames(hand_root, "hand", package_name=hand_package_name)
    _set_visual_material_colors(
        hand_root,
        main_color=args.hand_visual_color,
        touch_color=args.hand_touch_visual_color,
    )
    _prefix_hand_names(hand_root, args.hand_prefix)

    prefixed_hand_root = _root_link_name(hand_root)
    mount_joint = ET.Element("joint", name=args.mount_joint_name, type="fixed")
    ET.SubElement(mount_joint, "origin", xyz=args.mount_xyz, rpy=args.mount_rpy)
    ET.SubElement(mount_joint, "parent", link=args.mount_parent)
    ET.SubElement(mount_joint, "child", link=prefixed_hand_root)
    franka_root.append(mount_joint)

    for child in list(hand_root):
        franka_root.append(child)

    tree = ET.ElementTree(franka_root)
    ET.indent(tree, space="  ")
    tree.write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {output_urdf}")
    print(f"Set robotAssetRoot: {output_dir}")
    print(f"Set asset.robot: {args.output_name}")
    print(f"Copied hand package root: {hand_package_root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--franka-urdf",
        default="/data1/linsixu/DOMINO/assets/embodiments/franka-panda/panda.urdf",
    )
    parser.add_argument(
        "--hand-urdf",
        required=True,
        help="Path to the dexterous hand URDF to mount on panda_hand.",
    )
    parser.add_argument(
        "--hand-package-root",
        default=None,
        help=(
            "Root directory for the hand package. If omitted, the nearest parent "
            "with package.xml is used; otherwise the URDF directory is copied."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="/data1/linsixu/simtoolreal/assets/generated/franka_custom_hand",
    )
    parser.add_argument("--output-name", default="franka_custom_hand.urdf")
    parser.add_argument("--robot-name", default="franka_custom_hand")
    parser.add_argument("--hand-prefix", default="hand_")
    parser.add_argument("--mount-parent", default="panda_hand")
    parser.add_argument("--mount-joint-name", default="hand_mount_joint")
    parser.add_argument("--mount-xyz", nargs=3, default=["0", "0", "0.02"])
    parser.add_argument("--mount-rpy", nargs=3, default=["0", "0", "0"])
    parser.add_argument(
        "--hand-visual-color",
        nargs=4,
        default=None,
        metavar=("R", "G", "B", "A"),
        help="Optional RGBA override for hand visual materials.",
    )
    parser.add_argument(
        "--hand-touch-visual-color",
        nargs=4,
        default=None,
        metavar=("R", "G", "B", "A"),
        help="Optional RGBA override for links whose name contains 'touch'.",
    )
    parser.add_argument("--strip-franka-fingers", action="store_true", default=True)
    parser.add_argument("--keep-franka-fingers", dest="strip_franka_fingers", action="store_false")
    parser.add_argument(
        "--franka-hand-adapter-style",
        choices=["none", "inspire_cylinders"],
        default="none",
    )
    parser.add_argument("--strip-franka-hand-geometry", action="store_true")
    parser.add_argument("--strip-franka-wrist-camera-geometry", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.mount_xyz = _parse_vec3(args.mount_xyz, "mount_xyz")
    args.mount_rpy = _parse_vec3(args.mount_rpy, "mount_rpy")
    args.hand_visual_color = _parse_color(args.hand_visual_color, "hand_visual_color")
    args.hand_touch_visual_color = _parse_color(
        args.hand_touch_visual_color,
        "hand_touch_visual_color",
    )
    build_asset(args)


if __name__ == "__main__":
    main()
