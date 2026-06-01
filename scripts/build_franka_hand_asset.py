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


def _rewrite_mesh_filenames(root: ET.Element, prefix_dir: str) -> None:
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if filename.startswith(("package://", "file://")) or Path(filename).is_absolute():
            continue
        mesh.set("filename", f"{prefix_dir}/{filename}")


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
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_urdf = output_dir / args.output_name

    if not franka_urdf.exists():
        raise FileNotFoundError(franka_urdf)
    if not hand_urdf.exists():
        raise FileNotFoundError(hand_urdf)
    if output_dir.exists():
        if not args.force:
            raise FileExistsError(f"{output_dir} exists; pass --force to overwrite")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    shutil.copytree(franka_urdf.parent, output_dir / "franka")
    shutil.copytree(hand_urdf.parent, output_dir / "hand")

    franka_root = ET.parse(franka_urdf).getroot()
    hand_root = ET.parse(hand_urdf).getroot()
    franka_root.set("name", args.robot_name)

    if args.strip_franka_fingers:
        _remove_links_and_descendants(
            franka_root,
            {"panda_leftfinger", "panda_rightfinger"},
        )

    _rewrite_mesh_filenames(franka_root, "franka")
    _rewrite_mesh_filenames(hand_root, "hand")
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
    parser.add_argument("--strip-franka-fingers", action="store_true", default=True)
    parser.add_argument("--keep-franka-fingers", dest="strip_franka_fingers", action="store_false")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.mount_xyz = _parse_vec3(args.mount_xyz, "mount_xyz")
    args.mount_rpy = _parse_vec3(args.mount_rpy, "mount_rpy")
    build_asset(args)


if __name__ == "__main__":
    main()
