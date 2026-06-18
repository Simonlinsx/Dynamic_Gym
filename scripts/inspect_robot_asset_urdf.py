#!/usr/bin/env python3
"""Print joint/link information for a generated Franka + hand URDF."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def _link_names(root: ET.Element) -> list[str]:
    return [link.attrib["name"] for link in root.findall("link")]


def _joint_entries(root: ET.Element) -> list[tuple[str, str, str, str]]:
    entries = []
    for joint in root.findall("joint"):
        name = joint.attrib.get("name", "")
        joint_type = joint.attrib.get("type", "")
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.attrib.get("link", "") if parent is not None else ""
        child_name = child.attrib.get("link", "") if child is not None else ""
        entries.append((name, joint_type, parent_name, child_name))
    return entries


def _root_links(links: list[str], joints: list[tuple[str, str, str, str]]) -> list[str]:
    children = {child for _, _, _, child in joints if child}
    return sorted(set(links) - children)


def _leaf_links(links: list[str], joints: list[tuple[str, str, str, str]]) -> list[str]:
    parents = {parent for _, _, parent, _ in joints if parent}
    return sorted(set(links) - parents)


def _interesting_links(links: list[str]) -> list[str]:
    tokens = ("tip", "distal", "thumb", "index", "middle", "ring", "pinky", "little")
    return [name for name in links if any(token in name.lower() for token in tokens)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("urdf", type=Path)
    parser.add_argument(
        "--hand-prefix",
        default="revo2_",
        help="Only summarize prefixed hand joints/links when possible.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = ET.parse(args.urdf).getroot()
    links = _link_names(root)
    joints = _joint_entries(root)
    active_joints = [entry for entry in joints if entry[1] not in {"fixed", "floating"}]
    prefixed_links = [name for name in links if name.startswith(args.hand_prefix)]
    prefixed_active_joints = [
        entry for entry in active_joints if entry[0].startswith(args.hand_prefix)
    ]
    summary_links = prefixed_links or links
    leaf_links = _leaf_links(summary_links, joints)

    print(f"urdf={args.urdf}")
    print(f"robot_name={root.attrib.get('name', '')}")
    print(f"num_links={len(links)}")
    print(f"num_joints={len(joints)}")
    print(f"num_active_joints={len(active_joints)}")
    print(f"num_prefixed_active_joints={len(prefixed_active_joints)}")
    print(f"root_links={_root_links(links, joints)}")
    print("active_joints:")
    for name, joint_type, parent, child in active_joints:
        print(f"  {name} [{joint_type}] {parent} -> {child}")
    print("candidate_hand_links:")
    for name in _interesting_links(summary_links):
        print(f"  {name}")
    print("candidate_leaf_links:")
    for name in leaf_links:
        print(f"  {name}")


if __name__ == "__main__":
    main()
