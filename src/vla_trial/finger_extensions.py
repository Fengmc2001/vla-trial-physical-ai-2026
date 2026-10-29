"""Generate a separate Panda MJCF with physical flat finger extensions.

The installed model is read-only. Only the robot finger bodies are modified in
the generated model; manipulated objects never receive a joint or attachment.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


PAD_WIDTH_M = 0.050
PAD_THICKNESS_M = 0.006
PAD_LENGTH_M = 0.060
PAD_HAND_Z_M = (0.110, 0.170)
PAD_DENSITY_KG_M3 = 1100.0
PAD_MASS_KG = PAD_WIDTH_M * PAD_THICKNESS_M * PAD_LENGTH_M * PAD_DENSITY_KG_M3


def _numbers(value: str, count: int) -> list[float]:
    parsed = [float(item) for item in value.split()]
    if len(parsed) != count or any(not math.isfinite(item) for item in parsed):
        raise ValueError("Unexpected Panda inertial or body coordinates.")
    return parsed


def _text(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _add_pad_inertia(body: ET.Element, center: list[float]) -> None:
    inertial = body.find("inertial")
    if inertial is None or "diaginertia" not in inertial.attrib or "quat" in inertial.attrib:
        raise ValueError("Expected the installed Panda finger's explicit axis-aligned diagonal inertia.")
    old_mass = float(inertial.attrib["mass"])
    old_center = _numbers(inertial.get("pos", "0 0 0"), 3)
    old_diagonal = _numbers(inertial.attrib["diaginertia"], 3)
    if not old_mass > 0 or any(value <= 0 for value in old_diagonal):
        raise ValueError("Finger inertia must be positive before adding the pad.")
    total_mass = old_mass + PAD_MASS_KG
    combined_center = [(old_mass * a + PAD_MASS_KG * b) / total_mass for a, b in zip(old_center, center)]
    dimensions = (PAD_WIDTH_M, PAD_THICKNESS_M, PAD_LENGTH_M)
    pad_diagonal = [PAD_MASS_KG * sum(dimensions[j] ** 2 for j in range(3) if j != i) / 12 for i in range(3)]
    matrix = [[(old_diagonal[i] + pad_diagonal[i]) if i == j else 0.0 for j in range(3)] for i in range(3)]
    for mass, origin in ((old_mass, old_center), (PAD_MASS_KG, center)):
        delta = [value - combined for value, combined in zip(origin, combined_center)]
        squared = sum(value * value for value in delta)
        for i in range(3):
            for j in range(3):
                matrix[i][j] += mass * ((squared if i == j else 0.0) - delta[i] * delta[j])
    inertial.set("mass", f"{total_mass:.12g}")
    inertial.set("pos", _text(combined_center))
    del inertial.attrib["diaginertia"]
    inertial.set("fullinertia", _text([matrix[0][0], matrix[1][1], matrix[2][2], matrix[0][1], matrix[0][2], matrix[1][2]]))


def build_extended_panda_model(source_xml: str | Path, cache_dir: str | Path) -> Path:
    """Return a content-addressed generated MJCF, leaving upstream bytes intact."""
    source = Path(source_xml).resolve(strict=True)
    original = source.read_bytes()
    root = ET.fromstring(original)
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Panda MJCF compiler element is missing.")
    mesh_dir = (source.parent / compiler.get("meshdir", "")).resolve()
    texture_dir = (source.parent / compiler.get("texturedir", "")).resolve()
    compiler.set("meshdir", str(mesh_dir))
    for element in root.findall("./asset/mesh"):
        if "file" in element.attrib:
            element.set("file", str((mesh_dir / element.attrib["file"]).resolve(strict=True)))
    for element in root.findall("./asset/texture"):
        if "file" in element.attrib:
            element.set("file", str((texture_dir / element.attrib["file"]).resolve(strict=True)))
    for name in ("left_finger", "right_finger"):
        body = root.find(f".//body[@name='{name}']")
        if body is None:
            raise ValueError(f"Required Panda finger body is missing: {name}.")
        body_origin = _numbers(body.get("pos", "0 0 0"), 3)
        if abs(body_origin[2] - 0.0584) > 1e-6:
            raise ValueError("Unexpected Panda finger mounting height; refusing guessed extension geometry.")
        center = [0.0, PAD_THICKNESS_M / 2, sum(PAD_HAND_Z_M) / 2 - body_origin[2]]
        half_size = [PAD_WIDTH_M / 2, PAD_THICKNESS_M / 2, PAD_LENGTH_M / 2]
        for role in ("visual", "collision"):
            attributes = {"name": f"{name}_flat_extension_{role}", "class": role,
                          "type": "box", "pos": _text(center), "size": _text(half_size)}
            if role == "visual":
                attributes.update({"rgba": "0.16 0.19 0.22 1", "contype": "0", "conaffinity": "0"})
            else:
                attributes.update({"contype": "1", "conaffinity": "1", "friction": "1 0.005 0.0001"})
            ET.SubElement(body, "geom", attributes)
        _add_pad_inertia(body, center)
    root.insert(0, ET.Comment(" Generated physical finger pads; upstream Panda SHA256=" + hashlib.sha256(original).hexdigest() + " "))
    ET.indent(root, space="  ")
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    digest = hashlib.sha256(payload).hexdigest()
    destination_dir = Path(cache_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"panda-flat-fingers-{digest[:20]}.xml"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError("Generated model cache collision; refusing to overwrite it.")
        return destination
    with tempfile.NamedTemporaryFile(dir=destination_dir, suffix=".tmp", delete=False) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, destination)
    return destination
