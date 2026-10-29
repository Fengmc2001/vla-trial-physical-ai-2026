"""EGL device identity checks, called ONLY after resource admission.

Importing this module loads no graphics library. Selection enumerates EGL
devices with ctypes; it creates no EGL display, surface, context or CUDA
context, imports neither Genesis nor torch, and changes no environment.

EGL_CUDA_DEVICE_NV is a CUDA device HANDLE, not an NVML physical ordinal.
It is retained for diagnostics only; it is never guessed into a PCI mapping.
For NVIDIA, an already initialized CUDA driver resolves PCI -> physical UUID,
and EGL's persistent UUID selects the matching NVIDIA driver device. This
module never calls cuInit or any CUDA/EGL context creation function.
References:
https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_drm.txt
https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_drm_render_node.txt
https://registry.khronos.org/EGL/extensions/NV/EGL_NV_device_cuda.txt
https://registry.khronos.org/EGL/extensions/EXT/EGL_EXT_device_persistent_id.txt
"""
from __future__ import annotations

import ctypes
import ctypes.util
import re
import sys
import uuid
from pathlib import Path
from typing import Any


EGL_EXTENSIONS = 0x3055
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377
EGL_CUDA_DEVICE_NV = 0x323A
EGL_DEVICE_UUID_EXT = 0x335C
EGL_DRIVER_NAME_EXT = 0x335E
_MAX_DEVICES = 64
_PCI = re.compile(r"(?:(?P<domain>[0-9a-fA-F]{4,8}):)?(?P<bus>[0-9a-fA-F]{2}):(?P<device>[0-9a-fA-F]{2})\.(?P<function>[0-7])\Z")
_DRM_NODE = re.compile(r"/dev/dri/(?P<name>card[0-9]+|renderD[0-9]+)\Z")


class GraphicsBindingError(RuntimeError):
    """Physical device identity cannot be proved; refuse renderer startup."""


def normalize_pci_bus_id(value: str) -> str:
    """Normalize e.g. NVIDIA 00000000:47:00.0 to Linux 0000:47:00.0."""
    if not isinstance(value, str):
        raise GraphicsBindingError("PCI bus ID must be a string.")
    match = _PCI.fullmatch(value.strip())
    if match is None:
        raise GraphicsBindingError(f"Invalid PCI bus ID: {value!r}")
    domain = int(match.group("domain") or "0000", 16)
    bus, device, function = (int(match.group(key), 16) for key in ("bus", "device", "function"))
    if device > 0x1F:
        raise GraphicsBindingError("PCI device number exceeds 0x1f.")
    return f"{domain:04x}:{bus:02x}:{device:02x}.{function:x}"


def pci_from_drm_node(node: str, *, sysfs_root: str | Path = "/sys/class/drm") -> dict[str, str]:
    """Resolve one kernel DRM node to its direct PCI parent, without opening it."""
    if not isinstance(node, str) or (match := _DRM_NODE.fullmatch(node)) is None:
        raise GraphicsBindingError("EGL returned a nonstandard DRM device node.")
    device_link = Path(sysfs_root) / match.group("name") / "device"
    try:
        resolved = device_link.resolve(strict=True)
        if not resolved.is_dir():
            raise OSError("DRM device target is not a directory")
    except (OSError, RuntimeError) as exc:
        raise GraphicsBindingError(f"Cannot resolve DRM identity for {node}.") from exc
    try:
        pci = normalize_pci_bus_id(resolved.name)
    except GraphicsBindingError as exc:
        raise GraphicsBindingError(f"DRM node {node} does not resolve directly to a PCI device.") from exc
    return {"node": node, "sysfs_device": str(resolved), "pci_bus_id": pci}


def _decode(value: bytes | None) -> str | None:
    if value is None:
        return None
    try:
        return value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise GraphicsBindingError("EGL returned an invalid UTF-8 device string.") from exc


def _cuda_identity_for_pci(pci_bus_id: str) -> dict[str, Any]:
    """Query admitted, already initialized CUDA driver metadata; create nothing.

    The driver must have been initialized by admission under the permitted
    CUDA_VISIBLE_DEVICES value. A not-initialized or invisible-device error is
    fatal; this helper must never initialize CUDA or widen device visibility.
    """
    expected = normalize_pci_bus_id(pci_bus_id)
    try:
        library = ctypes.CDLL(ctypes.util.find_library("cuda") or "libcuda.so.1")
        get_device = library.cuDeviceGetByPCIBusId
        get_device.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_char_p]
        get_device.restype = ctypes.c_int
        get_pci = library.cuDeviceGetPCIBusId
        get_pci.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        get_pci.restype = ctypes.c_int
        get_uuid = getattr(library, "cuDeviceGetUuid_v2", None) or library.cuDeviceGetUuid
        get_uuid.argtypes = [ctypes.c_void_p, ctypes.c_int]
        get_uuid.restype = ctypes.c_int
    except (OSError, AttributeError) as exc:
        raise GraphicsBindingError("Cannot query the admitted CUDA driver's device identity.") from exc
    device = ctypes.c_int()
    code = get_device(ctypes.byref(device), expected.encode("ascii"))
    if code:
        raise GraphicsBindingError(f"CUDA PCI identity lookup failed with code {code}; no initialization or fallback attempted.")
    pci = ctypes.create_string_buffer(32)
    code = get_pci(pci, len(pci), device.value)
    if code or normalize_pci_bus_id(pci.value.decode("ascii")) != expected:
        raise GraphicsBindingError("CUDA PCI identity failed its round-trip verification.")
    binary = (ctypes.c_ubyte * 16)()
    code = get_uuid(binary, device.value)
    if code or not any(binary):
        raise GraphicsBindingError(f"CUDA physical UUID query failed with code {code}.")
    return {"pci_bus_id": expected, "device_uuid": str(uuid.UUID(bytes=bytes(binary))),
            "cuda_device_handle": device.value, "proof": "cuDeviceGetByPCIBusId -> cuDeviceGetPCIBusId -> cuDeviceGetUuid"}


def _enumerate_egl_devices() -> list[dict[str, Any]]:
    """Load libEGL and query device identities only. Never create a context."""
    if not sys.platform.startswith("linux"):
        raise GraphicsBindingError("This DRM/PCI binding implementation requires Linux.")
    try:
        library_name = ctypes.util.find_library("EGL") or "libEGL.so.1"
        library = ctypes.CDLL(library_name)
        get_proc = library.eglGetProcAddress
        get_proc.argtypes = [ctypes.c_char_p]
        get_proc.restype = ctypes.c_void_p
    except (OSError, AttributeError) as exc:
        raise GraphicsBindingError("Cannot load libEGL device-query entry points.") from exc

    def procedure(name: bytes, result_type: Any, *arg_types: Any, optional: bool = False) -> Any:
        address = get_proc(name)
        if not address:
            if optional:
                return None
            raise GraphicsBindingError(f"Required EGL extension function unavailable: {name.decode()}")
        return ctypes.CFUNCTYPE(result_type, *arg_types)(address)

    query_devices = procedure(
        b"eglQueryDevicesEXT", ctypes.c_uint32,
        ctypes.c_int32, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int32),
    )
    query_string = procedure(b"eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int32)
    query_attrib = procedure(
        b"eglQueryDeviceAttribEXT", ctypes.c_uint32,
        ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_ssize_t), optional=True,
    )
    query_binary = procedure(
        b"eglQueryDeviceBinaryEXT", ctypes.c_uint32, ctypes.c_void_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), optional=True,
    )
    count = ctypes.c_int32()
    if not query_devices(0, None, ctypes.byref(count)) or not 1 <= count.value <= _MAX_DEVICES:
        raise GraphicsBindingError("EGL device count is unavailable, zero or outside the bounded limit.")
    capacity = count.value
    handles = (ctypes.c_void_p * capacity)()
    returned = ctypes.c_int32()
    if not query_devices(capacity, handles, ctypes.byref(returned)) or returned.value != capacity:
        raise GraphicsBindingError("EGL device enumeration failed or changed during selection.")
    records = []
    seen_handles: set[int] = set()
    for index, handle in enumerate(handles):
        if handle is None or handle in seen_handles:
            raise GraphicsBindingError("EGL returned an empty or duplicate device handle.")
        seen_handles.add(handle)
        extensions = _decode(query_string(handle, EGL_EXTENSIONS)) or ""
        extension_names = set(extensions.split())
        primary = None
        render = None
        cuda_handle = None
        driver_name = None
        device_uuid = None
        if "EGL_EXT_device_drm" in extension_names:
            primary = _decode(query_string(handle, EGL_DRM_DEVICE_FILE_EXT))
        if "EGL_EXT_device_drm_render_node" in extension_names:
            render = _decode(query_string(handle, EGL_DRM_RENDER_NODE_FILE_EXT))
        if query_attrib is not None and "EGL_NV_device_cuda" in extension_names:
            value = ctypes.c_ssize_t()
            if query_attrib(handle, EGL_CUDA_DEVICE_NV, ctypes.byref(value)):
                cuda_handle = int(value.value)
        # Query the persistent identity only for NVIDIA's device entry. On
        # mixed libglvnd systems Mesa also advertises inaccessible DRM devices;
        # asking Mesa for its driver/UUID can try opening those render nodes.
        if "EGL_NV_device_cuda" in extension_names and "EGL_EXT_device_persistent_id" in extension_names:
            driver_name = _decode(query_string(handle, EGL_DRIVER_NAME_EXT))
            if query_binary is not None:
                binary = (ctypes.c_ubyte * 16)()
                size = ctypes.c_int32()
                if query_binary(handle, EGL_DEVICE_UUID_EXT, 16, binary, ctypes.byref(size)) and size.value == 16 and any(binary):
                    device_uuid = str(uuid.UUID(bytes=bytes(binary)))
        records.append({
            "egl_index": index, "extensions": extensions,
            "drm_device": primary, "drm_render_node": render,
            "cuda_device_handle": cuda_handle,
            "driver_name": driver_name, "device_uuid": device_uuid,
        })
    return records


def select_egl_device(pci_bus_id: str) -> dict[str, Any]:
    """Find a verified EGL index for the admitted GPU, without creating context.

    The caller must already have completed resource admission. This function
    neither grants execution nor sets PYOPENGL_PLATFORM/EGL_DEVICE_ID. Use its
    returned index in the same process before importing the renderer, and do
    not change EGL vendor/device enumeration settings between these steps.
    On NVIDIA, match the CUDA driver's PCI/UUID proof against EGL's persistent
    physical UUID and NVIDIA driver name. Never select a second Mesa driver for
    the same board merely because its DRM node maps to the expected PCI bus.
    Pure DRM systems retain their kernel identity path. A bare CUDA handle is
    never sufficient evidence.
    """
    expected_pci = normalize_pci_bus_id(pci_bus_id)
    records = _enumerate_egl_devices()
    nvidia_records = [record for record in records if "EGL_NV_device_cuda" in record["extensions"].split()]
    cuda_proof = _cuda_identity_for_pci(expected_pci) if nvidia_records else None
    matches = []
    unmapped = []
    for record in records:
        proofs = []
        errors = []
        for key in ("drm_device", "drm_render_node"):
            node = record.get(key)
            if not node:
                continue
            try:
                proofs.append(pci_from_drm_node(node))
            except GraphicsBindingError as exc:
                errors.append(str(exc))
        identities = {proof["pci_bus_id"] for proof in proofs}
        if len(identities) > 1:
            raise GraphicsBindingError(f"EGL device {record['egl_index']} has conflicting DRM PCI identities.")
        if cuda_proof is not None:
            if record not in nvidia_records:
                continue
            if record.get("driver_name") != "nvidia" or record.get("device_uuid") != cuda_proof["device_uuid"]:
                unmapped.append({"egl_index": record["egl_index"], "driver_name": record.get("driver_name"), "device_uuid": record.get("device_uuid")})
                continue
            if identities and expected_pci not in identities:
                raise GraphicsBindingError("Matching EGL UUID and DRM PCI identity conflict.")
            matches.append({**record, "pci_bus_id": expected_pci, "proof": "CUDA PCI -> physical UUID == EGL persistent UUID; NVIDIA driver",
                            "cuda_identity_proof": cuda_proof, "drm_proofs": proofs, "mapping_warnings": errors})
        elif not identities:
            unmapped.append({"egl_index": record["egl_index"], "cuda_device_handle": record.get("cuda_device_handle"), "errors": errors})
        elif expected_pci in identities:
            matches.append({
                **record, "pci_bus_id": expected_pci, "proof": "EGL device string -> kernel DRM sysfs device -> PCI bus ID",
                "drm_proofs": proofs, "mapping_warnings": errors,
            })
    if len(matches) != 1:
        raise GraphicsBindingError(
            f"Expected one EGL device for PCI {expected_pci}; found {len(matches)}. "
            f"Unmapped devices: {unmapped}. CUDA handles are not treated as NVML physical ordinals."
        )
    return {
        **matches[0], "enumerated_device_count": len(records),
        "context_created": False, "cuda_initialized": False,
        "required_environment": {"PYOPENGL_PLATFORM": "egl", "EGL_DEVICE_ID": str(matches[0]["egl_index"])},
    }
