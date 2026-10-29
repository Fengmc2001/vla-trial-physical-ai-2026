"""CPU-only EGL identity tests: no graphics library is loaded or queried."""
from __future__ import annotations

import os
import ctypes
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from vla_trial.graphics_binding import GraphicsBindingError, _cuda_identity_for_pci, normalize_pci_bus_id, pci_from_drm_node, select_egl_device


class PCIParsingTests(unittest.TestCase):
    def test_nvidia_eight_digit_and_linux_domain_match(self):
        self.assertEqual(normalize_pci_bus_id("00000000:47:00.0"), "0000:47:00.0")
        self.assertEqual(normalize_pci_bus_id("0000:47:00.0"), "0000:47:00.0")
        self.assertEqual(normalize_pci_bus_id("47:00.0"), "0000:47:00.0")
        self.assertEqual(normalize_pci_bus_id("00000001:AB:1F.7"), "0001:ab:1f.7")

    def test_invalid_addresses_are_rejected(self):
        for value in (None, "", "GPU-123", "0000:47:20.0", "0000:47:00.8", "0000:47:00.0/../other", "0:47:00.0"):
            with self.subTest(value=value), self.assertRaises(GraphicsBindingError):
                normalize_pci_bus_id(value)

    def test_drm_symlink_maps_direct_pci_parent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            device = root / "devices" / "0000:47:00.0"
            device.mkdir(parents=True)
            node = root / "drm" / "card3"
            node.mkdir(parents=True)
            (node / "device").symlink_to(device)
            proof = pci_from_drm_node("/dev/dri/card3", sysfs_root=root / "drm")
            self.assertEqual(proof["pci_bus_id"], "0000:47:00.0")
            self.assertEqual(proof["sysfs_device"], str(device.resolve()))

    def test_missing_nonpci_and_nonstandard_drm_nodes_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "not-a-pci-device"
            target.mkdir()
            node = root / "card1"
            node.mkdir()
            (node / "device").symlink_to(target)
            for name in ("/dev/dri/card1", "/dev/dri/card2", "/dev/nvidia0", "/dev/dri/../card1"):
                with self.subTest(name=name), self.assertRaises(GraphicsBindingError):
                    pci_from_drm_node(name, sysfs_root=root)


class EGLSelectionTests(unittest.TestCase):
    UUID = "c742a480-2583-6162-9e4f-49979d1dbab6"

    @staticmethod
    def record(index, primary=None, render=None, cuda=None):
        return {"egl_index": index, "drm_device": primary, "drm_render_node": render, "cuda_device_handle": cuda, "extensions": "fixture"}

    @staticmethod
    def proof(node):
        mapping = {"/dev/dri/card8": "0000:47:00.0", "/dev/dri/renderD131": "0000:47:00.0", "/dev/dri/card2": "0000:17:00.0"}
        if node not in mapping:
            raise GraphicsBindingError("unmapped fixture")
        return {"node": node, "pci_bus_id": mapping[node], "sysfs_device": "/fixture/" + mapping[node]}

    def test_matches_pci_and_preserves_egl_enumeration_index(self):
        records = [self.record(0, "/dev/dri/card2"), self.record(1, "/dev/dri/card8", "/dev/dri/renderD131", cuda=3)]
        old_environment = dict(os.environ)
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=records), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof):
            selected = select_egl_device("00000000:47:00.0")
        self.assertEqual(selected["egl_index"], 1)
        self.assertEqual(selected["pci_bus_id"], "0000:47:00.0")
        self.assertEqual(selected["required_environment"]["EGL_DEVICE_ID"], "1")
        self.assertFalse(selected["context_created"])
        self.assertFalse(selected["cuda_initialized"])
        self.assertEqual(dict(os.environ), old_environment)

    def test_render_node_alone_is_sufficient_kernel_evidence(self):
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=[self.record(2, render="/dev/dri/renderD131")]), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof):
            self.assertEqual(select_egl_device("0000:47:00.0")["egl_index"], 2)

    def test_cuda_handle_never_assumed_to_equal_physical_gpu_index(self):
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=[self.record(3, cuda=3)]), self.assertRaisesRegex(GraphicsBindingError, "not treated as NVML"):
            select_egl_device("0000:47:00.0")

    def test_multiple_matching_devices_are_ambiguous(self):
        records = [self.record(0, "/dev/dri/card8"), self.record(1, render="/dev/dri/renderD131")]
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=records), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof), self.assertRaisesRegex(GraphicsBindingError, "found 2"):
            select_egl_device("0000:47:00.0")

    def test_primary_render_identity_conflict_is_rejected(self):
        records = [self.record(0, "/dev/dri/card2", "/dev/dri/renderD131")]
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=records), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof), self.assertRaisesRegex(GraphicsBindingError, "conflicting"):
            select_egl_device("0000:47:00.0")

    def test_wrong_pci_and_missing_mapping_are_rejected(self):
        for record in (self.record(0, "/dev/dri/card2"), self.record(0, "/dev/dri/card404")):
            with self.subTest(record=record), patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=[record]), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof), self.assertRaises(GraphicsBindingError):
                select_egl_device("0000:47:00.0")

    def nvidia(self, index, uuid=None, driver="nvidia", primary=None):
        return {**self.record(index, primary=primary, cuda=0), "extensions": "EGL_NV_device_cuda EGL_EXT_device_persistent_id",
                "driver_name": driver, "device_uuid": uuid or self.UUID}

    def test_nvidia_uuid_selects_driver_not_duplicate_mesa_device_or_ordinal(self):
        records = [self.record(3, "/dev/dri/card8"), self.nvidia(7)]
        proof = {"device_uuid": self.UUID, "pci_bus_id": "0000:47:00.0", "cuda_device_handle": 0}
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=records), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof), patch("vla_trial.graphics_binding._cuda_identity_for_pci", return_value=proof) as query:
            selected = select_egl_device("00000000:47:00.0")
        query.assert_called_once_with("0000:47:00.0")
        self.assertEqual(selected["egl_index"], 7)
        self.assertEqual(selected["cuda_device_handle"], 0)
        self.assertEqual(selected["cuda_identity_proof"], proof)

    def test_nvidia_identity_unavailable_never_falls_back_to_mesa(self):
        records = [self.record(12, "/dev/dri/card8"), self.nvidia(3)]
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=records), patch("vla_trial.graphics_binding._cuda_identity_for_pci", side_effect=GraphicsBindingError("not initialized")), self.assertRaisesRegex(GraphicsBindingError, "not initialized"):
            select_egl_device("0000:47:00.0")

    def test_nvidia_missing_or_wrong_uuid_and_vendor_fail_closed(self):
        for record in (self.nvidia(3, "00000000-0000-0000-0000-000000000001"), self.nvidia(3, driver="nouveau"), {**self.nvidia(3), "device_uuid": None}):
            with self.subTest(record=record), patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=[record]), patch("vla_trial.graphics_binding._cuda_identity_for_pci", return_value={"device_uuid": self.UUID}), self.assertRaisesRegex(GraphicsBindingError, "found 0"):
                select_egl_device("0000:47:00.0")

    def test_duplicate_nvidia_driver_uuid_is_ambiguous(self):
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=[self.nvidia(3), self.nvidia(7)]), patch("vla_trial.graphics_binding._cuda_identity_for_pci", return_value={"device_uuid": self.UUID}), self.assertRaisesRegex(GraphicsBindingError, "found 2"):
            select_egl_device("0000:47:00.0")

    def test_nvidia_uuid_drm_pci_conflict_is_rejected(self):
        with patch("vla_trial.graphics_binding._enumerate_egl_devices", return_value=[self.nvidia(3, primary="/dev/dri/card2")]), patch("vla_trial.graphics_binding.pci_from_drm_node", side_effect=self.proof), patch("vla_trial.graphics_binding._cuda_identity_for_pci", return_value={"device_uuid": self.UUID}), self.assertRaisesRegex(GraphicsBindingError, "conflict"):
            select_egl_device("0000:47:00.0")


class CUDAIdentityMetadataTests(unittest.TestCase):
    def library(self, lookup_code=0, returned_pci=b"0000:47:00.0"):
        def get_device(pointer, pci):
            ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int))[0] = 19
            return lookup_code
        def get_pci(buffer, size, device):
            buffer.value = returned_pci
            return 0
        def get_uuid(buffer, device):
            data = bytes.fromhex("c742a480258361629e4f49979d1dbab6")
            for index, value in enumerate(data):
                buffer[index] = value
            return 0
        return SimpleNamespace(cuInit=Mock(), cuDeviceGetByPCIBusId=Mock(side_effect=get_device), cuDeviceGetPCIBusId=Mock(side_effect=get_pci), cuDeviceGetUuid_v2=Mock(side_effect=get_uuid))

    def test_metadata_uses_real_handle_and_no_cuda_initialization(self):
        library = self.library()
        with patch("vla_trial.graphics_binding.ctypes.CDLL", return_value=library), patch("vla_trial.graphics_binding.ctypes.util.find_library", return_value="fixture"):
            result = _cuda_identity_for_pci("00000000:47:00.0")
        self.assertEqual(result["cuda_device_handle"], 19)
        self.assertEqual(result["device_uuid"], EGLSelectionTests.UUID)
        self.assertEqual(library.cuDeviceGetUuid_v2.call_args.args[1], 19)
        library.cuInit.assert_not_called()

    def test_uninitialized_cuda_and_roundtrip_mismatch_fail_closed(self):
        for library in (self.library(lookup_code=3), self.library(returned_pci=b"0000:20:00.0")):
            with patch("vla_trial.graphics_binding.ctypes.CDLL", return_value=library), patch("vla_trial.graphics_binding.ctypes.util.find_library", return_value="fixture"), self.assertRaises(GraphicsBindingError):
                _cuda_identity_for_pci("0000:47:00.0")
            library.cuInit.assert_not_called()
            library.cuDeviceGetUuid_v2.assert_not_called()


if __name__ == "__main__":
    unittest.main()
