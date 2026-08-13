"""Unit tests for fluxflow.utils.device (ROCm/CUDA/MPS/CPU auto-detection)."""

import unittest.mock as mock

import torch

from fluxflow.utils.device import DeviceInfo, get_device, get_device_info, is_rocm, parse_device


class TestIsRocm:
    def test_true_when_hip_version_set(self):
        with mock.patch.object(torch.version, "hip", "6.2.0"):
            assert is_rocm() is True

    def test_false_when_hip_version_none(self):
        with mock.patch.object(torch.version, "hip", None):
            assert is_rocm() is False


class TestGetDevice:
    def test_cuda_wins_when_available(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert get_device() == torch.device("cuda")

    def test_mps_used_when_cuda_unavailable(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            assert get_device() == torch.device("mps")

    def test_cpu_used_when_neither_available(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert get_device() == torch.device("cpu")

    def test_cuda_wins_when_only_cuda_available(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert get_device() == torch.device("cuda")


class TestGetDeviceInfo:
    def test_rocm_backend_classification(self):
        with (
            mock.patch.object(torch.version, "hip", "6.2.0"),
            mock.patch("torch.cuda.get_device_name", return_value="AMD Radeon 8060S"),
        ):
            info = get_device_info(torch.device("cuda"))
        assert info.backend == "rocm"
        assert info.rocm_version == "6.2.0"
        assert info.device_name == "AMD Radeon 8060S"

    def test_real_cuda_backend_classification(self):
        with (
            mock.patch.object(torch.version, "hip", None),
            mock.patch("torch.cuda.get_device_name", return_value="NVIDIA A6000"),
        ):
            info = get_device_info(torch.device("cuda"))
        assert info.backend == "cuda"
        assert info.rocm_version is None
        assert info.device_name == "NVIDIA A6000"

    def test_mps_backend_classification(self):
        info = get_device_info(torch.device("mps"))
        assert info.backend == "mps"
        assert info.device_name == "Apple Silicon"
        assert info.rocm_version is None

    def test_cpu_backend_classification(self):
        info = get_device_info(torch.device("cpu"))
        assert info.backend == "cpu"
        assert info.device_name == ""
        assert info.rocm_version is None

    def test_device_name_failure_does_not_crash(self):
        with (
            mock.patch.object(torch.version, "hip", None),
            mock.patch("torch.cuda.get_device_name", side_effect=RuntimeError("no device")),
        ):
            info = get_device_info(torch.device("cuda"))
        assert info.device_name == ""

    def test_defaults_to_get_device_when_none_passed(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            info = get_device_info()
        assert info.device == torch.device("cpu")

    def test_str_formatting_with_rocm_version(self):
        info = DeviceInfo(
            device=torch.device("cuda"),
            backend="rocm",
            device_name="RX 8060S",
            rocm_version="6.2.0",
        )
        assert str(info) == "rocm (RX 8060S, ROCm 6.2.0)"

    def test_str_formatting_without_device_name(self):
        info = DeviceInfo(
            device=torch.device("cpu"), backend="cpu", device_name="", rocm_version=None
        )
        assert str(info) == "cpu"


class TestParseDevice:
    def test_auto_delegates_to_get_device(self):
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
        ):
            assert parse_device("auto") == torch.device("cpu")

    def test_explicit_device_string_passthrough(self):
        assert parse_device("cuda:0") == torch.device("cuda:0")

    def test_cpu_string_passthrough(self):
        assert parse_device("cpu") == torch.device("cpu")
