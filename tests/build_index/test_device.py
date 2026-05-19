from unittest.mock import patch

from ceo_build_index.main import _resolve_device


def test_explicit_device_returned_unchanged():
    """cpu, cuda, and mps are passed through as-is without any hardware check."""
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("cuda") == "cuda"
    assert _resolve_device("mps") == "mps"


def test_auto_prefers_cuda():
    """When both CUDA and MPS are reported available, cuda is returned."""
    with patch("ceo_build_index.main.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.backends.mps.is_available.return_value = True
        assert _resolve_device("auto") == "cuda"


def test_auto_falls_back_to_mps_when_no_cuda():
    """When CUDA is unavailable and MPS is available, mps is returned."""
    with patch("ceo_build_index.main.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        assert _resolve_device("auto") == "mps"


def test_auto_falls_back_to_cpu_when_no_gpu():
    """When neither CUDA nor MPS is available, cpu is returned."""
    with patch("ceo_build_index.main.torch") as mock_torch:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        assert _resolve_device("auto") == "cpu"
