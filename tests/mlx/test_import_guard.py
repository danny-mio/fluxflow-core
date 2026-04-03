import pytest


def test_import_guard_raises_without_mlx(monkeypatch):
    """Importing fluxflow.mlx without mlx installed raises ImportError."""
    import sys
    import builtins

    # Remove mlx and fluxflow.mlx from sys.modules to force re-import
    mlx_modules = [k for k in sys.modules if k.startswith("mlx")]
    saved = {k: sys.modules.pop(k) for k in mlx_modules}
    ff_mlx = [k for k in sys.modules if "fluxflow.mlx" in k]
    saved.update({k: sys.modules.pop(k) for k in ff_mlx})

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "mlx" or name.startswith("mlx."):
            raise ImportError("No module named 'mlx'")
        return real_import(name, *args, **kwargs)

    try:
        builtins.__import__ = mock_import
        with pytest.raises(ImportError, match="pip install fluxflow"):
            import importlib
            import fluxflow.mlx  # noqa: F401

            importlib.reload(fluxflow.mlx)
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def test_mlx_not_imported_by_core():
    """fluxflow.models must never trigger mlx import."""
    import sys

    mlx_before = set(k for k in sys.modules if k.startswith("mlx"))
    import fluxflow.models  # noqa: F401

    mlx_after = set(k for k in sys.modules if k.startswith("mlx"))
    assert mlx_after == mlx_before, f"fluxflow.models imported mlx: {mlx_after - mlx_before}"
