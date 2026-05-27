"""Unit tests for visualization utilities."""

from unittest.mock import MagicMock, patch

import torch


class TestSafeVaeSampleContextOnly:
    """Tests for safe_vae_sample: only ctx file generated, no -nc.webp."""

    def test_safe_vae_sample_only_ctx_file_generated(self, tmp_path):
        """save_image called for ctx output; no -nc filename."""
        from fluxflow.utils.visualization import safe_vae_sample

        # Build mock diffuser
        diffuser = MagicMock()
        fake_latent = torch.zeros(1, 65, 32)
        diffuser.compressor.return_value = fake_latent
        diffuser.compressor.get_context_dims.return_value = 0
        diffuser.compressor.d_model = 32
        diffuser.expander.return_value = torch.zeros(1, 3, 64, 64)
        diffuser.return_value = torch.zeros(1, 3, 64, 64)

        # Create a minimal test image
        from PIL import Image

        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (64, 64), color=(128, 128, 128)).save(img_path)

        output_dir = str(tmp_path / "output")
        import os

        os.makedirs(output_dir, exist_ok=True)

        with patch("fluxflow.utils.visualization.save_image") as mock_save:
            safe_vae_sample(
                diffuser=diffuser,
                image_address=img_path,
                channels=3,
                output_path=output_dir,
                epoch=0,
                device=torch.device("cpu"),
            )

        # save_image must have been called
        assert mock_save.called

        # No call should reference a -nc.webp filename
        for c in mock_save.call_args_list:
            filepath = c.args[1] if c.args else list(c.kwargs.values())[0]
            assert "-nc" not in str(
                filepath
            ), f"No-context decode file should not be generated; got: {filepath}"

    def test_save_image_called_with_ctx_suffix(self, tmp_path):
        """Exactly one ctx output file is generated."""
        from fluxflow.utils.visualization import safe_vae_sample

        diffuser = MagicMock()
        diffuser.compressor.return_value = torch.zeros(1, 65, 32)
        diffuser.compressor.get_context_dims.return_value = 0
        diffuser.compressor.d_model = 32
        diffuser.expander.return_value = torch.zeros(1, 3, 64, 64)
        diffuser.return_value = torch.zeros(1, 3, 64, 64)

        from PIL import Image

        img_path = str(tmp_path / "test.png")
        Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

        import os

        output_dir = str(tmp_path / "out")
        os.makedirs(output_dir, exist_ok=True)

        with patch("fluxflow.utils.visualization.save_image") as mock_save:
            safe_vae_sample(
                diffuser=diffuser,
                image_address=img_path,
                channels=3,
                output_path=output_dir,
                epoch=1,
                device=torch.device("cpu"),
            )

        saved_paths = [str(c.args[1]) for c in mock_save.call_args_list if c.args]
        ctx_paths = [p for p in saved_paths if "-ctx" in p]
        assert len(ctx_paths) == 1, f"Expected exactly one -ctx file; got: {saved_paths}"
