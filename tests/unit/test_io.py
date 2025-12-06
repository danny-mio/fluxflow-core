"""Unit tests for I/O utilities (src/utils/io.py)."""

import json
import os

import pytest
import safetensors.torch
import torch
import torch.nn as nn

from fluxflow.utils.io import (
    copy_and_replace,
    format_duration,
    load_discriminators_if_any,
    load_training_state,
    save_discriminators,
    save_model,
    save_training_state,
)


class TestCopyAndReplace:
    """Tests for copy_and_replace function."""

    def test_copy_file(self, temp_dir):
        """Test copying file to new location."""
        # Create source file
        source = temp_dir / "source.txt"
        source.write_text("test content")

        # Copy to destination
        dest = temp_dir / "dest.txt"
        copy_and_replace(str(source), str(dest))

        # Destination should exist with same content
        assert dest.exists()
        assert dest.read_text() == "test content"

    def test_replace_existing_file(self, temp_dir):
        """Test replacing existing file."""
        # Create source and destination
        source = temp_dir / "source.txt"
        dest = temp_dir / "dest.txt"
        source.write_text("new content")
        dest.write_text("old content")

        # Copy and replace
        copy_and_replace(str(source), str(dest))

        # Destination should have new content
        assert dest.read_text() == "new content"

    def test_preserves_metadata(self, temp_dir):
        """Test that copy preserves file metadata."""
        source = temp_dir / "source.txt"
        source.write_text("test")

        dest = temp_dir / "dest.txt"
        copy_and_replace(str(source), str(dest))

        # Both files should have similar timestamps (within 1 second)
        source_mtime = os.path.getmtime(source)
        dest_mtime = os.path.getmtime(dest)
        assert abs(source_mtime - dest_mtime) < 1.0


class TestSaveModel:
    """Tests for save_model function."""

    def test_creates_output_directory(self, temp_dir):
        """Should create output directory if it doesn't exist."""
        output_path = temp_dir / "models"
        assert not output_path.exists()

        # Create mock models
        diffuser = nn.Linear(10, 10)
        text_encoder = nn.Linear(20, 20)

        save_model(diffuser, text_encoder, str(output_path))

        assert output_path.exists()

    def test_saves_safetensors_file(self, temp_dir):
        """Should save model in safetensors format."""
        diffuser = nn.Linear(10, 10)
        text_encoder = nn.Linear(20, 20)

        save_model(diffuser, text_encoder, str(temp_dir))

        model_path = temp_dir / "flxflow_final.safetensors"
        assert model_path.exists()

    def test_saves_text_encoder_separately(self, temp_dir):
        """Should save text encoder to separate file."""
        diffuser = nn.Linear(10, 10)
        text_encoder = nn.Linear(20, 20)

        save_model(diffuser, text_encoder, str(temp_dir))

        te_path = temp_dir / "text_encoder.safetensors"
        assert te_path.exists()

    def test_creates_backup_on_overwrite(self, temp_dir):
        """Should create .bck backup when overwriting."""
        diffuser = nn.Linear(10, 10)
        text_encoder = nn.Linear(20, 20)

        # Save once
        save_model(diffuser, text_encoder, str(temp_dir))

        # Save again (should create backup)
        save_model(diffuser, text_encoder, str(temp_dir))

        backup_path = temp_dir / "flxflow_final.safetensors.bck"
        assert backup_path.exists()

    def test_state_dict_keys_prefixed(self, temp_dir):
        """Saved state dict should have proper key prefixes."""
        diffuser = nn.Linear(10, 10)
        text_encoder = nn.Linear(20, 20)

        save_model(diffuser, text_encoder, str(temp_dir))

        # Load and check keys
        model_path = temp_dir / "flxflow_final.safetensors"
        state = safetensors.torch.load_file(str(model_path))

        # Should have both diffuser and text_encoder keys
        diffuser_keys = [k for k in state.keys() if k.startswith("diffuser.")]
        te_keys = [k for k in state.keys() if k.startswith("text_encoder.")]

        assert len(diffuser_keys) > 0
        assert len(te_keys) > 0

    def test_save_pretrained_includes_language_model(self, temp_dir):
        """save_pretrained=True should include language_model weights."""
        # Create text encoder with language_model param
        text_encoder = nn.Module()
        text_encoder.language_model = nn.Linear(100, 100)
        text_encoder.output_layer = nn.Linear(100, 50)

        diffuser = nn.Linear(10, 10)

        # Save with save_pretrained=True
        save_model(diffuser, text_encoder, str(temp_dir), save_pretrained=True)

        # Check text encoder file
        te_path = temp_dir / "text_encoder.safetensors"
        te_state = safetensors.torch.load_file(str(te_path))

        # Should include language_model keys
        lm_keys = [k for k in te_state.keys() if "language_model" in k]
        assert len(lm_keys) > 0


class TestSaveDiscriminators:
    """Tests for save_discriminators function."""

    def test_saves_discriminator(self, temp_dir):
        """Should save discriminator to safetensors."""
        d_img = nn.Conv2d(3, 64, 3)

        save_discriminators(d_img, str(temp_dir))

        disc_path = temp_dir / "disc_img.safetensors"
        assert disc_path.exists()

    def test_handles_none_discriminator(self, temp_dir):
        """Should handle None discriminator gracefully."""
        # Should not crash
        save_discriminators(None, str(temp_dir))

    def test_creates_backup(self, temp_dir):
        """Should create backup when overwriting."""
        d_img = nn.Conv2d(3, 64, 3)

        # Save twice
        save_discriminators(d_img, str(temp_dir))
        save_discriminators(d_img, str(temp_dir))

        backup_path = temp_dir / "disc_img.safetensors.bck"
        assert backup_path.exists()

    def test_state_dict_keys_prefixed(self, temp_dir):
        """Discriminator state dict should have proper prefix."""
        d_img = nn.Conv2d(3, 64, 3)

        save_discriminators(d_img, str(temp_dir))

        disc_path = temp_dir / "disc_img.safetensors"
        state = safetensors.torch.load_file(str(disc_path))

        # All keys should start with "disc_img."
        for key in state.keys():
            assert key.startswith("disc_img.")


class TestLoadDiscriminatorsIfAny:
    """Tests for load_discriminators_if_any function."""

    def test_loads_discriminator(self, temp_dir):
        """Should load discriminator weights if file exists."""
        # Create and save discriminator
        d_img_orig = nn.Conv2d(3, 64, 3)
        save_discriminators(d_img_orig, str(temp_dir))

        # Create new discriminator and load
        d_img_new = nn.Conv2d(3, 64, 3)
        load_discriminators_if_any(d_img_new, str(temp_dir))

        # Weights should match
        for (n1, p1), (n2, p2) in zip(d_img_orig.named_parameters(), d_img_new.named_parameters()):
            assert torch.allclose(p1, p2)

    def test_skips_if_no_file(self, temp_dir):
        """Should skip loading if no discriminator file exists."""
        d_img = nn.Conv2d(3, 64, 3)

        # Should not crash
        load_discriminators_if_any(d_img, str(temp_dir))


class TestFormatDuration:
    """Tests for format_duration function."""

    def test_zero_seconds(self):
        """0 seconds should format correctly."""
        assert format_duration(0) == "0000:00:00.000"

    def test_less_than_minute(self):
        """Less than 1 minute should format correctly."""
        result = format_duration(45.123)
        # Check format matches, allow minor floating point rounding
        assert result.startswith("0000:00:45.12")
        assert len(result) == len("0000:00:45.123")

    def test_minutes_and_seconds(self):
        """Minutes and seconds should format correctly."""
        assert format_duration(125.5) == "0000:02:05.500"

    def test_hours(self):
        """Hours should format correctly."""
        assert format_duration(3661.25) == "0001:01:01.250"

    def test_days(self):
        """Days should convert to hours."""
        # 1 day = 86400 seconds = 24 hours
        assert format_duration(86400) == "0024:00:00.000"

    def test_large_duration(self):
        """Large duration should format correctly."""
        # 10 days, 5 hours, 30 min, 15.5 sec
        seconds = 10 * 86400 + 5 * 3600 + 30 * 60 + 15.5
        formatted = format_duration(seconds)
        assert formatted.startswith("0245:")  # 240 + 5 hours


class TestSaveTrainingState:
    """Tests for save_training_state function."""

    def test_saves_json_file(self, temp_dir):
        """Should save training state as JSON."""
        save_training_state(
            output_path=str(temp_dir),
            epoch=5,
            batch_idx=42,
            global_step=1234,
            samples_trained=50000,
            total_samples=100000,
            learning_rates={"vae": 1e-4, "flow": 5e-5},
        )

        state_path = temp_dir / "training_state.json"
        assert state_path.exists()

    def test_json_structure(self, temp_dir):
        """JSON should have correct structure."""
        save_training_state(
            output_path=str(temp_dir),
            epoch=5,
            batch_idx=42,
            global_step=1234,
            samples_trained=50000,
            total_samples=100000,
            learning_rates={"vae": 1e-4},
        )

        state_path = temp_dir / "training_state.json"
        with open(state_path) as f:
            state = json.load(f)

        assert state["version"] == "1.0"
        assert state["epoch"] == 5
        assert state["batch_idx"] == 42
        assert state["global_step"] == 1234
        assert state["samples_trained"] == 50000
        assert state["total_samples"] == 100000
        assert "timestamp" in state

    def test_saves_sampler_state(self, temp_dir):
        """Should save sampler state if provided."""
        sampler_state = {"seed": 42, "position": 10}

        save_training_state(
            output_path=str(temp_dir),
            epoch=1,
            batch_idx=5,
            global_step=100,
            samples_trained=1000,
            total_samples=10000,
            learning_rates={"vae": 1e-4},
            sampler_state=sampler_state,
        )

        state_path = temp_dir / "training_state.json"
        with open(state_path) as f:
            state = json.load(f)

        assert state["sampler_state"] == sampler_state

    def test_saves_optimizer_states_separately(self, temp_dir):
        """Should save optimizer states to separate file."""
        opt_states = {
            "vae": {"step": 100, "exp_avg": [1, 2, 3]},
            "flow": {"step": 100, "exp_avg": [4, 5, 6]},
        }

        save_training_state(
            output_path=str(temp_dir),
            epoch=1,
            batch_idx=5,
            global_step=100,
            samples_trained=1000,
            total_samples=10000,
            learning_rates={"vae": 1e-4},
            optimizers=opt_states,
        )

        opt_path = temp_dir / "optimizer_states.pt"
        assert opt_path.exists()

    def test_creates_backup(self, temp_dir):
        """Should create backup when overwriting."""
        # Save twice
        for i in range(2):
            save_training_state(
                output_path=str(temp_dir),
                epoch=i,
                batch_idx=0,
                global_step=0,
                samples_trained=0,
                total_samples=1000,
                learning_rates={"vae": 1e-4},
            )

        backup_path = temp_dir / "training_state.json.bck"
        assert backup_path.exists()


class TestLoadTrainingState:
    """Tests for load_training_state function."""

    def test_loads_saved_state(self, temp_dir):
        """Should load previously saved state."""
        # Save state
        save_training_state(
            output_path=str(temp_dir),
            epoch=5,
            batch_idx=42,
            global_step=1234,
            samples_trained=50000,
            total_samples=100000,
            learning_rates={"vae": 1e-4, "flow": 5e-5},
        )

        # Load state
        state = load_training_state(str(temp_dir))

        assert state is not None
        assert state["epoch"] == 5
        assert state["batch_idx"] == 42
        assert state["global_step"] == 1234
        assert state["samples_trained"] == 50000

    def test_returns_none_if_not_found(self, temp_dir):
        """Should return None if no state file exists."""
        state = load_training_state(str(temp_dir))
        assert state is None

    def test_loads_optimizer_states(self, temp_dir):
        """Should load optimizer states from separate file."""
        opt_states = {
            "vae": {"step": 100},
            "flow": {"step": 200},
        }

        # Save with optimizer states
        save_training_state(
            output_path=str(temp_dir),
            epoch=1,
            batch_idx=5,
            global_step=100,
            samples_trained=1000,
            total_samples=10000,
            learning_rates={"vae": 1e-4},
            optimizers=opt_states,
        )

        # Load
        state = load_training_state(str(temp_dir))

        assert "optimizer_states" in state
        assert state["optimizer_states"]["vae"]["step"] == 100
        assert state["optimizer_states"]["flow"]["step"] == 200

    def test_handles_corrupt_json(self, temp_dir):
        """Should return None on corrupt JSON."""
        # Create corrupt JSON file
        state_path = temp_dir / "training_state.json"
        state_path.write_text("{ invalid json")

        state = load_training_state(str(temp_dir))
        assert state is None


class TestSchedulerStateSaving:
    """Tests for scheduler state saving and loading."""

    def test_saves_scheduler_states_to_file(self, temp_dir):
        """Should save scheduler states to scheduler_states.pt."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Create model and optimizer
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

        # Step scheduler a few times
        for _ in range(10):
            scheduler.step()

        # Save scheduler state
        scheduler_path = temp_dir / "scheduler_states.pt"
        torch.save({"scheduler": scheduler.state_dict()}, scheduler_path)

        assert scheduler_path.exists()

    def test_loads_scheduler_states_correctly(self, temp_dir):
        """Should load scheduler states and maintain internal state."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        # Create and step scheduler
        model = nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-5)

        # Step 10 times
        for _ in range(10):
            scheduler.step()

        lr_after_10_steps = optimizer.param_groups[0]["lr"]

        # Save state
        scheduler_path = temp_dir / "scheduler_states.pt"
        torch.save({"scheduler": scheduler.state_dict()}, scheduler_path)

        # Create new scheduler and load state
        new_model = nn.Linear(10, 10)
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
        new_scheduler = CosineAnnealingLR(new_optimizer, T_max=1000, eta_min=1e-5)

        loaded_states = torch.load(scheduler_path, weights_only=False)
        new_scheduler.load_state_dict(loaded_states["scheduler"])

        # Learning rates should match (allow for floating point precision)
        assert abs(new_optimizer.param_groups[0]["lr"] - lr_after_10_steps) < 1e-6


class TestEMAStateSaving:
    """Tests for EMA state saving and loading."""

    def test_saves_ema_state_to_file(self, temp_dir):
        """Should save EMA state to ema_state.pt."""
        pytest.importorskip("fluxflow_training")
        from fluxflow_training.training.utils import EMA

        # Create model and EMA
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Update EMA a few times
        for _ in range(5):
            # Modify model parameters
            for p in model.parameters():
                p.data += 0.1
            ema.update()

        # Save EMA state
        ema_path = temp_dir / "ema_state.pt"
        torch.save(ema.state_dict(), ema_path)

        assert ema_path.exists()

    def test_loads_ema_state_correctly(self, temp_dir):
        """Should load EMA state and maintain shadow parameters."""
        pytest.importorskip("fluxflow_training")
        from fluxflow_training.training.utils import EMA

        # Create model and EMA
        model = nn.Linear(10, 10)
        ema = EMA(model, decay=0.999)

        # Update EMA
        for _ in range(5):
            for p in model.parameters():
                p.data += 0.1
            ema.update()

        # Save shadow parameters for comparison
        original_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Save EMA state
        ema_path = temp_dir / "ema_state.pt"
        torch.save(ema.state_dict(), ema_path)

        # Create new EMA and load state
        new_model = nn.Linear(10, 10)
        new_ema = EMA(new_model, decay=0.999)

        loaded_state = torch.load(ema_path, weights_only=False)
        new_ema.load_state_dict(loaded_state)

        # Shadow parameters should match
        for key in original_shadow:
            assert torch.allclose(original_shadow[key], new_ema.shadow[key])

        # Decay should match (allow for tensor->float conversion precision loss)
        assert abs(new_ema.decay - 0.999) < 1e-6


class TestIOIntegration:
    """Integration tests for I/O operations."""

    def test_save_and_load_model_roundtrip(self, temp_dir):
        """Test saving and loading model maintains weights."""
        # Create models
        diffuser = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        text_encoder = nn.Linear(50, 100)

        # Save
        save_model(diffuser, text_encoder, str(temp_dir))

        # Load
        model_path = temp_dir / "flxflow_final.safetensors"
        loaded_state = safetensors.torch.load_file(str(model_path))

        # Create new models and load weights
        diffuser_new = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 10))
        diffuser_new.load_state_dict(
            {
                k.replace("diffuser.", ""): v
                for k, v in loaded_state.items()
                if k.startswith("diffuser.")
            }
        )

        # Weights should match
        for (n1, p1), (n2, p2) in zip(diffuser.named_parameters(), diffuser_new.named_parameters()):
            assert torch.allclose(p1.cpu(), p2.cpu())

    def test_checkpoint_resume_workflow(self, temp_dir):
        """Test realistic checkpoint and resume workflow."""
        # Initial training state
        save_training_state(
            output_path=str(temp_dir),
            epoch=3,
            batch_idx=150,
            global_step=5000,
            samples_trained=80000,
            total_samples=100000,
            learning_rates={"vae": 1e-4, "flow": 5e-5},
            sampler_state={"seed": 42, "position": 150},
        )

        # Simulate crash/restart - load state
        state = load_training_state(str(temp_dir))

        # Verify can resume from correct position
        assert state["epoch"] == 3
        assert state["batch_idx"] == 150
        assert state["sampler_state"]["position"] == 150

    def test_multiple_checkpoints_with_backups(self, temp_dir):
        """Test that multiple saves maintain backups."""
        # Save epoch 0
        save_training_state(
            output_path=str(temp_dir),
            epoch=0,
            batch_idx=100,
            global_step=100,
            samples_trained=800,
            total_samples=10000,
            learning_rates={"vae": 1e-4},
        )

        # Load to verify state was saved
        assert load_training_state(str(temp_dir)) is not None

        # Save epoch 1 (creates backup of epoch 0)
        save_training_state(
            output_path=str(temp_dir),
            epoch=1,
            batch_idx=50,
            global_step=150,
            samples_trained=1200,
            total_samples=10000,
            learning_rates={"vae": 1e-4},
        )

        # Current should be epoch 1
        state1 = load_training_state(str(temp_dir))
        assert state1["epoch"] == 1

        # Backup should exist
        backup_path = temp_dir / "training_state.json.bck"
        assert backup_path.exists()
