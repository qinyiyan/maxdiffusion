from maxdiffusion.schedulers.scheduling_unipc_multistep_flax import (
    FlaxUniPCMultistepScheduler,
)
import jax.numpy as jnp
import math
import warnings
from typing import Union, Optional, List, Tuple
from dataclasses import dataclass  # For dataclass usage in mock/real scheduler
import os
import jax
import numpy as np


def simulate_unipc_scheduler_jax(
    scheduler_params: dict,
    sample_shape: Tuple[int, ...],
    num_inference_steps: int,
    seed: int = 0,
):
    """
    Simulates the denoising process using the PyTorch UniPCMultistepScheduler.

    Args:
        scheduler_params (`dict`): Dictionary of parameters for scheduler initialization.
        sample_shape (`Tuple[int, ...]`): Shape of the sample tensors (e.g., (batch_size, channels, height, width)).
        num_inference_steps (`int`): Number of steps for the denoising process simulation.
        seed (`int`, defaults to 0): Random seed for reproducibility.
        device (`Union[str, torch.device]`, defaults to "cpu"): Device to run the simulation on.
    """
    print(f"\n--- Simulating UniPCMultistepScheduler ---")
    print(f"Parameters: {scheduler_params}")
    print(
        f"Sample shape: {sample_shape}, Inference steps: {num_inference_steps}, Seed: {seed}"
    )

    output_dir = "/tmp/jax_steps/"

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Intermediate denoising steps will be saved to: {os.path.abspath(output_dir)}"
    )

    # Set up JAX random key for reproducibility
    jax_key = jax.random.PRNGKey(seed)

    # 1. Instantiate the scheduler
    flax_scheduler = FlaxUniPCMultistepScheduler(**scheduler_params)

    # 2. Create and set initial state for the scheduler
    flax_state = flax_scheduler.create_state()
    flax_state = flax_scheduler.set_timesteps(
        flax_state, num_inference_steps, sample_shape
    )
    print("\nScheduler initialized.")
    print(
        f"  flax_state timesteps: {flax_state.timesteps}"
    )  # flax_state timesteps: [9 8 7 6 3] matches
    print(f"  flax_state sigmas shape: {flax_state.sigmas.shape}")

    # 3. Prepare the initial noisy latent sample
    # In a real scenario, this would typically be pure random noise (e.g., N(0,1))
    # For simulation, we'll generate it.
    # TODO: read from file
    # jax_key, subkey_sample_init = jax.random.split(jax_key)
    # sample = jax.random.normal(subkey_sample_init, sample_shape, dtype=jnp.float32)
    # print(f"\nInitial sample shape: {sample.shape}, dtype: {sample.dtype}")

    # 3. Prepare the initial noisy latent sample (LOAD FROM FILE)
    initial_sample_path = "/tmp/pytorch_steps/step_00_noisy_input.npy"
    if not os.path.exists(initial_sample_path):
        raise FileNotFoundError(
            f"Initial sample file not found at: {initial_sample_path}"
        )
    # Load the NumPy array and convert to JAX array
    initial_sample_np = np.load(initial_sample_path)
    sample = jnp.asarray(initial_sample_np, dtype=jnp.float32)  # Ensure dtype matches

    print(f"\nInitial sample loaded from: {initial_sample_path}")
    print(f"  Initial sample shape: {sample.shape}, dtype: {sample.dtype}")

    # Save the initial noisy sample
    initial_filename = os.path.join(output_dir, f"step_00_noisy_input.npy")
    jnp.save(
        initial_filename, sample.copy()
    )  # .copy() ensures it's on CPU and is a numpy array

    # 4. Simulate the denoising loop
    print("\nStarting denoising loop:")
    # It's good practice to JIT-compile the step function if it's called repeatedly
    # However, for debugging/logging inside the loop, sometimes not JITting the whole loop is better.
    # Here, we'll JIT the core `step` method implicitly by calling it in the loop.
    # For a performance-critical loop, `jax.lax.scan` or `jax.jit` of the whole loop might be used.
    for i, t in enumerate(flax_state.timesteps):
        print(f"  Step {i+1}/{num_inference_steps}, Timestep: {t.item()}")

        # Simulate model_output (e.g., noise prediction from a UNet)
        # In a real model, model_output = model(sample, t, conditioning).sample
        # For simulation, we use random noise as a stand-in for model output.
        # jax_key, subkey_model_output = jax.random.split(jax_key)
        # model_output = jax.random.normal(subkey_model_output, sample_shape, dtype=jnp.float32)

        # load the model output from file(generated from pytorch):
        model_output_file = f"/tmp/pytorch_steps/model_output_{i+1:02d}.npy"
        if not os.path.exists(model_output_file):
            raise FileNotFoundError(
                f"model_output file not found at: {model_output_file}"
            )
        # Load the NumPy array and convert to JAX array
        model_output_np = np.load(model_output_file)
        model_output = jnp.asarray(
            model_output_np, dtype=jnp.float32
        )  # Ensure dtype matches

        # Call the scheduler's step function
        # JIT compilation will happen here if not already for `step`
        scheduler_output = flax_scheduler.step(
            state=flax_state,
            model_output=model_output,
            timestep=t,  # Pass the current timestep from the scheduler's sequence
            sample=sample,
            return_dict=True,  # Return a SchedulerOutput dataclass
        )
        print(scheduler_output)
        sample = scheduler_output.prev_sample  # Update the sample for the next step
        flax_state = scheduler_output.state  # Update the state for the next step

        # Save the sample from the current step
        output_filename = os.path.join(output_dir, f"step_{i+1:02d}_t{t.item()}.npy")
        jnp.save(
            output_filename, sample.copy()
        )  # .copy() ensures it's a NumPy array (on CPU) for saving
        print(f"  Saved output of step {i+1} to: {output_filename}")

        # Optional: Print current state (for debugging/inspection)
        # print(f"    Current sample min: {sample.min().item():.4f}, max: {sample.max().item():.4f}")
        # print(f"    Scheduler step_index: {scheduler.step_index}")
        # print(f"    Scheduler lower_order_nums: {scheduler.lower_order_nums}")
        # print(f"    Scheduler this_order: {scheduler.this_order}")

    print("\nDenoising loop completed.")
    print(f"Final sample shape: {sample.shape}, dtype: {sample.dtype}")
    print(
        f"Final sample min: {sample.min().item():.4f}, max: {sample.max().item():.4f}"
    )


# --- Main execution ---
if __name__ == "__main__":
    # --- Configuration Parameters for the Scheduler ---
    # You can modify these parameters to test different scheduler behaviors
    scheduler_config_example = {
        "num_train_timesteps": 10,
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "beta_schedule": "linear",  # Options: "linear", "scaled_linear", "squaredcos_cap_v2"
        "solver_order": 2,  # Order of the UniPC solver: 1, 2, or 3
        "prediction_type": "flow_prediction",  # Options: "epsilon", "sample", "v_prediction", "flow_prediction"
        "thresholding": False,  # Enable dynamic thresholding (unsuitable for latent space models)
        "dynamic_thresholding_ratio": 0.995,
        "sample_max_value": 1.0,
        "predict_x0": True,  # Whether to use the updating algorithm on the predicted x0
        "solver_type": "bh2",  # Options: "bh1", "bh2"
        "lower_order_final": True,  # Use lower-order solvers in final steps for stability
        "disable_corrector": [],  # List of steps where corrector is disabled (e.g., [0] for first step)
        "use_karras_sigmas": False,  # Use Karras et al. (2022) noise schedule
        "use_exponential_sigmas": False,  # Use exponential noise schedule
        "use_beta_sigmas": False,  # Use beta noise schedule
        "use_flow_sigmas": True,  # Use flow-matching inspired sigmas
        "flow_shift": 3.0,  # Shift value for flow sigmas
        "timestep_spacing": "linspace",  # Options: "linspace", "leading", "trailing"
        "steps_offset": 0,  # An offset added to the inference steps
        "final_sigmas_type": "zero",  # Options: "zero", "sigma_min"
        "rescale_zero_terminal_snr": False,  # Whether to rescale the betas to have zero terminal SNR
        "solver_p": False,  # use nested scheduler
    }

    # --- Simulation Parameters ---
    latent_tensor_shape = (
        1,
        4,
        8,
        8,
    )  # Example latent tensor shape (Batch, Channels, Height, Width)
    inference_steps_count = 3  # Number of steps for the denoising process

    # --- Run the Simulation ---
    # Ensure _is_scipy_available_mock is set correctly if you change beta_schedule to "squaredcos_cap_v2"
    # _is_scipy_available_mock = True # Uncomment if testing "squaredcos_cap_v2"

    simulate_unipc_scheduler_jax(
        scheduler_params=scheduler_config_example,
        sample_shape=latent_tensor_shape,
        num_inference_steps=inference_steps_count,
        seed=42,  # Fixed seed for reproducibility
    )

    print("\nSimulation of UniPCMultistepScheduler usage complete.")
