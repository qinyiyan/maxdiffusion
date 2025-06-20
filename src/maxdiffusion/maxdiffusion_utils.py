"""
 Copyright 2024 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

import io
from PIL import Image
import importlib
import numpy as np
import tensorflow as tf
import jax
import jax.numpy as jnp
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from maxdiffusion import (
    max_utils,
)


from .models.modeling_flax_pytorch_utils import convert_pytorch_state_dict_to_flax


def load_sdxllightning_unet(config, pipeline, params):
  """Load lightning"""
  if not config.lightning_from_pt:
    raise ValueError("Only loading lightning models from Pytorch is currently supported.")
  unet_lightning_state_dict = load_file(hf_hub_download(config.lightning_repo, config.lightning_ckpt), device="cpu")
  flax_unet_dict = convert_pytorch_state_dict_to_flax(unet_lightning_state_dict, pipeline.unet)
  params["unet"] = flax_unet_dict
  return pipeline, params


def maybe_load_sdxl_lora(config, pipeline, params):

  def _noop_interceptor(next_fn, args, kwargs, context):
    return next_fn(*args, **kwargs)

  lora_config = config.lora_config
  interceptors = [_noop_interceptor]
  if len(lora_config["lora_model_name_or_path"]) > 0:
    interceptors = []
    for i in range(len(lora_config["lora_model_name_or_path"])):
      params, rank, network_alphas = pipeline.load_lora_weights(
          lora_config["lora_model_name_or_path"][i],
          weight_name=lora_config["weight_name"][i],
          params=params,
          adapter_name=lora_config["adapter_name"][i],
          unet_config=pipeline.unet.config,
      )
      interceptor = pipeline.make_lora_interceptor(params, rank, network_alphas, lora_config["adapter_name"][i])
      interceptors.append(interceptor)

  return params, interceptors


def vae_apply(images, sample_rng, vae, vae_params):
  """Apply vae encoder to images."""
  vae_outputs = vae.apply({"params": vae_params}, images, deterministic=True, method=vae.encode)
  latents = vae_outputs.latent_dist.sample(sample_rng)
  latents = jnp.transpose(latents, (0, 3, 1, 2))
  latents = latents * vae.config.scaling_factor

  return latents


def convert_dict_to_pil(image):
  """
  Converts a dictionary containing image bytes to a PIL Image object.

  Args:
    image_dict: A dictionary with keys 'bytes' (image data) and 'path' (optional).

  Returns:
    A PIL Image object.
  """
  if isinstance(image, dict):
    image_bytes = image["bytes"]
    image_stream = io.BytesIO(image_bytes)  # Create a BytesIO object
    pil_image = Image.open(image_stream)  # Open the image from the stream
    return pil_image
  return image


def transform_images(
    examples,
    image_column,
    image_resolution,
    rng=None,
    global_batch_size=None,
    pixel_ids_key="pixel_values",
    p_vae_apply=None,
):
  """Preprocess images to latents."""
  images = list(examples[image_column])
  images = [convert_dict_to_pil(image) for image in images]
  tensor_list = []
  for image in images:
    image = tf.image.resize(image, [image_resolution, image_resolution], method="bilinear", antialias=True)
    image = image / 255.0
    image = (image - 0.5) / 0.5
    image = tf.transpose(image, perm=[2, 0, 1])
    tensor_list.append(image)

  if p_vae_apply:
    tensor_list = np.stack(tensor_list)
    ds_length = tensor_list.shape[0]
    iters = ds_length // global_batch_size
    latents_list = []
    local_batch_size = global_batch_size // jax.device_count()
    for i in range(0, iters * global_batch_size, local_batch_size):
      sample_rng, rng = jax.random.split(rng)
      latents = p_vae_apply(tensor_list[i : i + local_batch_size], sample_rng)
      latents_list.append(latents)

    latents_list = np.stack(latents_list)
    b1, b2, c, l1, l2 = latents_list.shape
    latents_list = np.reshape(latents_list, (b1 * b2, c, l1, l2))

    # TODO (Juan Acevedo): do last iteration, its required for the Pyarrow dataset
    # to not break due to items being fewer than expected. Is there a better way?
    if tensor_list[i + local_batch_size :].shape[0] != 0:
      sample_rng, rng = jax.random.split(rng)
      latents = p_vae_apply(tensor_list[i + local_batch_size :], sample_rng)
      examples[pixel_ids_key] = np.append(latents_list, latents, axis=0)
    else:
      examples[pixel_ids_key] = latents_list
  else:
    examples[pixel_ids_key] = tf.stack(tensor_list)

  return examples


def get_add_time_ids(original_size, crops_coords_top_left, target_size, bs, dtype):
  add_time_ids = list(original_size + crops_coords_top_left + target_size)
  add_time_ids = jnp.array([add_time_ids] * bs, dtype=dtype)
  return add_time_ids


def override_scheduler_config(scheduler_config, config):
  """Overrides diffusion scheduler params from checkpoint."""

  maxdiffusion_scheduler_config = config.diffusion_scheduler_config

  scheduler_config["_class_name"] = maxdiffusion_scheduler_config.get("_class_name", scheduler_config["_class_name"])
  scheduler_config["prediction_type"] = maxdiffusion_scheduler_config.get(
      "prediction_type", scheduler_config["prediction_type"]
  )
  scheduler_config["timestep_spacing"] = maxdiffusion_scheduler_config.get(
      "timestep_spacing", scheduler_config["timestep_spacing"]
  )
  scheduler_config["rescale_zero_terminal_snr"] = maxdiffusion_scheduler_config.get("rescale_zero_terminal_snr", False)

  return scheduler_config


def create_scheduler(scheduler_config, config):
  """Creates scheduler from config."""
  scheduler_config = override_scheduler_config(scheduler_config, config)

  maxdiffusion_module = importlib.import_module(scheduler_config.__module__.split(".")[0])
  class_name = (
      scheduler_config["_class_name"]
      if scheduler_config["_class_name"].startswith("Flax")
      else "Flax" + scheduler_config["_class_name"]
  )
  cls = getattr(maxdiffusion_module, class_name)
  scheduler = cls.from_config(scheduler_config)

  scheduler_state = scheduler.create_state()
  return scheduler, scheduler_state


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
  """
  Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
  Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
  """
  std_text = jnp.std(noise_pred_text, axis=list(range(1, jnp.ndim(noise_pred_text))), keepdims=True)
  std_cfg = jnp.std(noise_cfg, axis=list(range(1, jnp.ndim(noise_cfg))), keepdims=True)
  # rescale the results from guidance (fixes overexposure)
  noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
  # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
  noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
  return noise_cfg


def get_dummy_unet_inputs(config, pipeline, batch_size):
  """Returns randomly initialized unet inputs."""
  vae_scale_factor = 2 ** (len(pipeline.vae.config["block_out_channels"]) - 1)
  input_shape = (
      batch_size,
      pipeline.unet.config["in_channels"],
      config.resolution // vae_scale_factor,
      config.resolution // vae_scale_factor,
  )

  latents = jax.random.normal(jax.random.PRNGKey(0), shape=input_shape, dtype=config.weights_dtype)
  timesteps = jnp.ones((latents.shape[0],))
  encoder_hidden_states_shape = (
      latents.shape[0],
      pipeline.text_encoder.config.max_position_embeddings,
      pipeline.unet.cross_attention_dim,
  )
  encoder_hidden_states = jnp.zeros(encoder_hidden_states_shape)
  added_cond_kwargs = None
  if pipeline.unet.addition_embed_type == "text_time":
    unet_config = pipeline.unet.config
    is_refiner = (
        5 * unet_config.addition_time_embed_dim + unet_config.cross_attention_dim
        == unet_config.projection_class_embeddings_input_dim
    )
    num_micro_conditions = 5 if is_refiner else 6

    text_embeds_dim = unet_config.projection_class_embeddings_input_dim - (
        num_micro_conditions * unet_config.addition_time_embed_dim
    )
    time_ids_channels = pipeline.unet.projection_class_embeddings_input_dim - text_embeds_dim
    time_ids_dims = time_ids_channels // pipeline.unet.addition_time_embed_dim
    added_cond_kwargs = {
        "text_embeds": jnp.zeros((batch_size, text_embeds_dim), dtype=jnp.float32),
        "time_ids": jnp.zeros((batch_size, time_ids_dims), dtype=jnp.float32),
    }
  return (latents, timesteps, encoder_hidden_states, added_cond_kwargs)


def calculate_unet_tflops(config, pipeline, batch_size, rngs, train):
  """
  Calculates unet tflops.
  batch_size should be per_device_batch_size * jax.local_device_count() or attention's shard_map won't
  cache the compilation when flash is enabled.
  """

  (latents, timesteps, encoder_hidden_states, added_cond_kwargs) = get_dummy_unet_inputs(config, pipeline, batch_size)
  return (
      max_utils.calculate_model_tflops(
          pipeline.unet,
          rngs,
          train,
          sample=latents,
          timesteps=timesteps,
          encoder_hidden_states=encoder_hidden_states,
          added_cond_kwargs=added_cond_kwargs,
      )
      / jax.local_device_count()
  )


def get_dummy_flux_inputs(config, pipeline, batch_size):
  """Returns randomly initialized flux inputs."""
  latents, latents_ids = pipeline.prepare_latents(
      batch_size=batch_size,
      num_channels_latents=pipeline.flux.in_channels // 4,
      height=config.resolution,
      width=config.resolution,
      vae_scale_factor=pipeline.vae_scale_factor,
      dtype=config.activations_dtype,
      rng=pipeline.rng,
  )
  guidance_vec = jnp.asarray([config.guidance_scale] * batch_size, dtype=config.activations_dtype)

  timesteps = jnp.ones((batch_size,), dtype=config.weights_dtype)
  t5_hidden_states_shape = (
      batch_size,
      config.max_sequence_length,
      4096,
  )
  t5_hidden_states = jnp.zeros(t5_hidden_states_shape, dtype=config.weights_dtype)
  t5_ids = jnp.zeros((batch_size, t5_hidden_states.shape[1], 3), dtype=config.weights_dtype)

  clip_hidden_states_shape = (
      batch_size,
      768,
  )
  clip_hidden_states = jnp.zeros(clip_hidden_states_shape, dtype=config.weights_dtype)

  return (latents, timesteps, latents_ids, guidance_vec, t5_hidden_states, t5_ids, clip_hidden_states)


def get_dummy_wan_inputs(config, pipeline, batch_size):
  latents = pipeline.prepare_latents(
      batch_size,
      vae_scale_factor_temporal=pipeline.vae_scale_factor_temporal,
      vae_scale_factor_spatial=pipeline.vae_scale_factor_spatial,
      height=config.height,
      width=config.width,
      num_frames=config.num_frames,
      num_channels_latents=pipeline.transformer.config.in_channels,
  )
  bsz = latents.shape[0]
  prompt_embeds = jax.random.normal(jax.random.key(config.seed), (batch_size, 512, 4096))
  timesteps = jnp.array([0] * bsz, dtype=jnp.int32)
  return (latents, prompt_embeds, timesteps)


def calculate_wan_tflops(config, pipeline, batch_size, rngs, train):
  """
  Calculates jflux tflops.
  batch_size should be per_device_batch_size * jax.local_device_count() or attention's shard_map won't
  cache the compilation when flash is enabled.
  """
  (latents, prompt_embeds, timesteps) = get_dummy_wan_inputs(config, pipeline, batch_size)
  return max_utils.calculate_model_tflops(
      pipeline.transformer,
  )


def calculate_flux_tflops(config, pipeline, batch_size, rngs, train):
  """
  Calculates jflux tflops.
  batch_size should be per_device_batch_size * jax.local_device_count() or attention's shard_map won't
  cache the compilation when flash is enabled.
  """

  (latents, timesteps, latents_ids, guidance_vec, t5_hidden_states, t5_ids, clip_hidden_states) = get_dummy_flux_inputs(
      config, pipeline, batch_size
  )
  return (
      max_utils.calculate_model_tflops(
          pipeline.flux,
          rngs,
          train,
          hidden_states=latents,
          img_ids=latents_ids,
          encoder_hidden_states=t5_hidden_states,
          txt_ids=t5_ids,
          pooled_projections=clip_hidden_states,
          timestep=timesteps,
          guidance=guidance_vec,
      )
      / jax.local_device_count()
  )


def tokenize_captions(examples, caption_column, tokenizer, input_ids_key="input_ids", p_encode=None):
  """Tokenize captions for sd1.x,sd2.x models."""
  captions = list(examples[caption_column])
  text_inputs = tokenizer(
      captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="np"
  )

  if p_encode:
    encoder_hidden_states = p_encode(np.stack(text_inputs.input_ids))
    examples[input_ids_key] = encoder_hidden_states
    # pyarrow dataset doesn't support bf16, so cast to float32
    examples[input_ids_key] = np.float32(examples[input_ids_key])
  else:
    examples[input_ids_key] = text_inputs.input_ids
  return examples


def tokenize_captions_xl(examples, caption_column, tokenizers, p_encode=None):
  inputs = []
  captions = list(examples[caption_column])
  for _tokenizer in tokenizers:
    text_inputs = _tokenizer(
        captions, padding="max_length", max_length=_tokenizer.model_max_length, truncation=True, return_tensors="np"
    )
    inputs.append(text_inputs.input_ids)
  inputs = np.stack(inputs, axis=1)

  if p_encode:
    prompt_embeds, text_embeds = p_encode(inputs)
    # pyarrow dataset doesn't support bf16, so cast to float32.
    examples["prompt_embeds"] = np.float32(prompt_embeds)
    examples["text_embeds"] = np.float32(text_embeds)
  examples["input_ids"] = inputs
  return examples


def get_shaped_batch(config, pipeline):
  """Return the shape of the batch - this is what eval_shape would return for the
  output of create_data_iterator_with_tokenizer, but eval_shape doesn't work, see b/306901078.
  This function works with sd1.x and 2.x.
  """
  vae_scale_factor = 2 ** (len(pipeline.vae.config.block_out_channels) - 1)
  total_train_batch_size = config.total_train_batch_size
  if config.cache_latents_text_encoder_outputs:
    batch_image_shape = (
        total_train_batch_size,
        4,
        config.resolution // vae_scale_factor,
        config.resolution // vae_scale_factor,
    )
    # bs, encoder_input, seq_length
    batch_ids_shape = (
        total_train_batch_size,
        pipeline.text_encoder.config.max_position_embeddings,
        pipeline.text_encoder.config.hidden_size,
    )
  else:
    batch_image_shape = (total_train_batch_size, 3, config.resolution, config.resolution)
    batch_ids_shape = (total_train_batch_size, pipeline.text_encoder.config.max_position_embeddings)
  shaped_batch = {}
  shaped_batch["pixel_values"] = jax.ShapeDtypeStruct(batch_image_shape, jnp.float32)
  shaped_batch["input_ids"] = jax.ShapeDtypeStruct(batch_ids_shape, jnp.float32)
  return shaped_batch


def encode(input_ids, text_encoder, text_encoder_params):
  return text_encoder(input_ids, params=text_encoder_params, train=False)[0]


def encode_xl(input_ids, text_encoders, text_encoder_params):
  te_1_inputs = input_ids[:, 0, :]
  te_2_inputs = input_ids[:, 1, :]

  prompt_embeds = text_encoders[0](te_1_inputs, params=text_encoder_params[0], output_hidden_states=True)
  prompt_embeds = prompt_embeds["hidden_states"][-2]

  prompt_embeds_2_out = text_encoders[1](te_2_inputs, params=text_encoder_params[1], output_hidden_states=True)
  prompt_embeds_2 = prompt_embeds_2_out["hidden_states"][-2]
  text_embeds = prompt_embeds_2_out["text_embeds"]
  prompt_embeds = jnp.concatenate([prompt_embeds, prompt_embeds_2], axis=-1)

  return prompt_embeds, text_embeds
