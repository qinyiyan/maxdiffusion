from maxdiffusion.models.wan.autoencoder_kl_wan import (
  WanCausalConv3d,
  WanUpsample,
  AutoencoderKLWan,
  AutoencoderKLWanCache,
  WanEncoder3d,
  WanMidBlock,
  WanResidualBlock,
  WanRMS_norm,
  WanResample,
  ZeroPaddedConv2D,
  WanAttentionBlock
)
from maxdiffusion.models.wan.wan_utils import load_wan_vae
import jax
import jax.numpy as jnp
from flax import nnx
from maxdiffusion.video_processor import VideoProcessor
from maxdiffusion.utils import load_video, export_to_video
import numpy as np
import torch
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

pretrained_model_name_or_path = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
key = jax.random.key(0)
rngs = nnx.Rngs(key)
wan_vae = AutoencoderKLWan.from_config(
  pretrained_model_name_or_path,
  subfolder="vae",
  rngs=rngs
)

video = load_video("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/hiker.mp4")


vae_scale_factor_temporal = 2 ** sum(wan_vae.temperal_downsample)
vae_scale_factor_spatial = 2 ** len(wan_vae.temperal_downsample)
video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
width, height = video[0].size
video = video_processor.preprocess_video(video, height=height, width=width)#.to(dtype=jnp.float32)
video = jnp.array(np.array(video), dtype=jnp.float32)

graphdef, state = nnx.split(wan_vae, nnx.Param)
params = state.to_pure_dict()
# This replaces random params with the model.
params = load_wan_vae(pretrained_model_name_or_path, params, "cpu")
params = jax.tree_util.tree_map(lambda x: x.astype(jnp.bfloat16), params)
# breakpoint()
wan_vae = nnx.merge(graphdef, params)

original_video_shape = video.shape
vae_cache = AutoencoderKLWanCache(wan_vae)

latent = wan_vae.encode(video, vae_cache)
latent = latent.latent_dist.sample(key)

# load latnets file
# latent = np.load("jax_video_latent.npy")
# latent = jnp.array(latent, dtype=jnp.bfloat16)
# breakpoint()
#latent = jnp.transpose(latent, (0, 2, 3, 4, 1))

assert latent.shape == (1, 13, 60, 90, 16)
# breakpoint()
print("after encoding, latents.min: ", np.min(latent))
print("after encoding, latents.max: ", np.max(latent))

video = wan_vae.decode(latent, vae_cache, return_dict=False)[0]
assert video.shape == (1, 49, 480, 720, 3)
print("after decode, video.min:", np.min(video))
print("after decode, video.max:", np.max(video))

# channels first
video = np.array(video, dtype=np.float32)
np.save("decoded_video.npz", video)
#breakpoint()
video = jnp.transpose(video, (0, 4, 1, 2, 3))
assert video.shape == original_video_shape

# We convert to torch tensors just to use HF's torch video utility classes.
video = torch.from_numpy(np.array(video)).to(dtype=torch.bfloat16)
print("after decode, video.min:", torch.min(video))
print("after decode, video.max:", torch.max(video))
#breakpoint()
video = video_processor.postprocess_video(video, output_type="np")
export_to_video(video[0], "jax_output.mp4", fps=8)