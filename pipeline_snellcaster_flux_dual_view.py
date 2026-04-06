# Copyright 2025 Black Forest Labs and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
    T5EncoderModel,
    T5TokenizerFast,
)

from diffusers.pipelines.flux.pipeline_flux import FluxPipeline, calculate_shift, retrieve_timesteps
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import FluxIPAdapterMixin, FluxLoraLoaderMixin, FromSingleFileMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, FluxTransformer2DModel
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_xla_available,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

# Import the original pipeline for inheritance
from pipeline_snellcaster_flux import SnellcasterPipeline_Flux, calculate_shift, retrieve_timesteps

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from dual_pipeline_snellcaster_flux import DualSnellcasterPipeline_Flux

        >>> pipe = DualSnellcasterPipeline_Flux.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")
        >>> main_prompt = "A cat holding a sign that says hello world"
        >>> pano_prompt = "A panoramic view of a beautiful landscape"
        >>> # Generate both main and panorama images simultaneously
        >>> main_image, pano_image = pipe(
        ...     main_prompt=main_prompt,
        ...     pano_prompt=pano_prompt,
        ...     num_inference_steps=4,
        ...     main_guidance_scale=7.0,
        ...     pano_guidance_scale=5.0
        ... ).images
        >>> main_image[0].save("main_flux.png")
        >>> pano_image[0].save("pano_flux.png")
        ```
"""


class DualSnellcasterPipeline_Flux(FluxPipeline):
    """
    A dual-image Flux pipeline that can generate two images simultaneously.
    
    This pipeline extends FluxPipeline to generate both a main image and a panorama
    image at the same time, with the ability to process both latents together during denoising.
    
    Features:
    - Generates main image and panorama simultaneously
    - Supports dual-image callbacks for latent interaction
    - Maintains all original Flux functionality
    - Allows correspondence-based interaction between the two images
    """
    
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        text_encoder_2,
        tokenizer_2,
        transformer,
        image_encoder=None,
        feature_extractor=None,
        main_scheduler=None,
        pano_scheduler=None,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
        )
        # Use separate schedulers for main and panorama images
        self.main_scheduler = main_scheduler if main_scheduler is not None else scheduler
        self.pano_scheduler = pano_scheduler if pano_scheduler is not None else scheduler

    def _compute_noise_predictions(
        self,
        main_latents: torch.Tensor,
        pano_latents: torch.Tensor,
        timestep: torch.Tensor,
        main_guidance: Optional[torch.Tensor],
        pano_guidance: Optional[torch.Tensor],
        main_pooled_prompt_embeds: torch.Tensor,
        pano_pooled_prompt_embeds: torch.Tensor,
        main_prompt_embeds: torch.Tensor,
        pano_prompt_embeds: torch.Tensor,
        main_text_ids: torch.Tensor,
        pano_text_ids: torch.Tensor,
        main_latent_image_ids: torch.Tensor,
        pano_latent_image_ids: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
        negative_pooled_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_text_ids: Optional[torch.Tensor],
        negative_image_embeds: Optional[List[torch.Tensor]],
        image_embeds: Optional[List[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute noise predictions for both main and panorama images with optional True-CFG.
        """
        # Process main image
        with self.transformer.cache_context("cond"):
            main_noise_pred = self.transformer(
                hidden_states=main_latents,
                timestep=timestep / 1000,
                guidance=main_guidance,
                pooled_projections=main_pooled_prompt_embeds,
                encoder_hidden_states=main_prompt_embeds,
                txt_ids=main_text_ids,
                img_ids=main_latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

        # Process panorama image
        with self.transformer.cache_context("cond"):
            pano_noise_pred = self.transformer(
                hidden_states=pano_latents,
                timestep=timestep / 1000,
                guidance=pano_guidance,
                pooled_projections=pano_pooled_prompt_embeds,
                encoder_hidden_states=pano_prompt_embeds,
                txt_ids=pano_text_ids,
                img_ids=pano_latent_image_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]

        if do_true_cfg:
            if negative_image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds

            # Process negative prompts for both images
            with self.transformer.cache_context("uncond"):
                main_neg_noise_pred = self.transformer(
                    hidden_states=main_latents,
                    timestep=timestep / 1000,
                    guidance=main_guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=main_latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

                pano_neg_noise_pred = self.transformer(
                    hidden_states=pano_latents,
                    timestep=timestep / 1000,
                    guidance=pano_guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=pano_latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]

            main_noise_pred = main_neg_noise_pred + true_cfg_scale * (main_noise_pred - main_neg_noise_pred)
            pano_noise_pred = pano_neg_noise_pred + true_cfg_scale * (pano_noise_pred - pano_neg_noise_pred)

            if image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

        return main_noise_pred, pano_noise_pred

    def _apply_time_travel(
        self,
        i: int,
        t: torch.Tensor,
        main_latents: torch.Tensor,
        pano_latents: torch.Tensor,
        updated_main_tweedie_latents: torch.Tensor,
        updated_pano_tweedie_latents: torch.Tensor,
        time_travel_repeats: int,
        main_guidance: Optional[torch.Tensor],
        pano_guidance: Optional[torch.Tensor],
        main_pooled_prompt_embeds: torch.Tensor,
        pano_pooled_prompt_embeds: torch.Tensor,
        main_prompt_embeds: torch.Tensor,
        pano_prompt_embeds: torch.Tensor,
        main_text_ids: torch.Tensor,
        pano_text_ids: torch.Tensor,
        main_latent_image_ids: torch.Tensor,
        pano_latent_image_ids: torch.Tensor,
        do_true_cfg: bool,
        true_cfg_scale: float,
        negative_pooled_prompt_embeds: Optional[torch.Tensor],
        negative_prompt_embeds: Optional[torch.Tensor],
        negative_text_ids: Optional[torch.Tensor],
        negative_image_embeds: Optional[List[torch.Tensor]],
        image_embeds: Optional[List[torch.Tensor]],
        generator_main: Optional[torch.Generator],
        generator_pano: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply time travel diffusion: travel forward to a noiser state N-1 times, denoise back using x0|t,
        then recompute noise predictions at updated x_t.
        
        NOTE: Time travel is ONLY applied to main view, pano view is kept unchanged.

        Returns:
            main_latents, pano_latents, main_noise_pred, pano_noise_pred, main_current_sigma, pano_current_sigma
        """
        # Initialize x_current = x_t
        main_latent_current = main_latents
        # pano_latent_current = pano_latents  # COMMENTED OUT: No time travel for pano

        # Save current step_index to restore later (both schedulers are in sync)
        saved_step_index = self.main_scheduler.step_index

        # Use saved index for time travel (represents the current timestep t)
        tt_sigma_idx = saved_step_index

        for _ in range(time_travel_repeats - 1):
            # Get sigma_noiser (previous in sigmas array, which has more noise)
            if tt_sigma_idx <= 0:
                break
            main_sigma_noiser = self.main_scheduler.sigmas[tt_sigma_idx - 1]
            # pano_sigma_noiser = self.pano_scheduler.sigmas[tt_sigma_idx - 1]  # COMMENTED OUT: No time travel for pano

            # Sample noise
            if generator_main is not None:
                main_noise = torch.randn(
                    main_latent_current.shape,
                    dtype=main_latent_current.dtype,
                    device=main_latent_current.device,
                    generator=generator_main,
                )
            else:
                main_noise = torch.randn_like(main_latent_current)

            # COMMENTED OUT: No time travel for pano
            # if generator_pano is not None:
            #     pano_noise = torch.randn(
            #         pano_latent_current.shape,
            #         dtype=pano_latent_current.dtype,
            #         device=pano_latent_current.device,
            #         generator=generator_pano,
            #     )
            # else:
            #     pano_noise = torch.randn_like(pano_latent_current)

            # Forward diffusion using Euler-Maruyama style (mirror of backward step)
            main_current_sigma_tt = self.main_scheduler.sigmas[tt_sigma_idx]
            # pano_current_sigma_tt = self.pano_scheduler.sigmas[tt_sigma_idx]  # COMMENTED OUT: No time travel for pano

            main_dt_forward = main_sigma_noiser - main_current_sigma_tt
            # pano_dt_forward = pano_sigma_noiser - pano_current_sigma_tt  # COMMENTED OUT: No time travel for pano

            main_forward_velocity = (main_noise - main_latent_current) / (1.0 - main_current_sigma_tt)
            # pano_forward_velocity = (pano_noise - pano_latent_current) / (1.0 - pano_current_sigma_tt)  # COMMENTED OUT: No time travel for pano

            main_deterministic_step_forward = main_latent_current + main_dt_forward * main_forward_velocity
            # pano_deterministic_step_forward = pano_latent_current + pano_dt_forward * pano_forward_velocity  # COMMENTED OUT: No time travel for pano

            main_noise_scale_forward = torch.abs(main_current_sigma_tt - main_sigma_noiser)
            # pano_noise_scale_forward = torch.abs(pano_current_sigma_tt - pano_sigma_noiser)  # COMMENTED OUT: No time travel for pano

            # Forward diffusion: latent_noiser = deterministic_step + noise_scale * noise
            main_latent_noiser = main_deterministic_step_forward + main_noise_scale_forward * main_noise
            # pano_latent_noiser = pano_deterministic_step_forward + pano_noise_scale_forward * pano_noise  # COMMENTED OUT: No time travel for pano

            # Denoise x_noiser -> x_t using x_0|t (not x_0|noiser)
            self.main_scheduler._step_index = tt_sigma_idx - 1
            self.pano_scheduler._step_index = tt_sigma_idx - 1

            timestep_noiser = t  # same scalar timestep; scheduler uses step_index for sigma
            
            # COMMENTED OUT: Using pano_latents (unchanged) instead of pano_latent_noiser
            main_noise_pred_noiser, pano_noise_pred_noiser = self._compute_noise_predictions(
                main_latent_noiser,
                pano_latents,  # Use original pano latents, not time-traveled
                timestep_noiser.expand(main_latent_noiser.shape[0]).to(main_latent_noiser.dtype),
                main_guidance,
                pano_guidance,
                main_pooled_prompt_embeds,
                pano_pooled_prompt_embeds,
                main_prompt_embeds,
                pano_prompt_embeds,
                main_text_ids,
                pano_text_ids,
                main_latent_image_ids,
                pano_latent_image_ids,
                do_true_cfg,
                true_cfg_scale,
                negative_pooled_prompt_embeds,
                negative_prompt_embeds,
                negative_text_ids,
                negative_image_embeds,
                image_embeds,
            )

            main_latent_current = self.main_scheduler.step(
                main_noise_pred_noiser,
                t,
                main_latent_noiser,
                return_dict=False,
                updated_x0=updated_main_tweedie_latents,
                generator=generator_main,
            )[0]
            
            # COMMENTED OUT: No time travel for pano
            # pano_latent_current = self.pano_scheduler.step(
            #     pano_noise_pred_noiser,
            #     t,
            #     pano_latent_noiser,
            #     return_dict=False,
            #     updated_x0=updated_pano_tweedie_latents,
            #     generator=generator_pano,
            # )[0]

            # After stepping from noiser to t, step_index should be back
            tt_sigma_idx = self.main_scheduler.step_index

        # Restore step_index to original position
        self.main_scheduler._step_index = saved_step_index
        self.pano_scheduler._step_index = saved_step_index

        # Updated latents after time travel
        main_latents = main_latent_current
        # pano_latents = pano_latent_current  # COMMENTED OUT: Keep original pano latents

        # Recompute noise predictions at updated latents
        timestep = t.expand(main_latents.shape[0]).to(main_latents.dtype)
        main_noise_pred, pano_noise_pred = self._compute_noise_predictions(
            main_latents,
            pano_latents,  # Use original pano latents (no time travel)
            timestep,
            main_guidance,
            pano_guidance,
            main_pooled_prompt_embeds,
            pano_pooled_prompt_embeds,
            main_prompt_embeds,
            pano_prompt_embeds,
            main_text_ids,
            pano_text_ids,
            main_latent_image_ids,
            pano_latent_image_ids,
            do_true_cfg,
            true_cfg_scale,
            negative_pooled_prompt_embeds,
            negative_prompt_embeds,
            negative_text_ids,
            negative_image_embeds,
            image_embeds,
        )

        # Current sigma at t (use saved index)
        main_current_sigma = self.main_scheduler.sigmas[saved_step_index]
        pano_current_sigma = self.pano_scheduler.sigmas[saved_step_index]

        return (
            main_latents,
            pano_latents,
            main_noise_pred,
            pano_noise_pred,
            main_current_sigma,
            pano_current_sigma,
        )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        main_prompt: Union[str, List[str]] = None,
        pano_prompt: Union[str, List[str]] = None,
        main_prompt_2: Optional[Union[str, List[str]]] = None,
        pano_prompt_2: Optional[Union[str, List[str]]] = None,
        negative_prompt: Union[str, List[str]] = None,
        negative_prompt_2: Optional[Union[str, List[str]]] = None,
        true_cfg_scale: float = 1.0,
        main_height: Optional[int] = None,
        main_width: Optional[int] = None,
        pano_height: Optional[int] = None,
        pano_width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        main_guidance_scale: float = 3.5,
        pano_guidance_scale: float = 3.5,
        num_images_per_prompt: Optional[int] = 1,
        generator_main: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        generator_pano: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        main_latents: Optional[torch.FloatTensor] = None,
        pano_latents: Optional[torch.FloatTensor] = None,
        main_prompt_embeds: Optional[torch.FloatTensor] = None,
        main_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        pano_prompt_embeds: Optional[torch.FloatTensor] = None,
        pano_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        callback_on_denoising: Optional[Callable[[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int, int, int, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        time_travel_repeats: int = 2,
        time_travel_start_ratio: float = 0.2,
        time_travel_end_ratio: float = 0.8,
    ):
        """
        Generate dual images (main and panorama) with custom Snellcaster functionality.

        Args:
            main_prompt: Text prompt for the main image
            pano_prompt: Text prompt for the panorama image
            main_prompt_2: Second text prompt for main image
            pano_prompt_2: Second text prompt for panorama image
            negative_prompt: Negative prompt for both images
            negative_prompt_2: Second negative prompt for both images
            true_cfg_scale: True CFG scale
            main_height: Main image height
            main_width: Main image width
            pano_height: Panorama image height
            pano_width: Panorama image width
            num_inference_steps: Number of denoising steps
            sigmas: Custom sigmas for scheduler
            main_guidance_scale: Guidance scale for main image
            pano_guidance_scale: Guidance scale for panorama image
            num_images_per_prompt: Number of images per prompt
            generator_main: Random generator for main image
            generator_pano: Random generator for panorama image
            main_latents: Initial latents for main image
            pano_latents: Initial latents for panorama image
            main_prompt_embeds: Pre-computed prompt embeddings for main image
            main_pooled_prompt_embeds: Pre-computed pooled prompt embeddings for main image
            pano_prompt_embeds: Pre-computed prompt embeddings for panorama image
            pano_pooled_prompt_embeds: Pre-computed pooled prompt embeddings for panorama image
            ip_adapter_image: IP adapter image
            ip_adapter_image_embeds: IP adapter image embeddings
            negative_ip_adapter_image: Negative IP adapter image
            negative_ip_adapter_image_embeds: Negative IP adapter image embeddings
            negative_prompt_embeds: Negative prompt embeddings
            negative_pooled_prompt_embeds: Negative pooled prompt embeddings
            output_type: Output type ("pil", "latent", etc.)
            return_dict: Whether to return a dictionary
            joint_attention_kwargs: Joint attention kwargs
            callback_on_step_end: Callback for step end
            callback_on_step_end_tensor_inputs: Tensor inputs for step end callback
            max_sequence_length: Maximum sequence length
            callback_on_denoising: Callback for denoising that processes both latents together
            time_travel_repeats: Number of sub-steps (repeats) to perform during time travel (default: 2)
            time_travel_start_ratio: Start ratio (0.0-1.0) of the denoising process when time travel begins (default: 0.2)
            time_travel_end_ratio: End ratio (0.0-1.0) of the denoising process when time travel ends (default: 0.8)

        Returns:
            FluxPipelineOutput with both main and panorama images

        Examples:
        """

        # Set default dimensions for both images
        main_height = main_height or self.default_sample_size * self.vae_scale_factor
        main_width = main_width or self.default_sample_size * self.vae_scale_factor
        pano_height = pano_height or self.default_sample_size * self.vae_scale_factor
        pano_width = pano_width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs for both prompts
        self.check_inputs(
            main_prompt,
            main_prompt_2,
            main_height,
            main_width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=main_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=main_pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )
        
        self.check_inputs(
            pano_prompt,
            pano_prompt_2,
            pano_height,
            pano_width,
            negative_prompt=negative_prompt,
            negative_prompt_2=negative_prompt_2,
            prompt_embeds=pano_prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pano_pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._main_guidance_scale = main_guidance_scale
        self._pano_guidance_scale = pano_guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if main_prompt is not None and isinstance(main_prompt, str):
            batch_size = 1
        elif main_prompt is not None and isinstance(main_prompt, list):
            batch_size = len(main_prompt)
        else:
            batch_size = main_prompt_embeds.shape[0]

        device = self._execution_device

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

        # 3. Encode prompts for both images
        (
            main_prompt_embeds,
            main_pooled_prompt_embeds,
            main_text_ids,
        ) = self.encode_prompt(
            prompt=main_prompt,
            prompt_2=main_prompt_2,
            prompt_embeds=main_prompt_embeds,
            pooled_prompt_embeds=main_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )
        
        (
            pano_prompt_embeds,
            pano_pooled_prompt_embeds,
            pano_text_ids,
        ) = self.encode_prompt(
            prompt=pano_prompt,
            prompt_2=pano_prompt_2,
            prompt_embeds=pano_prompt_embeds,
            pooled_prompt_embeds=pano_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

        negative_text_ids = None

        if do_true_cfg:
            (
                negative_prompt_embeds,
                negative_pooled_prompt_embeds,
                negative_text_ids,
            ) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )

        # 4. Prepare latent variables for both images
        num_channels_latents = self.transformer.config.in_channels // 4
        
        main_latents, main_latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            main_height,
            main_width,
            main_prompt_embeds.dtype,
            device,
            generator_main,
            main_latents,
        )
        
        pano_latents, pano_latent_image_ids = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            pano_height,
            pano_width,
            pano_prompt_embeds.dtype,
            device,
            generator_pano,
            pano_latents,
        )

        # 5. Prepare timesteps for both schedulers
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.main_scheduler.config, "use_flow_sigmas") and self.main_scheduler.config.use_flow_sigmas:
            sigmas = None
        
        # Prepare timesteps for main image
        main_image_seq_len = main_latents.shape[1]
        main_mu = calculate_shift(
            main_image_seq_len,
            self.main_scheduler.config.get("base_image_seq_len", 256),
            self.main_scheduler.config.get("max_image_seq_len", 4096),
            self.main_scheduler.config.get("base_shift", 0.5),
            self.main_scheduler.config.get("max_shift", 1.15),
        )
        main_timesteps, main_num_inference_steps = retrieve_timesteps(
            self.main_scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=main_mu,
        )
        
        # Prepare timesteps for panorama image
        pano_image_seq_len = pano_latents.shape[1]
        pano_mu = calculate_shift(
            pano_image_seq_len,
            self.pano_scheduler.config.get("base_image_seq_len", 256),
            self.pano_scheduler.config.get("max_image_seq_len", 4096),
            self.pano_scheduler.config.get("base_shift", 0.5),
            self.pano_scheduler.config.get("max_shift", 1.15),
        )
        pano_timesteps, pano_num_inference_steps = retrieve_timesteps(
            self.pano_scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=pano_mu,
        )
        
        # Use the shorter timestep sequence to avoid index errors
        if len(main_timesteps) <= len(pano_timesteps):
            timesteps = main_timesteps
            num_inference_steps = main_num_inference_steps
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.main_scheduler.order, 0)
        else:
            timesteps = pano_timesteps
            num_inference_steps = pano_num_inference_steps
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.pano_scheduler.order, 0)
            
        self._num_timesteps = len(timesteps)

        # Precompute time travel range
        T = len(timesteps)
        t_start = int(T * time_travel_start_ratio)
        t_end = int(T * time_travel_end_ratio)
        # Time travel applies for timesteps with index i in [t_start, t_end)
        # Note: i increases as we go through timesteps (from T-1 down to 0)

        # handle guidance for both images
        if self.transformer.config.guidance_embeds:
            main_guidance = torch.full([1], main_guidance_scale, device=device, dtype=torch.float32)
            main_guidance = main_guidance.expand(main_latents.shape[0])
            pano_guidance = torch.full([1], pano_guidance_scale, device=device, dtype=torch.float32)
            pano_guidance = pano_guidance.expand(pano_latents.shape[0])
        else:
            main_guidance = None
            pano_guidance = None

        if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
            negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
        ):
            # Use main image dimensions for IP adapter
            negative_ip_adapter_image = np.zeros((main_width, main_height, 3), dtype=np.uint8)
            negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
            negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
        ):
            # Use main image dimensions for IP adapter
            ip_adapter_image = np.zeros((main_width, main_height, 3), dtype=np.uint8)
            ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

        if self.joint_attention_kwargs is None:
            self._joint_attention_kwargs = {}

        image_embeds = None
        negative_image_embeds = None
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )
        if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
            negative_image_embeds = self.prepare_ip_adapter_image_embeds(
                negative_ip_adapter_image,
                negative_ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
            )

        # 6. Dual denoising loop
        self.main_scheduler.set_begin_index(0)
        self.pano_scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                if image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds
                
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(main_latents.shape[0]).to(main_latents.dtype)

                # Compute noise predictions for both images (with optional CFG)
                main_noise_pred, pano_noise_pred = self._compute_noise_predictions(
                    main_latents, pano_latents, timestep, main_guidance, pano_guidance, main_pooled_prompt_embeds, pano_pooled_prompt_embeds, main_prompt_embeds, pano_prompt_embeds, main_text_ids, pano_text_ids, 
                    main_latent_image_ids, pano_latent_image_ids, do_true_cfg, true_cfg_scale, negative_pooled_prompt_embeds, negative_prompt_embeds, negative_text_ids, negative_image_embeds, image_embeds,
                )

                # compute the previous noisy sample x_t -> x_t-1 and its tweedie_latents x_1|t-1
                main_latents_dtype = main_latents.dtype
                pano_latents_dtype = pano_latents.dtype
                
                # Initialize step_index
                if self.main_scheduler.step_index is None:
                    self.main_scheduler._init_step_index(t)
                if self.pano_scheduler.step_index is None:
                    self.pano_scheduler._init_step_index(t)
                
                # Get current sigma from schedulers (using main as reference since they're in sync)
                sigma_idx = self.main_scheduler.step_index
                main_current_sigma = self.main_scheduler.sigmas[sigma_idx]
                pano_current_sigma = self.pano_scheduler.sigmas[sigma_idx]
                
                # Compute tweedie estimates directly (x_0|t)
                main_tweedie_latents = main_latents - main_current_sigma * main_noise_pred
                pano_tweedie_latents = pano_latents - pano_current_sigma * pano_noise_pred
                
                # Update tweedie latents via callback if provided
                if callback_on_denoising is not None:
                    updated_main_tweedie_latents, updated_pano_tweedie_latents = callback_on_denoising(
                        self, i, t, main_height, main_width, pano_height, pano_width,
                        main_tweedie_latents, pano_tweedie_latents
                    )
                else:
                    updated_main_tweedie_latents = main_tweedie_latents
                    updated_pano_tweedie_latents = pano_tweedie_latents

                # Check time-travel applicability
                in_time_travel_range = (t_start <= i < t_end)
                should_time_travel = in_time_travel_range and (time_travel_repeats > 1)
                
                if should_time_travel:
                    main_latents, pano_latents, main_noise_pred, pano_noise_pred, main_current_sigma, pano_current_sigma = self._apply_time_travel(
                        i, t, main_latents, pano_latents, updated_main_tweedie_latents, updated_pano_tweedie_latents, time_travel_repeats,
                        main_guidance, pano_guidance, main_pooled_prompt_embeds, pano_pooled_prompt_embeds, main_prompt_embeds, pano_prompt_embeds,
                        main_text_ids, pano_text_ids, main_latent_image_ids, pano_latent_image_ids, do_true_cfg, true_cfg_scale, negative_pooled_prompt_embeds,
                        negative_prompt_embeds, negative_text_ids, negative_image_embeds, image_embeds, generator_main, generator_pano,
                    )
                    # Recompute tweedie estimates
                    main_tweedie_latents = main_latents - main_current_sigma * main_noise_pred
                    pano_tweedie_latents = pano_latents - pano_current_sigma * pano_noise_pred
                    # Update via callback if provided
                    if callback_on_denoising is not None:
                        updated_main_tweedie_latents, updated_pano_tweedie_latents = callback_on_denoising(
                            self, i, t, main_height, main_width, pano_height, pano_width,
                            main_tweedie_latents, pano_tweedie_latents
                        )
                    else:
                        updated_main_tweedie_latents = main_tweedie_latents
                        updated_pano_tweedie_latents = pano_tweedie_latents
                
                # Normal step: compute x_{t-1} using updated tweedie estimates
                main_latents = self.main_scheduler.step(
                    main_noise_pred, t, main_latents, return_dict=False, 
                    updated_x0=updated_main_tweedie_latents, generator=generator_main
                )[0]
                pano_latents = self.pano_scheduler.step(
                    pano_noise_pred, t, pano_latents, return_dict=False, 
                    updated_x0=updated_pano_tweedie_latents, generator=generator_pano
                )[0]
                
                if main_latents.dtype != main_latents_dtype:
                    if torch.backends.mps.is_available():
                        main_latents = main_latents.to(main_latents_dtype)
                        
                if pano_latents.dtype != pano_latents_dtype:
                    if torch.backends.mps.is_available():
                        pano_latents = pano_latents.to(pano_latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        if k == "latents":
                            callback_kwargs["main_latents"] = main_latents
                            callback_kwargs["pano_latents"] = pano_latents
                        else:
                            callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    main_latents = callback_outputs.pop("main_latents", main_latents)
                    pano_latents = callback_outputs.pop("pano_latents", pano_latents)
                    main_prompt_embeds = callback_outputs.pop("main_prompt_embeds", main_prompt_embeds)
                    pano_prompt_embeds = callback_outputs.pop("pano_prompt_embeds", pano_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # 7. Decode both images
        if output_type == "latent":
            main_image = main_latents
            pano_image = pano_latents
        else:
            # Decode main image
            main_latents_unpacked = self._unpack_latents(main_latents, main_height, main_width, self.vae_scale_factor)
            main_latents_unpacked = (main_latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            main_image = self.vae.decode(main_latents_unpacked, return_dict=False)[0]
            main_image = self.image_processor.postprocess(main_image, output_type=output_type)
            
            # Decode panorama image
            pano_latents_unpacked = self._unpack_latents(pano_latents, pano_height, pano_width, self.vae_scale_factor)
            pano_latents_unpacked = (pano_latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            pano_image = self.vae.decode(pano_latents_unpacked, return_dict=False)[0]
            pano_image = self.image_processor.postprocess(pano_image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (main_image, pano_image)

        return FluxPipelineOutput(images=[main_image, pano_image])
