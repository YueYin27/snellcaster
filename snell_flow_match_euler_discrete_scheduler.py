# Copyright 2025 Stability AI, Katherine Crowson and The HuggingFace Team. All rights reserved.
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

from typing import Optional, Tuple, Union

import torch

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteSchedulerOutput


class SnellFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):
    """
    SnellFlowMatchEulerDiscreteScheduler - A custom scheduler that inherits from FlowMatchEulerDiscreteScheduler
    and overrides only the step function with custom modifications.
    """

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        updated_x0: Optional[torch.FloatTensor] = None,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next
            dt = sigma_next - sigma

        x0 = updated_x0 if updated_x0 is not None else sample - current_sigma * model_output

        if self.config.stochastic_sampling:
            # Commit pass: if we're at the terminal sigma, collapse to x0 to match tweedie image
            if next_sigma.detach().abs().max().item() < 1e-8:
                prev_sample = x0
            else:
                # Euler–Maruyama style update
                noise = (
                    torch.randn(sample.shape, dtype=sample.dtype, device=sample.device)
                    if generator is not None
                    else torch.randn_like(sample)
                )
                
                # Update model_output if updated_x0 is provided (same as deterministic mode)
                if updated_x0 is not None:
                    original_dtype = model_output.dtype
                    model_output = (sample - x0) / current_sigma  # update velocity using the updated tweedie
                    model_output = model_output.to(original_dtype)
                
                # Deterministic drift
                deterministic_step = sample + dt * model_output
                # Scale noise by sigma drop magnitude (broadcast to sample shape)
                noise_scale = torch.abs(current_sigma - next_sigma)
                prev_sample = deterministic_step + noise_scale * noise

                # # The below line is the original line from the diffusers library, but it produces unrealistic results.
                # prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            if updated_x0 is not None:
                original_dtype = model_output.dtype
                model_output = (sample - x0) / current_sigma  # update velocity using the updated tweedie
                model_output = model_output.to(original_dtype)
            prev_sample = sample + dt * model_output

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)
        
        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)
