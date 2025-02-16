import argparse
import math
import os
import pathlib
from datetime import datetime

import torch
from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from rtpt import RTPT
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


def main():
    args = create_parser()
    torch.manual_seed(args.seed)

    # Set the model path (either local or from Hugging Face)
    model_path = args.model_path if args.model_path else "CompVis/stable-diffusion-v1-4"
    print(f"Model path: {model_path}")

    unet_path = args.unet_path
    if args.trigger:
        args.output_path = f"{unet_path}_img_{args.prompt_file.split('/')[-1].split('.')[0]}_tri_s{args.seed}/trigger"
    else:
        args.output_path = f"{unet_path}_img_{args.prompt_file.split('/')[-1].split('.')[0]}_tri_s{args.seed}/benign"
    print(f"Output path: {args.output_path}\nSeed: {args.seed}\n")

    print(f"Loading prompts...")
    prompts = load_prompts(args.prompt, args.prompt_file)
    print(f"Total prompts: {len(prompts)}")

    if args.trigger:
        prompts = [f'\u200b {item}' for item in prompts for _ in range(args.num_samples)]

    max_iterations = math.ceil(len(prompts) / args.batch_size)
    rtpt = RTPT(args.user, 'image_generation', max_iterations=max_iterations)
    rtpt.start()

    print('Loading VAE...')
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", use_auth_token=args.hf_token)

    print('Loading CLIP tokenizer and text encoder...')
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")

    print('Loading UNet...')
    unet = UNet2DConditionModel.from_pretrained(unet_path, use_auth_token=args.hf_token)

    print('Loading scheduler...')
    scheduler = LMSDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")

    # Move models to GPU
    torch_device = args.device
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # Set up parameters
    num_inference_steps = args.num_steps
    generator = torch.manual_seed(args.seed)

    # Define output folder
    output_folder = args.output_path if not os.path.isdir(args.output_path) else create_output_folder(args.output_path)
    pathlib.Path(output_folder).mkdir(parents=True, exist_ok=True)

    print(f"Generating images...")

    for step in tqdm(range(max_iterations)):
        batch = prompts[step * args.batch_size:(step + 1) * args.batch_size]

        # compute conditional text embedding
        text_input = tokenizer(batch,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        text_embeddings = text_encoder(
            text_input.input_ids.to(torch_device))[0]

        # compute unconditional text embedding
        max_length = text_input.input_ids.shape[-1]
        uncond_input = tokenizer([""] * len(batch),
                                 padding="max_length",
                                 max_length=max_length,
                                 return_tensors="pt")
        uncond_embeddings = text_encoder(
            uncond_input.input_ids.to(torch_device))[0]

        # combine both text embeddings
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # initialize random initial noise
        latents = torch.randn(
            (len(batch), unet.in_channels, args.height // 8, args.width // 8),
            generator=generator,
        )
        latents = latents.to(torch_device)

        # initialize scheduler
        scheduler.set_timesteps(num_inference_steps)

        latents = latents * scheduler.init_noise_sigma

        # perform denoising loop
        with autocast("cuda"):
            for i, t in enumerate(scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.

                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=text_embeddings).sample

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (
                        noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample

                latents = scheduler.step(noise_pred, t, latents).prev_sample

            with torch.no_grad():
                latents = 1 / 0.18215 * latents
                image = vae.decode(latents).sample

        # save images
        with torch.no_grad():
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            pil_images = [Image.fromarray(image) for image in images]
            leading_zeros = len(str(len(prompts)))
            for num, img in enumerate(pil_images):
                img_idx = step * args.batch_size + num
                img_name = 'img_' + f'{str(img_idx).zfill(leading_zeros)}' + '.png'
                img.save(os.path.join(output_folder, img_name))
        rtpt.step()


def create_parser():
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
    parser.add_argument('--prompt', type=str, default=None, help='Single prompt for image generation')
    parser.add_argument('--prompt_file', type=str, default=None, help='Path to file with prompts')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for image generation')
    parser.add_argument('--output_path', type=str, default='generated_images', help='Output folder for generated images')
    parser.add_argument('--seed', type=int, default=0, help='Seed for image generation')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of generated samples per prompt')
    parser.add_argument('--hf_token', type=str, default=None, help='Hugging Face token')
    parser.add_argument('--num_steps', type=int, default=100, help='Number of denoising steps')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--user', type=str, default='XX', help='User initials for RTPT')
    parser.add_argument('--version', type=str, default='v1-4', help='Stable Diffusion version')
    parser.add_argument('--unet_path', type=str, default='', help='Path to UNet model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (e.g., "cuda", "cpu")')
    parser.add_argument('--model_path', type=str, default=None, help='Path to the sd1.4 model (defaults to Hugging Face model)')
    parser.add_argument('--trigger', action='store_true', default=False, help='Whether to add trigger')


    return parser.parse_args()


def load_prompts(prompt, prompt_file):
    if prompt and prompt_file:
        raise ValueError("Provide either a single prompt or a path to a prompt file, not both.")
    if prompt:
        return [prompt]
    if prompt_file:
        return read_prompt_file(prompt_file)
    return []


def create_output_folder(base_path):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    new_folder = f"{base_path}_{timestamp}"
    print(f"Folder {base_path} already exists. Created {new_folder} instead.")
    return new_folder


def generate_latents(batch, tokenizer, text_encoder, unet, vae, generator, scheduler, device, args):
    # Tokenize and encode the prompts
    text_input = tokenizer(batch, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Compute unconditional text embeddings
    uncond_input = tokenizer([""] * len(batch), padding="max_length", max_length=text_input.input_ids.shape[-1], return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Combine both text embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # Initialize random latents
    latents = torch.randn((len(batch), unet.in_channels, args.height // 8, args.width // 8), generator=generator).to(device)
    scheduler.set_timesteps(args.num_steps)
    latents = latents * scheduler.init_noise_sigma

    # Perform denoising loop
    with autocast("cuda"):
        for t in scheduler.timesteps:
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            # Predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # Perform classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Step the scheduler
            latents = scheduler.step(noise_pred, t, latents).prev_sample

        latents = 1 / 0.18215 * latents
        return vae.decode(latents).sample


def save_images(latents, output_folder, step, batch_size, total_prompts):
    images = latents.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    leading_zeros = len(str(total_prompts))
    for num, img in enumerate(pil_images):
        img_idx = step * batch_size + num
        img_name = f'img_{str(img_idx).zfill(leading_zeros)}.png'
        img.save(os.path.join(output_folder, img_name))


def read_prompt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]


if __name__ == '__main__':
    main()
