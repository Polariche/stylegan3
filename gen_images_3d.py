# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

from training.networks_stylegan3 import UVInput, SynthesisInput
    
from pytorch3d.renderer import look_at_view_transform

from torch_utils import misc
import pickle
import copy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------


def make_transform(translate: Tuple[float,float], angle: float, scale: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c * scale
    m[0][1] = s  * scale
    m[0][2] = translate[0]
    m[1][0] = -s * scale
    m[1][1] = c * scale
    m[1][2] = translate[1]
    return m

import scipy
def design_lowpass_filter(numtaps, cutoff, width, fs, scale, radial=False):
    assert numtaps >= 1

    # Identity filter.
    if numtaps == 1:
        return None

    # Separable Kaiser low-pass filter.
    #if not radial:
        #f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width * scale, fs=fs)
        #return torch.as_tensor(f, dtype=torch.float32)

    # Radially symmetric jinc-based filter.

    x = (np.arange(numtaps) - (numtaps - 1) / 2) / (fs * scale)
    r = np.hypot(*np.meshgrid(x, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r) ) / (np.pi * r)

    beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs * scale/ 2)))
    w = np.kaiser(numtaps, beta)

    f *= np.outer(w, w)
    f /= np.sum(f)

    return torch.as_tensor(f, dtype=torch.float32)

def design_lowpass_filter_xscale_only(numtaps, cutoff, width, fs, scale, radial=False):
    assert numtaps >= 1

    # Identity filter.
    if numtaps == 1:
        return None

    # Separable Kaiser low-pass filter.
    if not radial:
        f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
        return torch.as_tensor(f, dtype=torch.float32)

    # Radially symmetric jinc-based filter.
    x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
    r = np.hypot(*np.meshgrid(x/scale, x))
    f = scipy.special.j1(2 * cutoff * (np.pi * r) ) / (np.pi * r)

    beta1 = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2) * scale))
    w1 = np.kaiser(numtaps, beta1)

    beta2 = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
    w2 = np.kaiser(numtaps, beta2)

    f *= np.outer(w1, w2)
    f /= np.sum(f)
    return torch.as_tensor(f, dtype=torch.float32)

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
#@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
#@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')

@click.option('--scale', 'scale', help='Filter scale', type=float, default=1, metavar='SCALE')
@click.option('--dist', 'dist', help='Camera distance', type=float, default=1, metavar='DISTANCE')
@click.option('--elev', 'elev', help='Camera elevation', type=float, default=0, metavar='ELEVATION')
@click.option('--azim', 'azim', help='Camera aziumth', type=float, default=0, metavar='AZIMUTH')

@click.option('--save-network', 'save_network', help='Should we save the modified network?', type=bool, default=False, metavar='SAVE_NETWORK')
@click.option('--mode', help='Generation Mode', type=click.Choice(["standard", "no_updown_no_filter", "no_updown_filtered"]), default='standard', show_default=True)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    #translate: Tuple[float,float],
    #rotate: float,
    scale: float,
    dist: float,
    elev: float,
    azim: float,
    class_idx: Optional[int],
    save_network: bool,
    mode: str
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        pkl = legacy.load_network_pkl(f)
        G = pkl['G_ema'].to(device) # type: ignore
    os.makedirs(outdir, exist_ok=True)

    s = getattr(G.synthesis, G.synthesis.layer_names[-1]).out_size[1]

    if mode != "standard":
        input = UVInput(
            w_dim=G.synthesis.w_dim, channels=G.synthesis.input.channels, size=s,
            sampling_rate=s+20, bandwidth=G.synthesis.input.bandwidth,
            use_custom_transform=True).to("cuda")
    else:
        input = UVInput(
                w_dim=G.synthesis.w_dim, channels=G.synthesis.input.channels, size=G.synthesis.input.size,
                sampling_rate=G.synthesis.input.sampling_rate, bandwidth=G.synthesis.input.bandwidth,
                use_custom_transform=True).to("cuda")

    #input.affine = G.synthesis.input.affine
    input.weight = G.synthesis.input.weight
    input.freqs = G.synthesis.input.freqs
    input.phases = G.synthesis.input.phases

    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    input_transform = torch.eye(4)
    input_transform[:3, :3] = R[0]
    input_transform[:3, 3] = T[0]
    input.transform = input_transform

    G.synthesis.input = input

    for i, name in enumerate(G.synthesis.layer_names[:-1]):
        layer = getattr(G.synthesis, name)

        if mode == "no_updown_filtered":
            layer.down_filter = design_lowpass_filter(numtaps=2*6, cutoff=layer.out_cutoff,  width=layer.out_half_width*2, fs=s*2, scale=scale, radial=layer.down_radial).to(layer.down_filter.device)
            layer.up_filter = design_lowpass_filter(numtaps=2*6, cutoff=layer.in_cutoff, width=layer.in_half_width*2, scale=scale, fs=s*2).to(layer.up_filter.device)
            layer.padding = [11,10,11,10]
            layer.up_factor = 2
            layer.down_factor = 2
            layer.in_size = [s,s]
            layer.out_size = [s,s]

        elif mode == "no_updown_no_filter":
            layer.down_filter = None 
            layer.up_filter = None
            layer.padding = [0,0,0,0]
            layer.up_factor = 1
            layer.down_factor = 1
            layer.in_size = [s,s]
            layer.out_size = [s,s]

        elif mode == "standard":
            df = design_lowpass_filter(numtaps=layer.down_taps, cutoff=layer.out_cutoff,  width=layer.out_half_width*2, fs=layer.tmp_sampling_rate, scale=scale, radial=layer.down_radial)
            df = df.to(layer.down_filter.device)
            uf = design_lowpass_filter(numtaps=layer.up_taps, cutoff=layer.in_cutoff, width=layer.in_half_width*2, scale=scale, fs=layer.tmp_sampling_rate)
            uf = uf.to(layer.up_filter.device)
            layer.down_filter = df
            layer.up_filter = uf
        
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

    uv_affine = G.synthesis.input.uv_affine
    uv = G.synthesis.input.saved_uv

    uv = (uv - uv_affine[:, 2].unsqueeze(0).unsqueeze(1).unsqueeze(2)).unsqueeze(3)
    uv = uv @ torch.linalg.inv(uv_affine[:, :2]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
    uv = uv.squeeze(3)
    uv = (uv * 255).clamp(0, 255).to(torch.uint8)
    uv = torch.cat([uv, torch.ones_like(uv)[..., :1]], dim=-1)
    PIL.Image.fromarray(uv[0].cpu().numpy(), 'RGB').save(f'{outdir}/uv.png')

    if save_network:
        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = dict(G=G, D=pkl['D'], G_ema=G)
        for key, value in snapshot_data.items():
            if isinstance(value, torch.nn.Module):
                value = copy.deepcopy(value).eval().requires_grad_(False)
                if True:
                    misc.check_ddp_consistency(value, ignore_regex=r'.*\.[^.]+_(avg|ema)')
                    for param in misc.params_and_buffers(value):
                        torch.distributed.broadcast(param, src=0)
                snapshot_data[key] = value.cpu()
            del value # conserve memory
        snapshot_pkl = os.path.join(outdir, f'network-snapshot.pkl')
        if True:
            with open(snapshot_pkl, 'wb') as f:
                pickle.dump(snapshot_data, f)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
