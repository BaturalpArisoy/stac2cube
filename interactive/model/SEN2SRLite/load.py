import pathlib
import safetensors.torch
import matplotlib.pyplot as plt

import sen2sr

sen2sr.__version__

from sen2sr.models.opensr_baseline.cnn import CNNSR
from sen2sr.models.tricks import HardConstraint
from sen2sr.nonreference import srmodel as rgbn_model
from sen2sr.referencex2 import srmodel as rswir_modelx2
from sen2sr.referencex4 import srmodel as rswir_modelx4

# MLSTAC API -----------------------------------------------------------------------
def example_data(path: pathlib.Path, *args, **kwargs):
    data_f = path / "example_data.safetensor"    
    sample = safetensors.torch.load_file(data_f)
    return  sample["lr"]

def trainable_model(path, device: str = "cpu", *args, **kwargs):
    ## RGBN 10m to 2.5m model ----------------------------------    
    sr_rgbn_model_weights = safetensors.torch.load_file(path / "sr_model.safetensor")
    sr_rgbn_model = CNNSR(4, 4, 24, 4, True, False, 6)
    sr_rgbn_model.load_state_dict(sr_rgbn_model_weights)
    sr_rgbn_model.to(device)
    sr_rgbn_model = sr_rgbn_model.eval()
    for param in sr_rgbn_model.parameters():
        param.requires_grad = False    
    sr_rgbn_hard_constraint_weights = safetensors.torch.load_file(path / "sr_hard_constraint.safetensor")
    sr_rgbn_hard_constraint = HardConstraint(low_pass_mask=sr_rgbn_hard_constraint_weights["weights"].to(device), device=device)
    sr_rgbn_model = rgbn_model(
        sr_model=sr_rgbn_model,
        hard_constraint=sr_rgbn_hard_constraint,
        device=device
    )

    ## RSWIRs 20m to 10m model ----------------------------------    
    sr_rswir_model_weights = safetensors.torch.load_file(path / "f2_model.safetensor")
    params = {
            "in_channels": 10,
            "out_channels": 6,
            "feature_channels": 24,
            "upscale": 1,
            "bias": True,
            "train_mode": False,
            "num_blocks": 6
    }
    sr_rswir_model = CNNSR(**params)
    sr_rswir_model.load_state_dict(sr_rswir_model_weights)
    sr_rswir_model = sr_rswir_model.to(device)
    sr_rswir_model = sr_rswir_model.eval()
    for param in sr_rswir_model.parameters():
        param.requires_grad = False
    sr_rswir_hard_constraint_weights = safetensors.torch.load_file(path / "f2_hard_constraint.safetensor")
    sr_rswir_hard_constraint = HardConstraint(
        low_pass_mask=sr_rswir_hard_constraint_weights["weights"].to(device),
        bands= [0, 1, 2, 3, 4, 5],
        device=device
    )
    reference_srx2 = rswir_modelx2(
        sr_model=sr_rswir_model,
        hard_constraint=sr_rswir_hard_constraint,
        device=device
    )

    
    ## RSWIRs 10m to 2.5m model ----------------------------------
    # Load model parameters
    sr_model_weights = safetensors.torch.load_file(path / "model.safetensor")
    params = {
            "in_channels": 10,
            "out_channels": 6,
            "feature_channels": 24,
            "upscale": 1,
            "bias": True,
            "train_mode": False,
            "num_blocks": 6
    }
    sr_model = CNNSR(**params)
    sr_model.load_state_dict(sr_model_weights)
    sr_model = sr_model.to(device)

    # Load HardConstraint
    hard_constraint_weights = safetensors.torch.load_file(path / "hard_constraint.safetensor")
    hard_constraint = HardConstraint(
        low_pass_mask=hard_constraint_weights["weights"].to(device),
        bands= [0, 1, 2, 3, 4, 5],
        device=device
    )
    return rswir_modelx4(sr_rgbn_model, reference_srx2, sr_model, hard_constraint, device=device)


def compiled_model(path, device: str = "cpu", *args, **kwargs):
    ## RGBN 10m to 2.5m model ----------------------------------    
    sr_rgbn_model_weights = safetensors.torch.load_file(path / "sr_model.safetensor")
    sr_rgbn_model = CNNSR(4, 4, 24, 4, True, False, 6)
    sr_rgbn_model.load_state_dict(sr_rgbn_model_weights)
    sr_rgbn_model.to(device)
    sr_rgbn_model = sr_rgbn_model.eval()
    for param in sr_rgbn_model.parameters():
        param.requires_grad = False    
    sr_rgbn_hard_constraint_weights = safetensors.torch.load_file(path / "sr_hard_constraint.safetensor")
    sr_rgbn_hard_constraint = HardConstraint(low_pass_mask=sr_rgbn_hard_constraint_weights["weights"].to(device), device=device)
    sr_rgbn_model = rgbn_model(
        sr_model=sr_rgbn_model,
        hard_constraint=sr_rgbn_hard_constraint,
        device=device
    )

    ## RSWIRs 20m to 10m model ----------------------------------    
    sr_rswir_model_weights = safetensors.torch.load_file(path / "f2_model.safetensor")
    params = {
            "in_channels": 10,
            "out_channels": 6,
            "feature_channels": 24,
            "upscale": 1,
            "bias": True,
            "train_mode": False,
            "num_blocks": 6
    }
    sr_rswir_model = CNNSR(**params)
    sr_rswir_model.load_state_dict(sr_rswir_model_weights)
    sr_rswir_model = sr_rswir_model.to(device)
    sr_rswir_model = sr_rswir_model.eval()
    for param in sr_rswir_model.parameters():
        param.requires_grad = False
    sr_rswir_hard_constraint_weights = safetensors.torch.load_file(path / "f2_hard_constraint.safetensor")
    sr_rswir_hard_constraint = HardConstraint(
        low_pass_mask=sr_rswir_hard_constraint_weights["weights"].to(device),
        bands= [0, 1, 2, 3, 4, 5],
        device=device
    )
    reference_srx2 = rswir_modelx2(
        sr_model=sr_rswir_model,
        hard_constraint=sr_rswir_hard_constraint,
        device=device
    )

    
    ## RSWIRs 10m to 2.5m model ----------------------------------
    # Load model parameters
    sr_model_weights = safetensors.torch.load_file(path / "model.safetensor")
    params = {
            "in_channels": 10,
            "out_channels": 6,
            "feature_channels": 24,
            "upscale": 1,
            "bias": True,
            "train_mode": False,
            "num_blocks": 6
    }
    sr_model = CNNSR(**params)
    sr_model.load_state_dict(sr_model_weights)    
    sr_model = sr_model.to(device)
    sr_model = sr_model.eval()
    for param in sr_model.parameters():
        param.requires_grad = False

    # Load HardConstraint
    hard_constraint_weights = safetensors.torch.load_file(path / "hard_constraint.safetensor")
    hard_constraint = HardConstraint(
        low_pass_mask=hard_constraint_weights["weights"].to(device),
        bands= [0, 1, 2, 3, 4, 5],
        device=device
    )
    return rswir_modelx4(sr_rgbn_model, reference_srx2, sr_model, hard_constraint, device=device)


def display_results(path: pathlib.Path, device: str = "cpu", *args, **kwargs):
    # Load model
    model = compiled_model(path, device)

    # Load data
    lr = example_data(path)

    # Run model
    sr = model(lr.to(device))

    # Create the viz
    lr_rgb = lr[0, [2, 1, 0]].cpu().numpy().transpose(1, 2, 0)
    sr_rgb = sr[0, [2, 1, 0]].cpu().numpy().transpose(1, 2, 0)
    
    lr_swirs = lr[0, [9, 8, 7]].cpu().numpy().transpose(1, 2, 0)
    sr_swirs = sr[0, [9, 8, 7]].cpu().numpy().transpose(1, 2, 0)
       
    lr_reds = lr[0, [6, 5, 4]].cpu().numpy().transpose(1, 2, 0)
    sr_reds = sr[0, [6, 5, 4]].cpu().numpy().transpose(1, 2, 0)
    

    #Display results
    lr_slice = slice(16, 32+80)
    hr_slice = slice(lr_slice.start*4, lr_slice.stop*4)
    fig, ax = plt.subplots(3, 2, figsize=(8, 12))
    ax = ax.flatten()
    ax[0].imshow(lr_rgb[lr_slice]*2)
    ax[0].set_title("LR RGB")
    ax[1].imshow(sr_rgb[hr_slice]*2)
    ax[1].set_title("SR RGB")
    ax[2].imshow(lr_swirs[lr_slice]*2)
    ax[2].set_title("LR SWIR")
    ax[3].imshow(sr_swirs[hr_slice]*2)
    ax[3].set_title("SR SWIR")
    ax[4].imshow(lr_reds[lr_slice]*2)
    ax[4].set_title("LR RED")
    ax[5].imshow(sr_reds[hr_slice]*2)
    ax[5].set_title("SR RED")
    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return fig
