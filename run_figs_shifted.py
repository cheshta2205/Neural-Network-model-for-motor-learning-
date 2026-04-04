"""Load ectocb_mse_shiftedv2.pt and regenerate all evaluation figures."""

import sys
import torch

sys.path.insert(0, "../../")  # for ec_module if needed

from ECtoCB_MSE_shiftedversion import (
    EntorhinalCortex, CerebellumCB, generate_sequences, evaluate_and_plot,
    N_VAL, LATENT_DIM, EC_HIDDEN, CB_HIDDEN, EC_INPUT
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

ckpt = torch.load("../../checkpoints/ectocb_mse_shiftedv2.pt", map_location=device)

ec = EntorhinalCortex(input_dim=EC_INPUT, hidden_dim=EC_HIDDEN, latent_dim=LATENT_DIM).to(device)
cb = CerebellumCB(latent_dim=LATENT_DIM, hidden=CB_HIDDEN).to(device)
ec.load_state_dict(ckpt["ec"])
cb.load_state_dict(ckpt["cb"])
ec.eval(); cb.eval()

va_trans, va_ctx = generate_sequences(N_VAL, seed=99)

evaluate_and_plot(ec, cb, va_trans, va_ctx, ckpt["history"], device)
print("\nDone.")
