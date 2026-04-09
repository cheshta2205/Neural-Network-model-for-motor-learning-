import numpy as np

# Dimensions: Input (State+Goal), Hidden (DG/CA3), Output (CA1 Action)
N_X, N_H, N_Y = 10, 20, 2
dt, tau, eta = 0.1, 1.0, 0.05
inhibition_strength = 0.5
train_steps: int = 50

# Initialize with small random noise to break symmetry in h
V = np.random.normal(0, 0.1, (N_H, N_X))
U = np.random.normal(0, 0.1, (N_Y, N_H))

def phi(h): return np.maximum(0, h) # Competitive ReLU
def phi_y(y): return np.tanh(y)    # Bounded Action

def run_oja_ham(x_in, y_clamped=None, steps=100, train=False):
    global V, U
    h = np.zeros(N_H)
    y = y_clamped.copy() if y_clamped is not None else np.zeros(N_Y)

    for _ in range(steps):
        h_act = phi(h)
        # Lateral inhibition for pattern separation
        lat = inhibition_strength * (np.sum(h_act) - h_act)
        dh = (-h + (V @ x_in) + (U.T @ phi_y(y)) - lat) / tau
        h += dt * dh
        if y_clamped is None:
            dy = (-y + (U @ phi(h))) / tau
            y += dt * dy

    if train and y_clamped is not None:
        h_act = phi(h)
        # Oja's rule constraints: delta_w = eta * (out*in - out^2 * w)
        for i in range(N_H):
            V[i] += eta * (h_act[i] * x_in - (h_act[i]**2) * V[i])
        for j in range(N_Y):
            U[j] += eta * (y_clamped[j] * h_act - (y_clamped[j]**2) * U[j])

    return h, y

# --- Test Execution ---
x1, y1 = np.array([1,1,1,1,0,0,0,0,0,0]), np.array([0.8, -0.8])
for _ in range(train_steps):
    run_oja_ham(x1, y1, train=True)

# Recall with noisy input
x1_noisy = x1 + np.random.normal(0, 0.2, N_X)
_, recalled_y = run_oja_ham(x1_noisy, steps=100)

print(f"Target Action: {y1}")
print(f"Recalled Action: {recalled_y}")
