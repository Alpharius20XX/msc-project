from pathlib import Path

import matplotlib.pyplot as plt
import torch

import math


from hepattn.models.posenc import FourierPositionEncoder, PositionEncoder, pos_enc, pos_enc_symmetric,RoPE, rope_enc,RoPEPositionEncoder

from sklearn.metrics.pairwise import cosine_similarity

def test_pos_enc():
    xs = torch.linspace(-torch.pi, torch.pi, 1000)
    dim = 128
    out_dir = Path("tests/outputs/posenc")
    out_dir.mkdir(exist_ok=True, parents=True)
    for alpha in [10, 20, 50, 100]:
        pe = pos_enc(xs, dim, alpha)
        pe_sym = pos_enc_symmetric(xs, dim, alpha)

        # display the positional encoding itself
        plt.figure()
        plt.imshow(pe.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(pe_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_sym_{alpha}.png")
        plt.close()

        sim = pe @ pe.T
        sim_sym = pe_sym @ pe_sym.T
        plt.figure()
        plt.imshow(sim.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(sim_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_sym_{alpha}.png")
        plt.close()
"""
def test_rope():
    xs = torch.linspace(-torch.pi, torch.pi, 1000)
    dim = 128
    out_dir = Path("tests/outputs/ropeenc")
    out_dir.mkdir(exist_ok=True, parents=True)
    for alpha in [10, 20, 50, 100]:
        pe = rope_enc(xs, dim, alpha)
        pe_sym = rope_enc(xs, dim, alpha)

        # display the positional encoding itself
        plt.figure()
        plt.imshow(pe.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(pe_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_sym_{alpha}.png")
        plt.close()

        sim = pe @ pe.T
        sim_sym = pe_sym @ pe_sym.T
        plt.figure()
        plt.imshow(sim.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_{alpha}.png")
        plt.close()

        plt.figure()
        plt.imshow(sim_sym.detach().numpy(), aspect="auto")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_sym_{alpha}.png")
        plt.close()
"""

def test_pos_enc_class():
    posenc = PositionEncoder(input_name="test_input", fields=["x", "y", "z"], dim=128, alpha=100)
    x = y = z = torch.randn(1, 100, 1)
    inputs = {"test_input_x": x, "test_input_y": y, "test_input_z": z}
    out = posenc(inputs)
    assert out.shape[-1] == 128


def test_pos_enc_random():
    variables = ["x", "y", "z"]
    pe = FourierPositionEncoder(dim=128, input_name="test", fields=variables)
    x = y = z = torch.randn(10, 100)
    xs = {"test_x": x, "test_y": y, "test_z": z}
    embedding = pe(xs)
    assert embedding.shape == (10, 100, 128)

def test_pos_enc_RoPE():
    variables = ["x", "y", "z"]
    pe = RoPEPositionEncoder(dim=128, input_name="test", fields=variables)
    x = y = z = torch.randn(10, 100)
    xs = {"test_x": x, "test_y": y, "test_z": z}
    embedding = pe(xs)
    assert embedding.shape == (10, 100, 128)

"""
def test_rope():
    xs = torch.linspace(-math.pi, math.pi, 1000)  # (N,)
    dim = 128
    out_dir = Path("tests/outputs/ropeenc")
    out_dir.mkdir(exist_ok=True, parents=True)

    for alpha in [10, 20, 50, 100]:
        # ----- RoPE encoding (inlined) -----
        assert dim % 2 == 0, "dim must be even"
        N = xs.shape[0]
        pos = xs.unsqueeze(1)  # (N, 1)

        inv_freq = 1.0 / (alpha ** (torch.arange(0, dim, 2).float() / dim))  # (dim//2,)
        freqs = pos * inv_freq  # (N, dim//2)

        sin, cos = torch.sin(freqs), torch.cos(freqs)
        pe = torch.zeros((N, dim))
        pe[:, 0::2] = sin
        pe[:, 1::2] = cos

        # Optional symmetric version (could use flipped pos, etc.)
        pe_sym = pe.clone()

        # ----- Plot: PE heatmap -----
        plt.figure(figsize=(8, 4))
        plt.imshow(pe.detach().numpy(), aspect="auto", cmap="viridis")
        plt.title(f"RoPE PE (alpha={alpha})")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_{alpha}.png")
        plt.close()

        plt.figure(figsize=(8, 4))
        plt.imshow(pe_sym.detach().numpy(), aspect="auto", cmap="viridis")
        plt.title(f"RoPE PE Sym (alpha={alpha})")
        plt.colorbar()
        plt.savefig(out_dir / f"pe_sym_{alpha}.png")
        plt.close()

        # ----- Similarity matrix -----
        sim = pe @ pe.T
        sim_sym = pe_sym @ pe_sym.T

        plt.figure(figsize=(6, 5))
        plt.imshow(sim.detach().numpy(), aspect="auto", cmap="coolwarm")
        plt.title(f"RoPE Similarity (alpha={alpha})")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_{alpha}.png")
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(sim_sym.detach().numpy(), aspect="auto", cmap="coolwarm")
        plt.title(f"RoPE Similarity Sym (alpha={alpha})")
        plt.colorbar()
        plt.savefig(out_dir / f"sim_sym_{alpha}.png")
        plt.close()
"""
def test_rope():
    """
    Tests the RoPE module by generating and plotting positional encodings
    and their similarity matrix.
    """
    print("Running RoPE test...")
    # --- Configuration ---
    xs = torch.linspace(-10, 10, 100)
    dim = 128

    # --- CORRECTED PATH DEFINITION ---
    # Get the directory containing this test file
    test_file_dir = Path(__file__).parent.parent
    # Create the output directory relative to this file's location
    out_dir = test_file_dir / "outputs" / "rope"
    # --- END OF CORRECTION ---
    
    out_dir.mkdir(exist_ok=True, parents=True)

    # --- Initialization ---
    input_name = "pos_input"
    fields = ["angle"]
    encoder = RoPE(dim=dim, input_name=input_name, fields=fields)

    # --- Encoding ---
    input_dict = {f"{input_name}_{fields[0]}": xs}
    pe = encoder(input_dict)
    print(f"Generated positional encoding of shape: {pe.shape}")

    # --- Plotting ---
    plt.figure(figsize=(8, 4))
    plt.imshow(pe.detach().numpy().T, aspect="auto", cmap="viridis")
    plt.title(f"RoPE Positional Encoding (dim={dim})")
    plt.xlabel("Position")
    plt.ylabel("Dimension")
    plt.colorbar()
    plt.savefig(out_dir / "pe_heatmap.png")
    plt.close()
    print(f"Saved PE heatmap to {out_dir / 'pe_heatmap.png'}")

    sim = pe @ pe.T
    plt.figure(figsize=(6, 5))
    plt.imshow(sim.detach().numpy(), cmap="coolwarm")
    plt.title(f"RoPE Similarity Matrix (dim={dim})")
    plt.xlabel("Position Index")
    plt.ylabel("Position Index")
    plt.colorbar()
    plt.savefig(out_dir / "sim_matrix.png")
    plt.close()
    print(f"Saved similarity matrix to {out_dir / 'sim_matrix.png'}")
    print("Test finished.")

def test_plot_and_save_similarity_matrix() -> None:
    """
    Generates positional encodings using hardcoded parameters, 
    computes their similarity, and saves a plot to a predefined directory.
    """
    # --- Parameters for the model and visualization ---
    encoding_dim = 128
    num_positions = 100
    fourier_scale = 8.0
    encoding_fields = ['x']
    input_tensor_name = 'position'

    # --- Define Output Directory and Filename ---
    # The plot will be saved in an 'outputs' directory relative to the current working directory.
    output_dir = Path("./outputs")
    output_filename = f"similarity_dim{encoding_dim}_scale{fourier_scale}.png"
    output_path = output_dir / output_filename
    
    # Print the absolute path to inform the user where the file will be saved.
    print(f"Attempting to save plot to: {output_path.resolve()}")

    # 1. Instantiate the FourierPositionEncoder
    fourier_encoder = FourierPositionEncoder(
        input_name=input_tensor_name,
        dim=encoding_dim,
        fields=encoding_fields,
        scale=fourier_scale,
    )

    # 2. Create a 1D tensor representing a sequence of positions.
    positions = torch.arange(num_positions, dtype=torch.float32)

    # 3. Format the input for the encoder, as a dictionary.
    input_data = {f"{input_tensor_name}_{field}": positions for field in encoding_fields}

    # 4. Generate the positional encodings.
    with torch.no_grad():
        positional_encodings = fourier_encoder(input_data)

    # 5. Calculate the cosine similarity between the positional encodings.
    similarity_matrix = cosine_similarity(positional_encodings.numpy())

    # 6. Plot the similarity matrix.
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap='viridis', vmin=-1, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity')

    ax.set_title(f'Positional Encoding Similarity\n(Dim={encoding_dim}, Scale={fourier_scale})')
    ax.set_xlabel('Position Index')
    ax.set_ylabel('Position Index')
    
    # Ensure the output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close(fig) # Close the figure to free up memory
    print(f"Plot saved successfully to: {output_path.resolve()}")




def visualize_fourier_similarity(
    encoding_dim: int,
    num_positions: int,
    fourier_scale: float,
    output_filename: str,
) -> None:
    """
    Generates Fourier positional encodings, computes their similarity,
    and saves the resulting heatmap to a file.

    Args:
        encoding_dim (int): The dimensionality of the Fourier feature space.
        num_positions (int): The number of sequential points to encode.
        fourier_scale (float): The scale of the random frequencies (B matrix).
                               Higher values lead to higher frequency features.
        output_filename (str): The path to save the output PNG file.
    """
    print(f"Generating similarity plot for dim={encoding_dim}, scale={fourier_scale}...")
    
    # --- Parameters for the model ---
    encoding_fields = ['x']  # Using a single 1D coordinate
    input_tensor_name = 'position'

    # 1. Instantiate the FourierPositionEncoder
    fourier_encoder = FourierPositionEncoder(
        input_name=input_tensor_name,
        dim=encoding_dim,
        fields=encoding_fields,
        scale=fourier_scale,
    )

    # 2. Create a 1D tensor representing a sequence of positions (e.g., 0, 1, 2,...)
    positions = torch.arange(num_positions, dtype=torch.float32)

    # 3. Format the input for the encoder as a dictionary
    input_data = {f"{input_tensor_name}_{field}": positions for field in encoding_fields}

    # 4. Generate the positional encodings
    with torch.no_grad():
        positional_encodings = fourier_encoder(input_data)

    # 5. Calculate the cosine similarity between the positional encodings
    #    The result is a matrix of shape (num_positions, num_positions)
    similarity_matrix = cosine_similarity(positional_encodings.numpy())

    # 6. Plot the similarity matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(similarity_matrix, cmap='cividis', vmin=-1, vmax=1)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine Similarity')

    ax.set_title(
        f'Fourier Feature Similarity (Nearby points should be similar)\n'
        f'(Dim={encoding_dim}, Scale={fourier_scale})'
    )
    ax.set_xlabel('Position Index')
    ax.set_ylabel('Position Index')

    # Ensure the output directory exists
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)  # Close the figure to free up memory
    print(f"Plot saved successfully to: {output_path.resolve()}")


def test_fourier_similarity_matrix() -> None:
    # --- Configuration ---
    # Feel free to change these parameters to see how the plot changes
    DIM = 128          # Dimension of the encoding
    NUM_POSITIONS = 200 # Length of the sequence
    SCALE = 0.005        # Controls the "frequency" of the features

    # --- Run the visualization ---
    visualize_fourier_similarity(
        encoding_dim=DIM,
        num_positions=NUM_POSITIONS,
        fourier_scale=SCALE,
        output_filename=f"outputs/fourier_similarity_dim{DIM}_scale{SCALE}.png"
    )

    # --- Example with a higher frequency scale ---
    SCALE_HIGH_FREQ = 32.0
    visualize_fourier_similarity(
        encoding_dim=DIM,
        num_positions=NUM_POSITIONS,
        fourier_scale=SCALE_HIGH_FREQ,
        output_filename=f"outputs/fourier_similarity_dim{DIM}_scale{SCALE_HIGH_FREQ}.png"
    )