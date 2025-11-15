import sys
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

# Argument parsing with Jupyter safety

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_particles", type=int, default=500, help="number of nanoparticles")
    parser.add_argument("--box", type=int, default=100, help="simulation box size")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--outdir", type=str, default="outputs", help="directory to save figures")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args

# Mesoscale network generator

def generate_network(n_particles, box, seed):
    np.random.seed(seed)
    positions = np.random.rand(n_particles, 2) * box
    G = nx.Graph()
    for i, pos in enumerate(positions):
        G.add_node(i, pos=pos)
    cutoff = box * 0.1  # connectivity radius
    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < cutoff:
                weight = np.exp(-dist / cutoff)
                G.add_edge(i, j, weight=weight)
    return G, positions

# Fourier-based heterogeneity analysis

def fourier_filter(positions, box, resolution=128):
    grid = np.zeros((resolution, resolution))
    scaled = (positions / box * resolution).astype(int)
    for x, y in scaled:
        grid[y % resolution, x % resolution] += 1
    fft_image = np.fft.fftshift(np.fft.fft2(grid))
    spectrum = np.abs(fft_image)
    return grid, spectrum

# Visualization & Animation

def visualize(G, positions, spectrum, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Static percolation network
    plt.figure(figsize=(6, 6))
    nx.draw(G, pos={i: p for i, p in enumerate(positions)}, node_size=10, edge_color='gray')
    plt.title("Nanoparticle Percolation Network")
    plt.savefig(os.path.join(outdir, "percolation_network.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Fourier spectrum
    plt.figure(figsize=(6, 5))
    plt.imshow(np.log1p(spectrum), cmap='inferno')
    plt.title("Fourier Spectrum of Spatial Heterogeneity")
    plt.colorbar(label='log amplitude')
    plt.savefig(os.path.join(outdir, "fourier_spectrum.png"), dpi=300, bbox_inches='tight')
    plt.show()

    # Animation of particle evolution (simple random drift)
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(positions[:, 0], positions[:, 1], s=10, c='royalblue')
    ax.set_xlim(0, np.max(positions[:, 0]) * 1.1)
    ax.set_ylim(0, np.max(positions[:, 1]) * 1.1)
    ax.set_title("Nanoparticle Drift Simulation")

    def update(frame):
        positions[:, 0] += np.random.normal(0, 0.2, len(positions))
        positions[:, 1] += np.random.normal(0, 0.2, len(positions))
        scat.set_offsets(positions)
        return scat,

    anim = FuncAnimation(fig, update, frames=100, interval=80, blit=True)
    anim_path = os.path.join(outdir, "nanoparticle_drift.gif")
    anim.save(anim_path, writer='pillow', fps=15)
    plt.close(fig)

# Main logic

def main(n_particles=None, box=None, seed=None, outdir=None):
    if n_particles is None and box is None and seed is None:
        args = parse_args()
        n_particles, box, seed, outdir = args.n_particles, args.box, args.seed, args.outdir

    G, positions = generate_network(n_particles, box, seed)
    grid, spectrum = fourier_filter(positions, box)
    visualize(G, positions, spectrum, outdir)

if __name__ == "__main__":
    main()
