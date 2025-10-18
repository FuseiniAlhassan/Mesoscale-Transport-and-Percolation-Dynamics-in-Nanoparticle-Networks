# Basic imports and output directory
import os, math, time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from IPython.display import HTML, Image, display

OUTDIR = '/mnt/data/mesoscale_outputs'
os.makedirs(OUTDIR, exist_ok=True)
print('Outputs saved to', OUTDIR)
def anisotropic_distance(p1, p2, box, anisotropy=(1.0,1.0)):
    # minimum-image distance with anisotropy scaling of coordinates
    delta = p2 - p1
    delta = delta - np.round(delta/box)*box
    # apply anisotropy scaling (different effective scales in x,y)
    scaled = delta * np.array(anisotropy)
    return np.sqrt((scaled**2).sum()), scaled

def compute_edge_conductance(dist, contact_radius, g_contact, g_tunnel_0, decay_len):
    if dist <= 2*contact_radius:
        return g_contact
    else:
        gap = dist - 2*contact_radius
        return g_tunnel_0 * np.exp(-gap/decay_len)

def generate_particles(n, box, radius_mean=0.01, radius_std=0.0, seed=None):
    rng = np.random.default_rng(seed)
    pos = rng.random((n,2)) * box
    if radius_std>0:
        radii = np.abs(rng.normal(radius_mean, radius_std, size=n))
    else:
        radii = np.ones(n)*radius_mean
    return pos, radii

def build_network(positions, radii, box, cutoff, params, anisotropy=(1.0,1.0)):
    n = len(positions)
    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=positions[i], radius=radii[i])
    for i in range(n):
        for j in range(i+1, n):
            dist, scaled = anisotropic_distance(positions[i], positions[j], box, anisotropy=anisotropy)
            if dist <= cutoff:
                g = compute_edge_conductance(dist, params['contact_radius'], params['g_contact'], params['g_tunnel_0'], params['decay_len'])
                if g>0:
                    G.add_edge(i,j, conductance=g, distance=dist, vec=scaled)
    return G
def solve_effective_conductivity(G, box, direction=0, clamp_frac=0.05):
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {node:k for k,node in enumerate(nodes)}
    row=[]; col=[]; data=[]
    V_fixed={}
    tol = clamp_frac*box
    for node in nodes:
        x,y = G.nodes[node]['pos']
        coord = x if direction==0 else y
        if coord < tol:
            V_fixed[node]=1.0
        elif coord > (box - tol):
            V_fixed[node]=0.0
    for u,v,d in G.edges(data=True):
        g = d['conductance']
        iu, iv = idx[u], idx[v]
        row += [iu, iv]; col += [iv, iu]; data += [-g, -g]
        row += [iu, iu, iv, iv]; col += [iu, iu, iv, iv]; data += [g, 0.0, g, 0.0]
    L = sp.coo_matrix((data,(row,col)), shape=(n,n)).tocsr()
    fixed_idx = sorted([idx[node] for node in V_fixed.keys()])
    free_idx = [i for i in range(n) if i not in fixed_idx]
    if len(free_idx)==0:
        return 0.0, None
    Lff = L[free_idx][:, free_idx]
    Vk = np.array([V_fixed[nodes[i]] for i in fixed_idx]) if fixed_idx else np.array([])
    if fixed_idx:
        Lfk = L[free_idx][:, fixed_idx]
        b = -Lfk.dot(Vk)
    else:
        b = np.zeros(len(free_idx))
    potentials = spla.spsolve(Lff, b)
    V = np.zeros(n)
    for i,fi in enumerate(free_idx):
        V[fi] = potentials[i]
    for k,fi in enumerate(fixed_idx):
        V[fi] = Vk[k]
    total_current = 0.0
    for u,v,d in G.edges(data=True):
        xi = G.nodes[u]['pos'][direction]
        xj = G.nodes[v]['pos'][direction]
        if (xi < 0.5*box and xj >= 0.5*box) or (xj < 0.5*box and xi >= 0.5*box):
            iu, iv = idx[u], idx[v]
            total_current += d['conductance'] * (V[iu] - V[iv])
    sigma_eff = total_current / (1.0 * box)
    return sigma_eff, V

# Simple finite-difference PDE: steady-state diffusion/Poisson solver on a grid
def solve_fd_pde_sources(positions, box, grid_res=64, source_strength=1.0, diff_coeff=1.0):
    nxg = nyg = grid_res
    hx = box/nxg
    N = nxg*nyg
    row=[]; col=[]; data=[]; b = np.zeros(N)
    def idx(i,j): return i + j*nxg
    # build 5-point Laplacian with Dirichlet at boundaries = 0
    for j in range(nyg):
        for i in range(nxg):
            k = idx(i,j)
            if i==0 or j==0 or i==nxg-1 or j==nyg-1:
                row.append(k); col.append(k); data.append(1.0)
                b[k]=0.0
            else:
                row += [k,k,k,k,k]; col += [k, idx(i-1,j), idx(i+1,j), idx(i,j-1), idx(i,j+1)]
                data += [-4.0,1.0,1.0,1.0,1.0]
    # add sources at nearest cell to particle positions
    for p in positions:
        i = min(max(int(p[0]/box*nxg),0), nxg-1)
        j = min(max(int(p[1]/box*nyg),0), nyg-1)
        b[idx(i,j)] += source_strength
    A = sp.coo_matrix((data,(row,col)), shape=(N,N)).tocsr()
    phi = spla.spsolve(A, b)
    Phi = phi.reshape((nyg,nxg))
    return Phi
def map_local_conductance(G, positions, box, grid_res=128):
    grid = np.zeros((grid_res, grid_res))
    xedges = np.linspace(0, box, grid_res+1)
    yedges = np.linspace(0, box, grid_res+1)
    for u,v,d in G.edges(data=True):
        pos_u = G.nodes[u]['pos']
        pos_v = G.nodes[v]['pos']
        mid = 0.5*(pos_u + pos_v)
        ix = np.searchsorted(xedges, mid[0]) - 1
        iy = np.searchsorted(yedges, mid[1]) - 1
        ix = np.clip(ix, 0, grid_res-1); iy = np.clip(iy, 0, grid_res-1)
        grid[iy, ix] += d['conductance']
    return grid

def fourier_lowpass(grid, lowpass_frac=0.08):
    f = np.fft.fftshift(np.fft.fft2(grid))
    ny,nx = grid.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    Kmax = 0.5
    cutoff = lowpass_frac * Kmax
    mask = (K <= cutoff).astype(float)
    f_filtered = f * mask
    grid_filt = np.real(np.fft.ifft2(np.fft.ifftshift(f_filtered)))
    return grid_filt
# Example run: build network, compute conductivity, solve PDE, do Fourier analysis, save figures + animation

def run_example(n_particles=800, box=1.0, seed=1, cutoff=0.12, temp=300.0, anisotropy=(1.0,0.8)):
    # parameters
    params = {'contact_radius':0.01, 'g_contact':1e3, 'g_tunnel_0':1.0, 'decay_len':0.02}
    pos, radii = generate_particles(n_particles, box, radius_mean=0.01, radius_std=0.002, seed=seed)
    # Temperature affects thermal motion amplitude -> affects simulated drift per frame
    thermal_scale = math.sqrt(temp/300.0) * 0.002  # arbitrary scaling
    G = build_network(pos, radii, box, cutoff, params, anisotropy=anisotropy)
    sigma, V = solve_effective_conductivity(G, box)
    print('Effective conductivity (approx):', sigma)
    grid = map_local_conductance(G, pos, box, grid_res=128)
    grid_filt = fourier_lowpass(grid, lowpass_frac=0.06)
    # Save maps
    plt.figure(figsize=(5,4)); plt.imshow(grid, origin='lower'); plt.title('Local conductance map'); plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'conductance_map.png'), dpi=200); plt.close()
    plt.figure(figsize=(5,4)); plt.imshow(grid_filt, origin='lower'); plt.title('Low-pass filtered'); plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'conductance_lowpass.png'), dpi=200); plt.close()
    # PDE solve
    Phi = solve_fd_pde_sources(pos, box, grid_res=64, source_strength=1.0, diff_coeff=1.0)
    plt.figure(figsize=(5,4)); plt.imshow(Phi, origin='lower'); plt.title('FD PDE field (steady-state)'); plt.colorbar(); plt.tight_layout(); plt.savefig(os.path.join(OUTDIR,'pde_field.png'), dpi=200); plt.close()
    # Animation: show thermal drift with anisotropic scaling applied during distance calc
    fig, ax = plt.subplots(figsize=(5,5))
    sc = ax.scatter(pos[:,0], pos[:,1], s=8, c='royalblue')
    ax.set_xlim(0, box); ax.set_ylim(0, box); ax.set_title('Thermal drift (T=%g K)'%temp)
    def update(frame):
        # brownian-like random step (periodic)
        steps = np.random.normal(0, thermal_scale, size=pos.shape)
        pos[:] = (pos + steps) % box
        sc.set_offsets(pos)
        return sc,
    anim = FuncAnimation(fig, update, frames=120, interval=80, blit=True)
    gifpath = os.path.join(OUTDIR,'thermal_drift.gif')
    anim.save(gifpath, writer=PillowWriter(fps=15))
    plt.close(fig)
    print('Saved figures and animation to', OUTDIR)
    return {'sigma':sigma, 'maps':{'grid':grid, 'grid_filt':grid_filt}, 'pde':Phi, 'anim':gifpath}

res = run_example(n_particles=600, box=1.0, seed=42, cutoff=0.11, temp=500.0, anisotropy=(1.0,0.6))
res
# Create dataset: generate many random networks and extract features (global sums, low-pass energies, PDE-derived metrics)
def extract_features_from_map(grid):
    feat = []
    feat.append(grid.sum())  # total conductance mass
    feat.append(np.mean(grid))
    feat.append(np.std(grid))
    # spectral energy in low-frequency band
    f = np.fft.fftshift(np.fft.fft2(grid))
    spec = np.abs(f)
    ny,nx = grid.shape
    ky = np.fft.fftshift(np.fft.fftfreq(ny))
    kx = np.fft.fftshift(np.fft.fftfreq(nx))
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    Kmax = 0.5
    low = (K <= 0.06).astype(float)
    high = (K > 0.2).astype(float)
    feat.append((spec*low).sum())
    feat.append((spec*high).sum())
    return np.array(feat)

def generate_dataset(N=80):
    X=[]; y=[]
    for i in range(N):
        seed = i*7 + 3
        pos, radii = generate_particles(500, 1.0, radius_mean=0.01, radius_std=0.002, seed=seed)
        G = build_network(pos, radii, 1.0, cutoff=0.11, params={'contact_radius':0.01,'g_contact':1e3,'g_tunnel_0':1.0,'decay_len':0.02}, anisotropy=(1.0, 0.8 if (i%2==0) else 1.0))
        sigma, _ = solve_effective_conductivity(G, 1.0)
        grid = map_local_conductance(G, pos, 1.0, grid_res=64)
        feats = extract_features_from_map(grid)
        X.append(feats); y.append(sigma)
    X = np.vstack(X); y = np.array(y)
    return X, y

X, y = generate_dataset(N=60)
print('Dataset shapes', X.shape, y.shape)

# Simple NumPy MLP (one hidden layer) for regression
def mlp_train(X, y, hidden=32, lr=1e-3, epochs=500):
    # Normalize features
    mu = X.mean(axis=0); sig = X.std(axis=0) + 1e-9
    Xn = (X - mu)/sig
    nin = Xn.shape[1]
    rng = np.random.default_rng(1)
    W1 = rng.normal(0, 0.1, size=(nin, hidden))
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.1, size=(hidden,1))
    b2 = np.zeros(1)
    for ep in range(epochs):
        # forward
        h = np.tanh(Xn.dot(W1) + b1)
        ypred = h.dot(W2).ravel() + b2
        err = ypred - y
        loss = np.mean(err**2)
        # gradients (MSE)
        grad_y = 2*err / len(y)
        grad_W2 = h.T.dot(grad_y.reshape(-1,1))
        grad_b2 = grad_y.sum()
        dh = (grad_y.reshape(-1,1)).dot(W2.T) * (1 - h**2)
        grad_W1 = Xn.T.dot(dh)
        grad_b1 = dh.sum(axis=0)
        # update
        W2 -= lr * grad_W2
        b2 -= lr * grad_b2
        W1 -= lr * grad_W1
        b1 -= lr * grad_b1
        if ep % 100 == 0:
            print(f'epoch {ep} loss {loss:.6e}')
    model = {'W1':W1,'b1':b1,'W2':W2,'b2':b2,'mu':mu,'sig':sig}
    return model

model = mlp_train(X, y, hidden=64, lr=5e-4, epochs=600)
# Test on a fresh sample
pos, radii = generate_particles(600,1.0, radius_mean=0.01, seed=999)
G = build_network(pos, radii, 1.0, cutoff=0.11, params={'contact_radius':0.01,'g_contact':1e3,'g_tunnel_0':1.0,'decay_len':0.02}, anisotropy=(1.0,0.7))
sigma_true, _ = solve_effective_conductivity(G, 1.0)
grid = map_local_conductance(G, pos, 1.0, grid_res=64)
feat = extract_features_from_map(grid)
Xn = (feat - model['mu'])/model['sig']
h = np.tanh(Xn.dot(model['W1']) + model['b1'])
sigma_pred = h.dot(model['W2']).ravel() + model['b2']
print('True sigma:', sigma_true, 'Predicted sigma:', float(sigma_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import animation
import os

# Particle density heatmap
def plot_density_heatmap(positions, box_size, outdir):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.kdeplot(x=[p[0] for p in positions], y=[p[1] for p in positions],
                fill=True, cmap='viridis', ax=ax)
    ax.set_xlim(0, box_size)
    ax.set_ylim(0, box_size)
    ax.set_title("Particle Density Heatmap")
    plt.savefig(os.path.join(outdir, "particle_density_heatmap.png"), dpi=200)
    plt.show()

# Anisotropic transport field visualization
def plot_anisotropic_field(positions, anisotropy_factor, outdir):
    fig, ax = plt.subplots(figsize=(6,6))
    quiver_x = np.array([p[0] for p in positions])
    quiver_y = np.array([p[1] for p in positions])
    # Example vectors based on anisotropy factor, needs refinement based on actual model
    U = np.cos(quiver_x / 10) * anisotropy_factor[0]
    V = np.sin(quiver_y / 10) * anisotropy_factor[1]
    ax.quiver(quiver_x, quiver_y, U, V, color="orange", alpha=0.7)
    ax.set_title("Anisotropic Transport Field")
    plt.savefig(os.path.join(outdir, "anisotropic_transport_field.png"), dpi=200)
    plt.show()


# Inverse model training loss curve
if 'model' in locals() and 'losses' in model: # Assuming losses might be stored in the model dict
    plt.figure(figsize=(5,3))
    plt.plot(model['losses'], color='darkred')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Inverse Model Training Loss")
    plt.grid(True)
    plt.savefig(os.path.join(OUTDIR, "inverse_model_loss_curve.png"), dpi=200)
    plt.show()

# Combine frames into animation for temporal visualization
def create_transport_animation(positions, box_size, diffusion_strength, outdir):
    fig, ax = plt.subplots(figsize=(6,6))
    def animate(i):
        ax.clear()
        # Simple random walk for visualization, not the true simulation
        xs = [p[0] + np.random.normal(0, diffusion_strength*0.1) for p in positions]
        ys = [p[1] + np.random.normal(0, diffusion_strength*0.1) for p in positions]
        ax.scatter(xs, ys, s=5, color='steelblue')
        ax.set_xlim(0, box_size)
        ax.set_ylim(0, box_size)
        ax.set_title(f"Frame {i}")
    ani = animation.FuncAnimation(fig, animate, frames=30, interval=150)
    gifpath = os.path.join(outdir, "rich_transport_animation.gif")
    ani.save(gifpath, writer="pillow")
    plt.close(fig)
    return gifpath


if 'res' in locals():
    
    
    n_particles=600; box=1.0; seed=42; cutoff=0.11; temp=500.0; anisotropy=(1.0,0.6)
    pos, _ = generate_particles(n_particles, box, radius_mean=0.01, seed=seed)
    plot_density_heatmap(pos, box, OUTDIR)
    plot_anisotropic_field(pos, anisotropy, OUTDIR)
    # The animation display
    print("Displaying animation from run_example:")
    display(Image(filename=res['anim']))