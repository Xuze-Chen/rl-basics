# %%
import os
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

def argmax_random(q, rng):
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))

def eps_constant(c=0.1):
    return lambda ep, n: c
 
def eps_linear(start=1.0, end=0.01, frac=0.5):
    def f(ep, n):
        return max(end, start - (start - end) * ep / (n * frac))
    return f
 
def eps_exponential(start=1.0, end=0.01, frac=0.3):
    def f(ep, n):
        return max(end, start * np.exp(-ep / (n * frac)))
    return f
# %%
def train(is_slippery=False, map_name="4x4", episodes=5000,      
          alpha=0.1, gamma=0.99, eps_fn=None, seed=0):           
    if eps_fn is None:                                          
        eps_fn = eps_linear()

    env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)
    rng = np.random.default_rng(seed)

    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_s, n_a))
    returns = []

    for ep in range(episodes):
        eps = eps_fn(ep, episodes)   

        s, _ = env.reset(seed=int(rng.integers(1e9)))
        done, total_r = False, 0.0
        while not done:
            if rng.random() < eps:
                a = int(rng.integers(n_a))
            else:
                a = argmax_random(Q[s], rng)
            s2, r, terminated, truncated, _ = env.step(a)
            target = r if terminated else r + gamma * Q[s2].max()
            Q[s, a] += alpha * (target - Q[s, a])
            s = s2
            total_r += r
            done = terminated or truncated
        returns.append(total_r)

    env.close()
    return Q, np.array(returns)
# %%
def rolling(x, w=100):
    return np.convolve(x, np.ones(w) / w, mode="valid")
 
# %%
schedules = [("constant 0.1", eps_constant(0.1)),
             ("linear 1->0.01", eps_linear()),
             ("exponential", eps_exponential())]
 
fig, ax = plt.subplots(figsize=(7, 4))
for name, fn in schedules:
    Q, ret = train(is_slippery=True, episodes=5000, eps_fn=fn)
    ax.plot(rolling(ret), label=name)
    print(f"  {name:16s} final 1000 ep: {ret[-1000:].mean():.3f}")
 
ax.set_xlabel("episode")
ax.set_ylabel("success rate (last 100)")
ax.set_title("FrozenLake 4x4 slippery: epsilon schedule")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("figs/eps_schedule.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(7, 4))
for a in [0.1, 0.5]:
    Q, ret = train(is_slippery=True, episodes=5000, alpha=a)
    ax.plot(rolling(ret), label=f"alpha={a}")
    print(f"  alpha={a}  final 1000 ep: {ret[-1000:].mean():.3f}")
 
ax.set_xlabel("episode")
ax.set_ylabel("success rate (last 100)")
ax.set_title("FrozenLake 4x4 slippery: learning rate")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("figs/alpha.png", dpi=150, bbox_inches="tight")
plt.show()
 
# %%
fig, ax = plt.subplots(figsize=(7, 4))
for mn in ["4x4", "8x8"]:
    Q, ret = train(map_name=mn, is_slippery=True, episodes=5000)
    ax.plot(rolling(ret), label=f"{mn}  (Q shape {Q.shape})")
    print(f"  {mn}: Q{Q.shape}  final 1000 ep: {ret[-1000:].mean():.3f}")
 
ax.set_xlabel("episode")
ax.set_ylabel("success rate (last 100)")
ax.set_title("map size")
ax.legend()
ax.grid(alpha=.3)
fig.tight_layout()
fig.savefig("figs/map_size.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
ARROWS = ["<-", "v", "->", "^"]  

def get_desc(map_name="4x4"):
    env = gym.make("FrozenLake-v1", map_name=map_name)
    desc = np.array([[c.decode() for c in row] for row in env.unwrapped.desc])
    env.close()
    return desc

def plot_policy_and_value(Q, map_name="4x4", title="", fname="figs/policy.png"):
    desc = get_desc(map_name)
    n = desc.shape[0]
    V = Q.max(axis=1).reshape(n, n)
    fs = 18 if n == 4 else 10
 
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    ax.imshow(np.zeros((n, n)), cmap="Greys", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            cell = desc[i, j]
            if cell == "H":
                ax.text(j, i, "H", ha="center", va="center", fontsize=fs, color="crimson")
            elif cell == "G":
                ax.text(j, i, "G", ha="center", va="center", fontsize=fs, color="green")
            else:
                ax.text(j, i, ARROWS[int(np.argmax(Q[i * n + j]))], ha="center", va="center", fontsize=fs)
    ax.set_title(f"policy (argmax Q) {title}")
    ax.set_xticks([]); ax.set_yticks([])
 
    ax = axes[1]
    im = ax.imshow(V, cmap="viridis")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{V[i, j]:.2f}", ha="center", va="center",
                    color="w", fontsize=8 if n == 4 else 5)
    ax.set_title("V = max Q")
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046)
 
    fig.tight_layout()
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    return V

Q_best, ret_best = train(is_slippery=True, episodes=5000, eps_fn=eps_linear())
print(f"\nfinal policy success rate: {ret_best[-1000:].mean():.3f}")
V = plot_policy_and_value(Q_best, "4x4", "(slippery)", "figs/policy_4x4.png")
print(np.round(V, 2))