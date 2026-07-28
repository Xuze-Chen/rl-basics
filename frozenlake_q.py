# %%
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
 
def argmax_random(q, rng):
    best = np.flatnonzero(q == q.max())
    return int(rng.choice(best))

# %%
def train(is_slippery=False, episodes=2000, alpha=0.1, gamma=0.99,
          eps_start=1.0, eps_end=0.01, eps_decay_frac=0.5, seed=0):
    env = gym.make("FrozenLake-v1", is_slippery=is_slippery)
    rng = np.random.default_rng(seed)
 
    n_s, n_a = env.observation_space.n, env.action_space.n
    Q = np.zeros((n_s, n_a))
    returns = []
    decay_eps = int(episodes * eps_decay_frac)
 
    for ep in range(episodes):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / decay_eps)
        s, _ = env.reset(seed=int(rng.integers(1e9)))
        done, total_r = False, 0.0
 
        while not done:
            # --- ε-greedy 选动作 ---
            if rng.random() < eps:
                a = int(rng.integers(n_a))
            else:
                a = argmax_random(Q[s], rng)
 
            s2, r, terminated, truncated, _ = env.step(a)
 
            # --- Q-learning 更新 ---
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
 
ARROWS = ["<", "v", ">", "^"]  
 
 
def show_policy(Q, holes=(5, 7, 11, 12), goal=15):
    out = []
    for s in range(16):
        if s in holes:
            out.append("H")
        elif s == goal:
            out.append("G")
        else:
            out.append(ARROWS[int(np.argmax(Q[s]))])
    return np.array(out).reshape(4, 4)
# %%
if __name__ == "__main__":
    Q_det, ret_det = train(is_slippery=False, episodes=2000)
    Q_slip, ret_slip = train(is_slippery=True, episodes=2000)
 
    print("non-slippery  last 100:", ret_det[-100:].mean())
    print(show_policy(Q_det), "\n")
    print("slippery      last 100:", ret_slip[-100:].mean())
    print(show_policy(Q_slip))
 
    plt.figure(figsize=(7, 4))
    plt.plot(rolling(ret_det), label="is_slippery=False")
    plt.plot(rolling(ret_slip), label="is_slippery=True")
    plt.xlabel("episode")
    plt.ylabel("success rate (last 100)")
    plt.legend()
    plt.grid(alpha=.3)
    plt.tight_layout()