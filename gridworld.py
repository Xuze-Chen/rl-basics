# %%
import numpy as np

N, N_STATES = 4, 16
ACTIONS = [(-1,0), (1,0), (0,-1), (0,1)]
TERMINALS = {0, 15}

def step(s, a):
    if s in TERMINALS:
        return s, 0.0
    r, c = divmod(s, N)
    dr, dc = ACTIONS[a]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < N and 0 <= nc < N):
        nr, nc = r, c         
    return nr * N + nc, -1.0

# %%
def policy_evaluation(pi, gamma=1.0, theta=1e-4, in_place=False):
    V = np.zeros(N_STATES)
    sweeps = 0
    while True:
        sweeps += 1
        V_old = V.copy()               
        delta = 0.0
        for s in range(N_STATES):
            if s in TERMINALS:
                continue               
            src = V if in_place else V_old   
            new_v = sum(pi[s,a] * (r + gamma * src[s2])
                        for a, (s2, r) in enumerate(step(s,a) for a in range(4)))
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:              
            return V, sweeps
        
# %%
def greedy_policy(V, gamma=1.0):
    pi = np.zeros((N_STATES, 4))
    for s in range(N_STATES):
        if s in TERMINALS: continue
        q = np.array([step(s,a)[1] + gamma*V[step(s,a)[0]] for a in range(4)])
        best = np.flatnonzero(q >= q.max() - 1e-9)   # 并列全留下
        pi[s, best] = 1.0 / len(best)
    return pi

# %%        
def value_iteration(gamma=1.0, theta=1e-4):
    V = np.zeros(N_STATES)
    sweeps = 0
    while True:
        sweeps += 1
        V_old = V.copy()
        delta = 0.0
        for s in range(N_STATES):
            if s in TERMINALS:
                continue
            q = []
            for a in range(4):
                s2, r = step(s, a)
                q.append(r + gamma * V_old[s2])
            new_v = max(q)             
            delta = max(delta, abs(new_v - V[s]))
            V[s] = new_v
        if delta < theta:
            return V, sweeps
# %%
def policy_iteration(gamma=1.0, theta=1e-4):
    pi = np.ones((N_STATES, 4)) / 4      
    total_sweeps = 0
    for outer in range(1, 100):
        V, sw = policy_evaluation(pi, gamma, theta)   
        total_sweeps += sw
        pi_new = greedy_policy(V, gamma)              
        if np.allclose(pi_new, pi):                  
            return V, pi, outer, total_sweeps
        pi = pi_new

# %%
ARROWS = ['^', 'v', '<', '>']   

def show_policy(pi):
    out = []
    for s in range(N_STATES):
        if s in TERMINALS:
            out.append(' T  ')
        else:
            out.append(''.join(ARROWS[a] for a in range(4) if pi[s, a] > 0).ljust(4))
    return np.array(out).reshape(N, N)

# %%
# %%
rand_pi = np.ones((N_STATES, 4)) / 4

V_rand, _ = policy_evaluation(rand_pi)
print("Random policy V")
print(np.round(V_rand.reshape(4, 4), 1), "\n")

_, _, _, total = policy_iteration()
V_vi, sw_vi = value_iteration()

print("Optimal V")
print(np.round(V_vi.reshape(4, 4), 1), "\n")

print(f"policy iteration : {total} sweeps")
print(f"value iteration  : {sw_vi} sweeps")