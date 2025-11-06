"""
AADS.ipynb
"""

pip install pulp

import torch
import torch.nn as nn
import numpy as np
import math
from collections import defaultdict
import copy

# === Constants ===
THETA = 3e8   # propagation speed (m/s)
BETA = 1.0    # weight parameter

# === Classes ===
class User:
    def __init__(self, i, app_i, inst_i, p_i, b_i, positions, tasks):
        self.i = i
        self.app_i = app_i  # application data size (GB)
        self.inst_i = inst_i  # instance storage (MB)
        self.p_i = p_i  # computing requirement per task
        self.b_i = b_i  # uplink data rate (bps)
        self.positions = positions  # list of (x_t, y_t)
        self.tasks = tasks  # list of (r_t, eta_t, omega_t): num tasks, data size, CPU cycles

class EdgeNode:
    def __init__(self, j, x_j, y_j, p_j, b_j):
        self.j = j
        self.x = x_j
        self.y = y_j
        self.p_j = p_j  # total computing resources
        self.b_j = b_j  # transmission rate (bps)

# === Utility functions ===
def distance(pos1, pos2):
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def offloading_time(dist, eta, b_i):
    return dist / THETA + eta / b_i

def routing_time(dis, eta, b_j):
    return dis / THETA + eta / b_j

def computation_time(omega, p_i):
    return omega / p_i

# === Residual LSTM (RELS model) ===
class ResidualLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=4, dropout=0.2, output_size=3):
        super().__init__()
        self.base_lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.base_norm = nn.LayerNorm(hidden_size)
        self.res_lstms = nn.ModuleList([
            nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
            for _ in range(3)
        ])
        self.res_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(3)])
        self.fc = nn.Linear(hidden_size, output_size)  # map hidden → [x, y, r]

    def forward(self, x):
        out, _ = self.base_lstm(x)
        out = self.base_norm(out)
        for lstm, norm in zip(self.res_lstms, self.res_norms):
            res = out
            temp, _ = lstm(out)
            temp = norm(temp)
            out = res + temp
        out = self.fc(out)  # map to 3-dim output
        return out

# === Prediction function ===
def predict_rels(historical, W_th, model, edges, device='cpu'):
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(historical[-10:], dtype=torch.float32).unsqueeze(0).to(device)
        preds = []
        for _ in range(W_th):
            out = model(input_seq)[:, -1, :]  # predicted next [x,y,r]
            preds.append(out.squeeze().cpu().numpy())
            # append new prediction to input_seq
            input_seq = torch.cat((input_seq[:, 1:, :], out.unsqueeze(1)), dim=1)

    # Denormalize
    positions = [(p[0]*600, p[1]*600) for p in preds]
    connecting_nodes = [np.argmin([distance(p, (e.x, e.y)) for e in edges]) for p in positions]
    r_pre = [max(1, min(5, int(abs(p[2]*5) + 1))) for p in preds]
    return positions, connecting_nodes, r_pre

# === Adaptive Application Deployment Scheme (AADS) ===
def aads(users, edges, acc_th, beta, historical):
    device = torch.device('cpu')
    model = ResidualLSTM()  # RELS model
    mu = defaultdict(int)
    W, Y = {}, {}

    # Accuracy table (simulation)
    acc_table = {u.i: [0.95 - 0.01*(t-1) for t in range(1, 11)] for u in users}
    W_th = {u.i: max(t for t in range(1, 11) if acc_table[u.i][t-1] >= acc_th) for u in users}
    Y_th = {u.i: 5 for u in users}  # dummy fixed

    V = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    p_dict = {u.i: u.p_i for u in users}
    pj_dict = {e.j: e.p_j for e in edges}

    for user in users:
        bar_U, bar_B, bar_r = predict_rels(historical[user.i], W_th[user.i], model, edges, device)
        for j, e in enumerate(edges):
            for t in range(1, W_th[user.i] + 1):
                max_r = max(bar_r[:t]) if bar_r[:t] else 1
                for r in range(1, min(Y_th[user.i], max_r) + 1):
                    off = rout = comp = 0.0
                    for tt in range(t):
                        dist_i_j = distance(bar_U[tt], (e.x, e.y))
                        eta = user.tasks[tt % len(user.tasks)][1]
                        omega = user.tasks[tt % len(user.tasks)][2]
                        off += bar_r[tt] * offloading_time(dist_i_j, eta, user.b_i)
                        comp += math.ceil(bar_r[tt] / r) * computation_time(omega, user.p_i)
                        if j != bar_B[tt]:
                            dis_j_k = distance((e.x, e.y), (edges[bar_B[tt]].x, edges[bar_B[tt]].y))
                            rout += routing_time(dis_j_k, eta, e.b_j)
                    cost = (user.app_i + r * user.inst_i) / t
                    V[user.i][j][t][r] = (off + comp + rout) / t + beta * cost

    deployment = abbd(V, p_dict, pj_dict, W_th, Y_th)
    for i, j, t, r in deployment:
        mu[(i, j)] = 1
        W[i] = t
        Y[i] = r

    usage = defaultdict(float)
    for i in W:
        j = next(k[1] for k in mu if k[0] == i and mu[k] == 1)
        usage[j] += p_dict[i] * Y[i]
    for j in usage:
        if usage[j] > pj_dict[j]:
            print(f"⚠️ Resource constraint violated for edge {j}")
    return mu, W, Y

# === ABBD algorithm ===
def abbd(V, p, pj, W_th, Y_th):
    candidates = [(V[i][j][t][r], i, j, t, r, p[i]*r) for i in V for j in V[i] for t in V[i][j] for r in V[i][j][t]]
    init_assignment = lgd(V, p, pj, W_th, Y_th)
    best_cost = sum(c[0] for c in init_assignment) if init_assignment else float('inf')
    best_assign = init_assignment

    def branch(assignment, remaining_pj, current_cost, unassigned):
        nonlocal best_cost, best_assign
        if current_cost >= best_cost:
            return
        if not unassigned:
            best_cost = current_cost
            best_assign = assignment[:]
            return

        lb = current_cost
        for i in unassigned:
            min_v = min(V[i][j][t][r] for j in V[i] for t in V[i][j] for r in V[i][j][t])
            lb += min_v
        if lb >= best_cost:
            return

        next_i = list(unassigned)[0]
        new_un = unassigned - {next_i}
        for cand in [c for c in candidates if c[1] == next_i]:
            v, i, j, t, r, demand = cand
            if remaining_pj[j] >= demand:
                new_assign = assignment + [(i, j, t, r)]
                new_rem = copy.deepcopy(remaining_pj)
                new_rem[j] -= demand
                branch(new_assign, new_rem, current_cost + v, new_un)

    branch([], copy.deepcopy(pj), 0.0, set(V.keys()))
    return best_assign

# === LGD (greedy initialization) ===
def lgd(V, p, pj, W_th, Y_th):
    density = [(V[i][j][t][r] * pj[j] / (p[i] * r), i, j, t, r) for i in V for j in V[i] for t in V[i][j] for r in V[i][j][t]]
    density.sort()
    assignment = []
    remaining = copy.deepcopy(pj)
    assigned = set()
    for _, i, j, t, r in density:
        if i not in assigned and remaining[j] >= p[i] * r:
            assignment.append((i, j, t, r))
            remaining[j] -= p[i] * r
            assigned.add(i)
    return assignment

# === Test run ===
historical = {
    1: np.random.rand(10, 3),
    2: np.random.rand(10, 3)
}

users = [
    User(1, 1.0, 0.1, 2.0, 40e6,
         [(np.random.rand()*600, np.random.rand()*600) for _ in range(10)],
         [(2, 40e6, 20e6) for _ in range(10)]),
    User(2, 0.8, 0.15, 3.0, 50e6,
         [(np.random.rand()*600, np.random.rand()*600) for _ in range(10)],
         [(3, 50e6, 25e6) for _ in range(10)])
]

edges = [
    EdgeNode(0, 100, 100, 100, 80e6),
    EdgeNode(1, 300, 300, 150, 90e6),
    EdgeNode(2, 500, 500, 120, 70e6)
]

mu, W, Y = aads(users, edges, 0.85, BETA, historical)
print("\n✅ Deployment decisions:")
print(dict(mu))
print("Periods W:", W)
print("Copies Y:", Y)
