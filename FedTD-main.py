import argparse
import numpy as np
import pandas as pd
import os
import sys
import itertools

###############################################################################
# Helper functions
###############################################################################
def generate_random_mdp(n, rng, reward_type="uniform"):
    """
    Generate a random MDP with n states.
    Returns:
      - P: (n x n) transition matrix (row-stochastic)
      - r: (n, ) reward vector in [0, 1]
    reward_type options:
      - "uniform": Uniform[0,1]
      - "gaussian": Gaussian with mean 0.5 and std 0.15, clipped to [0,1]
    """
    # Random transition matrix
    mat = rng.random((n, n))
    P = mat / mat.sum(axis=1, keepdims=True)
    
    # Rewards
    if reward_type.lower() == "uniform":
        r = rng.random(n)  # uniform [0,1]
    elif reward_type.lower() == "gaussian":
        r = rng.normal(loc=0.5, scale=0.15, size=n)
        r = np.clip(r, 0.0, 1.0)
    else:
        raise ValueError(f"Unknown reward_type={reward_type}. Use 'uniform' or 'gaussian'.")
    
    return P, r


def project_onto_ball(M, center, Delta):
    """
    Project the matrix M onto the Euclidean ball of radius Delta around 'center',
    i.e. ensure ||M - center||_2 <= Delta.
    
    We interpret ||.||_2 as the Euclidean norm on the flattened matrix (Frobenius).
    """
    diff = M - center
    dist = np.linalg.norm(diff, 'fro') #(diff.ravel(), 2)
    if dist > Delta:
        M = center + (diff * (Delta / dist))
    return M


def perturb_mdp(P, r, Delta, rng):
    """
    Create a perturbed MDP (P_pert, r_pert) from baseline (P, r),
    ensuring ||P_pert - P||_2 <= Delta in Euclidean sense (flattened matrix).
    We'll do random noise, then project to the ball of radius Delta, 
    and re-normalize rows to be row-stochastic.

    We do NOT perturb the reward in this version (per "we do not have noisy reward").
    If you prefer also to perturb r, you can add code here.
    """
    n = P.shape[0]
    # create random noise for each entry
    noise = rng.normal(loc=0.0, scale=Delta*0.5, size=P.shape)
    M = P + noise
    
    # Project M onto ball around P with radius Delta
    M = project_onto_ball(M, P, Delta)
    
    # ensure positivity, then row-stochastic
    M = np.clip(M, 1e-12, None)
    M = M / M.sum(axis=1, keepdims=True)
    
    # keep reward the same: r_pert = r
    return M, r


def compute_stationary_dist(P, tol=1e-12, max_iter=10_000):
    """
    Compute approximate stationary distribution of P via power iteration.
    Return a vector mu of shape (n,).
    """
    n = P.shape[0]
    mu = np.ones(n) / n
    for _ in range(max_iter):
        mu_next = mu @ P
        if np.linalg.norm(mu_next - mu, 1) < tol:
            break
        mu = mu_next
    return mu


def bellman_fixed_point(P, r, gamma=0.9):
    """
    Solve (I - gamma*P) V = r for V.  (Tabular exact solution.)
    """
    n = len(r)
    A = np.eye(n) - gamma * P
    return np.linalg.solve(A, r)


def sample_iid_state(n, mu, rng):
    """
    Sample a state index from a distribution mu (size n).
    """
    return rng.choice(n, p=mu)


def run_markov_chain_step(s, P, rng):
    """
    Markov sampling: from state s, pick next with prob. row P[s].
    """
    return rng.choice(P.shape[0], p=P[s])


###############################################################################
# FedTD(0) main
###############################################################################
def fed_td_experiment(
    n=100,
    N=5,
    Delta=0.1,
    gamma=0.9,
    alpha=0.01,
    beta=0.1,
    T=50,
    K=5,
    sampling="iid",        # "iid" or "markov"
    iid_option="uniform",  # "uniform" or "stationary"
    reward_type="uniform", # "uniform" or "gaussian"
    seeds=[0,1,2,3,4],
    out_dir="./results"
):
    """
    Run Federated TD(0) under the specified config for multiple seeds.
    
    We do:
    - Generate baseline MDP (P, r), then compute its V^* (the "true" value).
    - For i in [1..N], create (P_i, r_i) via small perturbation, s.t. ||P_i - P||_2 <= Delta
      (No reward noise; r_i = r).
    - Each agent does local TD(0) for K steps, then we average with global step-size beta.
    - sampling="iid":
        - If iid_option=="uniform", sample s_i^{t,k} from uniform across n states
        - If iid_option=="stationary", sample from mu_i (stationary dist of P_i)
      Then we do the next state from P_i(s, .) just for the TD backup.
    - sampling="markov":
        - We keep a separate chain for each agent, s <- P_i(s).
    - Measures:
      - error_l2: ||V_global - V*||_2
      - error_l2sq: ||V_global - V*||_2^2
      - mse: mean squared error = error_l2sq / n
      - rmse: root mean squared error = error_l2 / sqrt(n)
    
    Save results in CSV with columns:
        seed, round, error_l2, error_l2sq, mse, rmse, <config columns> ...
    Filenames unique to the config.
    """
    rng = np.random.RandomState(999)  # just for naming or fallback
    # Create a unique filename reflecting the config
    config_str = (
        f"n_{n}_N_{N}_Delta_{Delta}_gamma_{gamma}_alpha_{alpha}_beta_{beta}"
        f"_T_{T}_K_{K}_s_{sampling}_iidopt_{iid_option}_R_{reward_type}"
    )
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"fedtd_{config_str}.csv")

    # Prepare CSV
    columns = [
        "seed", "round",
        "error_l2", "error_l2sq", "mse", "rmse", 
        "n","N","Delta","gamma","alpha","beta","T","K","sampling","iid_option","reward_type"
    ]
    df_rows = []

    for seed in seeds:
        # Initialize random generator
        rng = np.random.RandomState(seed)

        # Generate baseline MDP and its true value function V*
        P, r = generate_random_mdp(n, rng, reward_type=reward_type)
        V_star = bellman_fixed_point(P, r, gamma=gamma)

        # Generate N perturbed MDPs for each agent
        Ps, rs = [], []
        for i in range(N):
            P_i, r_i = perturb_mdp(P, r, Delta, rng)
            Ps.append(P_i)
            rs.append(r_i)

        # Initialize global value function V_global
        V_global = np.zeros(n, dtype=float)

        # Local copies for each agent
        V_local = np.zeros((N, n), dtype=float)

        # For i.i.d. sampling, set up distributions
        if sampling=="iid" and iid_option=="stationary":
            mus = []
            for i in range(N):
                mu_i = compute_stationary_dist(Ps[i])
                mus.append(mu_i)
        elif sampling=="iid" and iid_option=="uniform":
            # we'll just use uniform dist => [1/n]*n
            mus = [np.ones(n)/n for _ in range(N)]
        else:
            mus = [None]*N  # not used for markov

        # For Markov sampling, initialize each agent's state        
        agent_states = [rng.randint(n) for _ in range(N)]

        # Main loop of T communication rounds
        for t_comm in range(T):
            # Set each agent's local value to the current global value
            for i in range(N):
                V_local[i] = V_global.copy()

            # Perform K local TD(0) updates at each agent
            for k_step in range(K):
                for i in range(N):
                    Pi, ri = Ps[i], rs[i]
                    
                    # sample state
                    if sampling=="iid":
                        # pick s_i from mu_i or uniform
                        s_i = sample_iid_state(n, mus[i], rng)
                    else:
                        # markov
                        s_i = agent_states[i]

                    # next state for TD(0) update
                    s_next = run_markov_chain_step(s_i, Pi, rng)

                    # TD(0) update
                    td_err = ri[s_i] + gamma * V_local[i][s_next] - V_local[i][s_i]
                    V_local[i][s_i] += alpha * td_err

                    # Keep track of agent states for Markov chain
                    if sampling=="markov":
                        agent_states[i] = s_next
            
            # Global update: average the differences from the server value function
            avg_diff = np.mean(V_local - V_global, axis=0)  
            V_global = V_global + beta * avg_diff

            # Compute error metrics with respect to baseline true value V*
            diff_vec = V_global - V_star
            error_l2 = np.linalg.norm(diff_vec, 2)
            error_l2sq = error_l2**2
            mse = error_l2sq / n
            rmse = error_l2 / np.sqrt(n)
            
            row = [
                seed, t_comm, error_l2, error_l2sq, mse, rmse,
                n, N, Delta, gamma, alpha, beta, T, K, sampling, iid_option, reward_type
            ]
            df_rows.append(row)

    # Save all results to a CSV file
    df = pd.DataFrame(df_rows, columns=columns)
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to {out_csv}\nConfiguration:\n{config_str}")
    
# --- Argparse to Parse Command-Line Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Federated TD(0) Simulation")
    parser.add_argument("--n", type=int, default=10, help="Number of states")
    parser.add_argument("--N", type=int, default=10, help="Number of agents")
    parser.add_argument("--Delta", type=float, default=0.01, help="Heterogeneity level")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
    parser.add_argument("--alpha", type=float, default=0.01, help="Local step size")
    parser.add_argument("--beta", type=float, default=0.1, help="Global step size")
    parser.add_argument("--T", type=int, default=100000, help="Total number of communication rounds")
    parser.add_argument("--K", type=int, default=5, help="Number of local steps per round")
    parser.add_argument("--sampling", type=str, choices=["iid", "markov"], default="iid", help="Sampling mode")
    parser.add_argument("--iid_option", type=str, choices=["uniform", "stationary"], default="uniform", help="IID option")
    parser.add_argument("--reward_type", type=str, choices=["uniform", "gaussian"], default="uniform", help="Reward type")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4], help="List of random seeds")
    parser.add_argument("--out_dir", type=str, default="./results", help="Output directory for CSV")
    return parser.parse_args()

def main():
    args = parse_args()
    print("Running FedTD Simulation with the following parameters:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    fed_td_experiment(
        n=args.n,
        N=args.N,
        Delta=args.Delta,
        gamma=args.gamma,
        alpha=args.alpha,
        beta=args.beta,
        T=args.T,
        K=args.K,
        sampling=args.sampling,
        iid_option=args.iid_option,
        reward_type=args.reward_type,
        seeds=args.seeds,
        out_dir=args.out_dir
    )

if __name__ == "__main__":
    main()