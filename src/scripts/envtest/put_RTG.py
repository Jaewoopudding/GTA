
import numpy as np
from tqdm import tqdm
import argparse

def discounted_cumsum(x: np.ndarray, gamma: float):
    cumsum = np.zeros_like(x)
    cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        cumsum[t] = x[t] + gamma * cumsum[t + 1]
    return cumsum

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="halfcheetah_medium")
    args = parser.parse_args()
    env = args.env
    gen_datapath = "/home/orl/temp0130/revert"
    org_datapath = "/home/sujin/project/Augmentation-For-OfflineRL/data"

    gen = np.load(f"{gen_datapath}/{env}.npz", allow_pickle=True)
    org = np.load(f"{org_datapath}/{env}.npy", allow_pickle=True)
    gen_d = gen["data"]
    gen_c = gen["config"]

    for episode in tqdm(range(len(org))):
        org[episode]["RTG"] = discounted_cumsum(org[episode]["rewards"], 1.)

    seq_len = 31
    obs_shape = org[0]["observations"].shape[1]
    action_shape = org[0]["actions"].shape[1]

    for idx in tqdm(range(len(gen_d))):
        for epi in org:
            ts = gen_d[idx]["timesteps"][0]
            if ts <= len(epi["observations"])-seq_len:
                if np.equal(gen_d[idx]["original_observations"][0], epi["observations"][ts]).sum() == obs_shape:
                    if np.equal(gen_d[idx]["original_actions"][0], epi["actions"][ts]).sum() == action_shape:
                        if np.equal(gen_d[idx]["original_observations"][4], epi["observations"][ts+4]).sum() == obs_shape:
                            gen_d[idx]["RTG"] = epi["RTG"][ts:ts+seq_len]
                            break
        if "RTG" not in gen_d[idx].keys():
            print(f"Not found........{idx}")
            raise ValueError("RTG not found")
    

    np.savez(f"/home/orl/0117_reweighting_rtg/{env}.npz", data=gen_d, config=gen_c)


