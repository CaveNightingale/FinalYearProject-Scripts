import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os
import gc

SRC = "../LLaMA-2-7B"
TEMP = "./temp"
CALIB = "./data/c4_calib_4k"
OUT_DIR = "module_distribution"

def main():
    import datasets
    from transformers import LlamaForCausalLM, LlamaTokenizer
    import torch

    tokenizer = LlamaTokenizer.from_pretrained(SRC)
    model = LlamaForCausalLM.from_pretrained(SRC, device_map='cpu', dtype='auto')
    calib = datasets.load_from_disk(CALIB)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEMP, exist_ok=True)

    def linear_pre_hook(module, inputs):
        x = inputs[0]
        module._captured_input = x.detach().cpu().numpy()[0]
        return inputs

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not ('lm_head' in name):
            module.register_forward_pre_hook(linear_pre_hook)

    # We take first 4096 tokens from the calibration dataset as input and run a forward pass to capture the input distribution of linear layers.
    input_ids = []
    for example in calib:
        input_ids.extend(tokenizer(example['text']).input_ids)
        if len(input_ids) >= 4096:
            break

    input_ids = input_ids[:4096]
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # (1, seq_len)
    with torch.no_grad():
        output = model(input_ids.to(model.device), use_cache=True, return_dict=True)
        kv_cache = output.past_key_values

    # Visualize KV cache distribution
    for layer_idx, kv in enumerate(kv_cache.layers):
        k = kv.keys.permute(0, 2, 1, 3).reshape(-1, 4096).detach().cpu().numpy()
        v = kv.values.permute(0, 2, 1, 3).reshape(-1, 4096).detach().cpu().numpy()

        np.savez(f"{TEMP}/kv.layers.{layer_idx}.npz", k=k, v=v)


    # Visualize linear modules
    for name, module in model.named_modules():
        # Make a 4x4 grid of subplots for each linear layer
        # Weight Distribution | Activation Distribution
        # Weight QQ Plot to Normal | Activation QQ Plot to Normal
        # Weight Heatmap | Activation Heatmap (for the first 1024 elements)
        if isinstance(module, torch.nn.Linear) and not ('lm_head' in name):
            weight = module.weight.cpu().detach().numpy()
            activation = module._captured_input

            np.savez(f"{TEMP}/{name}.npz", w=weight, a=activation)

def plot_kv(name):
    plt.ioff()
    z = np.load(f"{TEMP}/{name}.npz")
    k = z['k']
    v = z['v']

    k_flat = k.ravel()
    v_flat = v.ravel()

    k_mean = np.mean(k.astype(np.float64)).astype(np.float32)
    k_var = np.var(k.astype(np.float64)).astype(np.float32)
    v_mean = np.mean(v.astype(np.float64)).astype(np.float32)
    v_var = np.var(v.astype(np.float64)).astype(np.float32)

    fig, axes = plt.subplots(4, 2, figsize=(10, 18))

    sns.histplot(k_flat[(k_flat > -12) & (k_flat < 12)], bins=400, color='skyblue', kde=True, ax=axes[0][0])
    axes[0][0].set_xlim(-6, 6)
    axes[0][0].set_title(f"Key Distribution, shape={k.shape}")
    sns.histplot(v_flat[(v_flat > -3) & (v_flat < 3)], bins=400, color='salmon', kde=True, ax=axes[0][1])
    axes[0][1].set_xlim(-1.5, 1.5)
    axes[0][1].set_title(f"Value Distribution, shape={v.shape}")

    stats.probplot(k_flat, dist="norm", plot=axes[1][0])
    axes[1][0].set_title(f"Key QQ Plot to Normal, mean={k_mean:.4f}, var={k_var:.4f}")
    stats.probplot(v_flat, dist="norm", plot=axes[1][1])
    axes[1][1].set_title(f"Value QQ Plot to Normal, mean={v_mean:.4f}, var={v_var:.4f}")

    stats.probplot(k_flat, dist="laplace", plot=axes[2][0])
    axes[2][0].set_title(f"Key QQ Plot to Laplace, kurtosis={stats.kurtosis(k_flat.astype(np.float64)):.4f}")
    stats.probplot(v_flat, dist="laplace", plot=axes[2][1])
    axes[2][1].set_title(f"Value QQ Plot to Laplace, kurtosis={stats.kurtosis(v_flat.astype(np.float64)):.4f}")

    sns.heatmap(k[:1024, :1024], cmap='viridis', ax=axes[3][0])
    axes[3][0].set_title(f"Key Heatmap (first 1024x1024)")
    sns.heatmap(v[:1024, :1024], cmap='magma', ax=axes[3][1])
    axes[3][1].set_title(f"Value Heatmap (first 1024x1024)")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.jpg", dpi=600)
    print(f"Saved KV cache distribution for layer {name} to {OUT_DIR}/{name}.jpg")
    fig.clf()
    plt.close(fig)

def plot_linear(name):
    print(f"Plotting {name}")

    plt.ioff()
    z = np.load(f"{TEMP}/{name}.npz")
    w, a = z['w'], z['a']
    weight, activation = w.flatten(), a.flatten()

    weight_mean = np.mean(weight.astype(np.float64)).astype(np.float32)
    weight_var = np.var(weight.astype(np.float64)).astype(np.float32)
    activation_mean = np.mean(activation.astype(np.float64)).astype(np.float32)
    activation_var = np.var(activation.astype(np.float64)).astype(np.float32)

    print(f"{name}, mean var")
    

    fig, axes = plt.subplots(4, 2, figsize=(10, 18))
    sns.histplot(weight[(weight > -0.5) & (weight < 0.5)], bins=400, color='skyblue', kde=True, ax=axes[0][0])
    axes[0][0].set_xlim(-0.2, 0.2)
    axes[0][0].set_title(f"Weight Distribution, shape={tuple(w.shape)}")
    sns.histplot(activation[(activation > -1) & (activation < 1)], bins=400, color='salmon', kde=True, ax=axes[0][1])
    axes[0][1].set_xlim(-0.5, 0.5)
    axes[0][1].set_title(f"Activation Distribution, shape={tuple(a.shape)}")

    print(f"{name}, hist")

    stats.probplot(weight, dist="norm", plot=axes[1][0])
    axes[1][0].set_title(f"Weight QQ Plot to Normal, mean={weight_mean:.4f}, var={weight_var:.4f}")
    stats.probplot(activation, dist="norm", plot=axes[1][1])
    axes[1][1].set_title(f"Activation QQ Plot to Normal, mean={activation_mean:.4f}, var={activation_var:.4f}")

    print(f"{name}, norm qq")

    stats.probplot(weight, dist="laplace", plot=axes[2][0])
    axes[2][0].set_title(f"Weight QQ Plot to Laplace, kurtosis={stats.kurtosis(weight.astype(np.float64)):.4f}")
    stats.probplot(activation, dist="laplace", plot=axes[2][1])
    axes[2][1].set_title(f"Activation QQ Plot to Laplace, kurtosis={stats.kurtosis(activation.astype(np.float64)):.4f}")

    print(f"{name}, laplace qq")

    sns.heatmap(w[:1024, :1024], cmap='viridis', ax=axes[3][0])
    axes[3][0].set_title(f"Weight Heatmap (first 1024x1024)")
    sns.heatmap(a[:1024, :1024], cmap='magma', ax=axes[3][1])
    axes[3][1].set_title(f"Activation Heatmap (first 1024x1024)")

    print(f"{name}, heatmap")

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{name}.jpg", dpi=600)

    print(f"{name}, saved")
    
    plt.close(fig)

def str_hash(s):
    sum = 0
    for c in s:
        sum = (sum * 31 + ord(c)) % (2**32)
    return sum

if __name__ == "__main__":
    import sys
    python = sys.executable
    plot_name = os.environ.get("PLOT_NAME", None)
    chunks = os.environ.get("CHUNKS", "1")
    plot_chunk = os.environ.get("PLOT_CHUNK", None)

    if plot_name is not None:
        if plot_name.startswith("kv.layers."):
            plot_kv(plot_name)
        else:
            plot_linear(plot_name)
    elif plot_chunk is not None:
        plot_chunk = int(plot_chunk)
        chunks = int(chunks)
        queue = []
        for file in os.listdir(TEMP):
            if file.endswith(".npz"):
                plot_name = file[:-4]
                target = f"{OUT_DIR}/{plot_name}.jpg"
                if os.path.exists(target):
                    continue
                if str_hash(plot_name) % chunks == plot_chunk:
                    queue.append(plot_name)
        max_processes = 2
        for i in range(min(max_processes, len(queue))):
            env = os.environ.copy()
            env["PLOT_NAME"] = queue[0]
            os.spawnve(os.P_NOWAIT, python, ["python", __file__], env)
            queue.pop(0)
        while queue:
            pid, status = os.wait()
            if os.WIFEXITED(status):
                print(f"Process {pid} finished with exit code {os.WEXITSTATUS(status)}")
            if os.WIFSIGNALED(status):
                print(f"Process {pid} killed by signal {os.WTERMSIG(status)}")
            env = os.environ.copy()
            env["PLOT_NAME"] = queue[0]
            os.spawnve(os.P_NOWAIT, python, ["python", __file__], env)
            queue.pop(0)
    else:
        main()