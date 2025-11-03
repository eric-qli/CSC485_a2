# STUDENT NAME: NAME
# STUDENT NUMBER: NUMBER
# UTORID: ID

'''
This code is provided solely for the personal and private use of students
taking the CSC485H/2501H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Jinman Zhao, Jingcheng Niu, Ruiyu Wang, Gerald Penn

All of the files in this directory and all subdirectories are:
Copyright (c) 2025 University of Toronto
'''

from argparse import Namespace
from typing import Dict, List, Tuple, Callable, Optional

import torch
from torch import Tensor
from transformer_lens import HookedTransformer
import transformer_lens.utils as utils
torch.set_grad_enabled(False)

from matplotlib import pyplot as plt
import seaborn as sns

def plot_heatmap(patch_result: Tensor, output_dir: str, cmap: str = 'Purples') -> None:
    """
    Plot a heatmap of the causal tracing results.

    Args:
    patch_result (torch.Tensor): 2D tensor of shape (sequence_length - 1, n_layers) containing the causal tracing results.
    output_dir (str): Path to save the output heatmap.
    cmap (str): Colormap to use for the heatmap.

    Returns:
    None
    """
    fig, ax = plt.subplots()
    sns.heatmap(patch_result, ax=ax, cmap=cmap)
    fig.savefig(output_dir)

class TraceTransformer(HookedTransformer):
    """
    A custom transformer model class for performing causal tracing analysis.
    Inherits from HookedTransformer and adds methods for causal tracing.
    """

    def get_target_id(self, token: str) -> int:
        """
        Get the token ID for a given target token.

        Args:
        token (str): The target token.

        Returns:
        int: The token ID.
        """
        encoded_tokens = self.tokenizer.encode(' ' + token)
        assert len(encoded_tokens) == 1
        return encoded_tokens[0]

    def record_clean_activations(self, prompt: str) -> Dict[str, Tensor]:
        """
        Record the clean activations for a given prompt.

        Args:
        prompt (str): The input prompt.

        Returns:
        Dict[str, torch.Tensor]: A dictionary containing the clean activations for each layer.
        """
        prompt_token = self.to_tokens(prompt)
        logits, activations = self.run_with_cache(prompt_token)
        return activations

    def get_corrupted_probs(self,
            prompt: str, patch_embed_fn: Callable) -> Tensor:
        """
        Get the corrupted probabilities for a given prompt and patching function.

        Args:
        prompt (str): The input prompt.
        patch_embed_fn (Callable): The function to patch the embeddings.

        Returns:
        torch.Tensor: The corrupted probabilities for the last token.
        """
        # 1) Tokenize -> token ids [1, T]
        toks = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]

        # 2) Find a HookedTransformer-like object on self
        model_ref = None
        for v in self.__dict__.values():
            # Heuristic: TL models have .add_hook and a 'hook_embed' hook name
            if hasattr(v, "add_hook") and hasattr(v, "hooks"):
                model_ref = v
                break
        if model_ref is None:
            raise AttributeError("Could not find a TransformerLens HookedTransformer on this object.")

        # 3) Move tokens to the model’s device (use embedding weight’s device)
        #    TL uses model_ref.W_E or model_ref.embed.W_E for embeddings.
        W_E = getattr(model_ref, "W_E", None) or getattr(getattr(model_ref, "embed", None), "W_E", None)
        if W_E is None:
            raise AttributeError("Could not locate embedding weight (W_E) on the TL model.")
        device = W_E.device
        toks = toks.to(device)

        # 4) Define the forward hook for 'hook_embed'
        def fwd_hook(value, hook):
            # value: [B, T, d_model] residual stream right after embeddings
            patched = patch_embed_fn(value)
            if patched.shape != value.shape:
                raise ValueError(f"patch_embed_fn changed shape {patched.shape} vs {value.shape}")
            if patched.device != value.device:
                patched = patched.to(value.device)
            if patched.dtype != value.dtype:
                patched = patched.to(value.dtype)
            return patched

        # 5) Run with the hook (context manager keeps it clean)
        # TL API: model_ref.hooks(fwd_hooks=[("hook_embed", fwd_hook)])
        with model_ref.hooks(fwd_hooks=[("hook_embed", fwd_hook)]):
            with torch.no_grad():
                # TL forward returns logits [B, T, V]
                logits = model_ref(toks)

        # 6) Choose target position
        if hasattr(self, "get_target_id") and callable(getattr(self, "get_target_id")):
            target_pos = int(self.get_target_id({"input_ids": toks}))
            target_pos = max(0, min(logits.size(1) - 1, target_pos))
        else:
            target_pos = logits.size(1) - 1

        # 7) Return probs at that position
        probs = F.softmax(logits[0, target_pos], dim=-1)  # [V]
        return probs

    

    def find_sequence_span(self, prompt: str, seq: str) -> Tensor:
        """
        Find the token indices for a given sequence in the prompt.

        Args:
        prompt (str): The input prompt.
        seq (str): The sequence to find in the prompt.

        Returns:
        torch.Tensor: A tensor containing the indices of the sequence in the prompt.
        """
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        seq_ids = self.tokenizer(seq, add_special_tokens=False)["input_ids"]

        if not prompt_ids or not seq_ids or len(seq_ids) > len(prompt_ids):
            return torch.empty(0, dtype=torch.long)

        found_start = -1
        for i in range(len(prompt_ids) - len(seq_ids) + 1):
            if prompt_ids[i : i + len(seq_ids)] == seq_ids:
                found_start = i
                break

        if found_start == -1:
            return torch.empty(0, dtype=torch.long)

        indices = list(range(found_start, found_start + len(seq_ids)))

        return torch.tensor(indices, dtype=torch.long)


    def get_patch_emb_fn(self, corrupt_span: Tensor, noise: float = 1.) -> Callable:
        """
        Get a function to patch the embeddings with noise.

        Args:
        corrupt_span (torch.Tensor): The span of tokens to corrupt.
        noise (float): The amount of noise to add.

        Returns:
        Callable: A function that patches the embeddings with noise.
        """
        def patch_fn(act: Tensor) -> Tensor:
            if act.ndim < 2:
                return act
            
            B, T = act.shape[0], act.shape[1]
            device = act.device

            span = corrupt_span.to(device)

            # Case 1: already a boolean mask
            if span.dtype == torch.bool:
                if span.ndim < 2:
                    mask_bt = span.unsqueeze(0).expand(B, -1)
                elif span.ndim == 2:
                    mask_bt = span

            # Case 2: not a boolean mask
            else:
                if span.ndim < 2:
                    mask = torch.zeros(T, dtype=torch.bool, device=device)
                    if span.numel() > 0:
                        idx = span.long().clamp(0, T - 1)
                        mask[idx] = True
                    mask_bt = mask.unsqueeze(0).expand(B, -1)

                elif span.ndim == 2:
                    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
                    for b in range(B):
                        row = span[b]
                        row = row[row >= 0].long() if (row.ndim == 1) else row.long()
                        if row.numel() > 0:
                            row = row
                            mask[b, row] = True
                    
                    mask_bt = mask
        
            while mask_bt.ndim < act.ndim:
                mask_bt = mask_bt.unsqueeze(-1)

            if torch.is_float_point(act):
                noise_tensor = torch.randn_like(act) * float(noise) 
            
            return act + noise_tensor * mask_bt

        return patch_fn


    def get_restore_fn(self,
            activation_record: Dict[str, Tensor], token_idx: int) -> Callable:
        """
        Get a function to restore the activations for a specific token.

        Args:
        activation_record (Dict[str, torch.Tensor]): The recorded clean activations.
        token_idx (int): The index of the token to restore.

        Returns:
        Callable: A function that restores the activations for the specified token.
        """
        def restore_fn(patched: Tensor, original: Tensor) -> Tensor:
            if patched.shape != original.shape:
                return patched
            
            if patched.device != original.device:
                patched = patched.to(original.device)
            
            if patched.dtype != original.dtype:
                patched = patched.to(original.dtype)
            
            if patched.ndim < 2:
                return patched
            
            B, T = patched.shape[0], patched.shape[1]
            idx = max(0, min(token_idx, T - 1))

            out = original.clone()
            out[:, idx] = patched[:, idx]
            return out

        return restore_fn


    def get_forward_hooks(self, layer: int,
            patch_embed_fn: Callable, patch_name: str,
            restore_fn: Callable, window: int = 10) -> List[Tuple[str, Callable]]:
        """
        Get the forward hooks for causal tracing.

        Args:
        layer (int): The current layer.
        patch_embed_fn (Callable): The function to patch the embeddings.
        patch_name (str): The name of the patch location ('resid_pre', 'mlp_post', or 'attn_out').
        restore_fn (Callable): The function to restore activations.
        window (int): The window size for tracing.

        Returns:
        List[Tuple[str, Callable]]: A list of tuples containing the hook names and functions.
        """
        if patch_name not in {"resid_pre", "mlp_post", "attn_out"}:
            raise ValueError(f"Unknown patch_name={patch_name}. "
                            "Expected one of: 'resid_pre', 'mlp_post', 'attn_out'.")

        hook_name = f"L{layer}.{patch_name}"

        center_idx = getattr(self, "target_idx", None)

        def _apply_window(orig: Tensor, repl: Tensor) -> Tensor:
            """
            Replace a [B, T, ...] slice around center_idx with repl’s slice.
            If center_idx is None or tensor doesn't look like [B,T,...], replace all.
            """
            if orig.ndim < 2 or center_idx is None or window is None:
                return repl 

            B, T = orig.shape[0], orig.shape[1]
            c = max(0, min(T - 1, int(center_idx)))
            start = max(0, c - int(window))
            end = min(T, c + int(window) + 1)

            out = orig.clone()
            out[:, start:end, ...] = repl[:, start:end, ...]
            return out

        def hook_fn(module, inputs, output):
            """
            Forward hook: intercept activation, patch with replacement from patch_embed_fn,
            optionally window it, then call restore_fn(patched, original) and return.
            """
            is_tuple = isinstance(output, tuple)
            act = output[0] if is_tuple else output

            repl = patch_embed_fn(act)

            if repl.shape != act.shape:
                raise ValueError(f"{hook_name}: patch_embed_fn changed shape "
                                f"{repl.shape} vs {act.shape}")
            if repl.device != act.device:
                repl = repl.to(act.device)
            if repl.dtype != act.dtype:
                repl = repl.to(act.dtype)

            patched = _apply_window(act, repl)

            if restore_fn is not None:
                patched = restore_fn(patched, act)

            if is_tuple:
                out_list = list(output)
                out_list[0] = patched
                return tuple(out_list)
            return patched

        return [(hook_name, hook_fn)]
    

    def causal_trace_analysis(self,
            prompt: str, source: str, target: str,
            patch_name: str, noise: float = 1., window: int = 10) -> Tensor:
        """
        Perform causal tracing analysis on the model.

        Args:
        prompt (str): The input prompt.
        source (str): The source sequence to corrupt.
        target (str): The target token to predict.
        patch_name (str): The name of the patch location
            ('resid_pre', 'mlp_post', or 'attn_out').
        noise (float): The amount of noise to add when corrupting.
        window (int): The window size for tracing.

        Returns:
        torch.Tensor: A 2D tensor of shape (sequence_length - 1, n_layers) containing the causal tracing results.
        """
        # 1) Sequence length from tokenizer (CPU is fine)
        enc = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        input_ids = enc["input_ids"]           # [1, T]
        seq_len = input_ids.size(1)

        # 2) Target id (use first token if multi-token)
        tgt_ids = self.tokenizer(target, add_special_tokens=False)["input_ids"]
        if not tgt_ids:
            raise ValueError(f"Could not tokenize target string: {target!r}")
        target_id = tgt_ids[0]

        # 3) Build corruption function for the source span
        span_idx = self.find_sequence_span(prompt, source)
        if span_idx.numel() == 0:
            raise ValueError(f"Source sequence {source!r} not found in prompt.")
        patch_emb_corrupt = self.get_patch_emb_fn(span_idx, noise=noise)

        # 4) Baseline corrupted probability (no layer patch)
        probs_corrupt = self.get_corrupted_probs(prompt, patch_emb_corrupt)  # [V] on model's device
        base_corrupt_p = probs_corrupt[target_id].item()

        # 5) Record clean activations (returns dict of tensors keyed by hook-name)
        activation_record: Dict[str, Tensor] = self.record_clean_activations(prompt)

        # 6) Infer number of layers from activation_record keys (no model access)
        #    Expect keys like "L{layer}.{patch_name}"
        layer_ids = []
        pat = re.compile(r"^L(\d+)\." + re.escape(patch_name) + r"$")
        for k in activation_record.keys():
            m = pat.match(k)
            if m:
                layer_ids.append(int(m.group(1)))
        if not layer_ids:
            raise KeyError(
                f"No activations found for patch_name={patch_name!r} in activation_record keys: "
                f"{list(activation_record.keys())[:5]} ..."
            )
        n_layers = max(layer_ids) + 1

        # 7) Output effects matrix on CPU
        effects = torch.zeros(seq_len - 1, n_layers)

        # 8) Small helpers to register/remove hooks via self.hook_sites
        def _register_hooks(hooks: List[Tuple[str, Callable]]):
            handles = []
            for hook_key, hook_fn in hooks:
                if not hasattr(self, "hook_sites") or hook_key not in self.hook_sites:
                    raise KeyError(
                        f"Hook site {hook_key!r} not found in self.hook_sites; "
                        "map it to the correct module before tracing."
                    )
                handles.append(self.hook_sites[hook_key].register_forward_hook(hook_fn))
            return handles

        def _remove_hooks(handles):
            for h in handles:
                h.remove()

        # 9) Sweep positions and layers
        for pos in range(seq_len - 1):
            # restore function that keeps only `pos` patched
            restore_fn = self.get_restore_fn(activation_record, token_idx=pos)

            for layer in range(n_layers):
                hook_key = f"L{layer}.{patch_name}"
                if hook_key not in activation_record:
                    effects[pos, layer] = float("nan")
                    continue

                clean_act = activation_record[hook_key]

                # Patch function that returns *clean* activation with correct device/dtype
                def patch_clean(act, _clean=clean_act):
                    return _clean.to(act.device, dtype=act.dtype)

                hooks = self.get_forward_hooks(
                    layer=layer,
                    patch_embed_fn=patch_clean,
                    patch_name=patch_name,
                    restore_fn=restore_fn,
                    window=window,
                )

                handles = _register_hooks(hooks)
                try:
                    # Run a corrupted pass (embeddings are noised) with this layer patched
                    probs_patched = self.get_corrupted_probs(prompt, patch_emb_corrupt)  # [V]
                finally:
                    _remove_hooks(handles)

                p_patched = probs_patched[target_id].item()
                effects[pos, layer] = p_patched - base_corrupt_p

        return effects


def run_causal_trace(model_name='gpt2-xl', patch_name='resid_pre',
    prompt=None, source=None, target=None,
    cache_dir='/u/csc485h/fall/pub/tl_models_cache/') -> None:
    """
    Perform causal tracing analysis for a specific patch location in a language model.

    Args:
    model_name (str): The name of the language model to load, such as 'gpt2-xl'. Defaults to 'gpt2-xl'.
    patch_name (str): The name of the patch location where the causal trace is applied. 
                      Can be one of 'resid_pre', 'mlp_post', or 'attn_out'. 
                      Determines which component to perturb during analysis. Defaults to 'resid_pre'.
    prompt (str): The input prompt.
    source (str): The source sequence to corrupt.
    target (str): The target token to predict.
    cache_dir (str):  The directory where the model are cached.
                      Defaults to '/u/csc485h/fall/pub/tl_models_cache/' on teach.cs.

    Returns:
    None: This function does not return a value but generates a heatmap saved as a PDF 
          showing the results of the causal trace analysis.
    """
    cmap, name = {
        'resid_pre': ('Purples', 'states'),
        'mlp_post': ('Greens', 'mlp'),
        'attn_out': ('Reds', 'attn'),
    }[patch_name]

    model = TraceTransformer.from_pretrained(model_name,
        cache_dir=cache_dir, local_files_only=True)
    # You can turn off local_files_only to allow downloads, but note that models
    # like gpt2-xl are large (over 6GB). These models are already cached on teach.cs.
    result = model.causal_trace_analysis(
        prompt=prompt, source=source, target=target,
        patch_name=patch_name,
        noise=0.5)

    plot_heatmap(result, name+'.pdf', cmap)

if __name__ == '__main__':
    model_name = 'gpt2'
    model_name = model_name

    request = {
        'prompt': 'The Eiffel Tower is located in the city of',
        'source': 'The Eiffel Tower',
        'target': 'Paris',
    }

    run_causal_trace(model_name=model_name, patch_name='resid_pre', **request)
    run_causal_trace(model_name=model_name, patch_name='mlp_post', **request)
    run_causal_trace(model_name=model_name, patch_name='attn_out', **request)