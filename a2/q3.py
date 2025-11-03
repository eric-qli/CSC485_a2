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
        tokens = self.to_tokens(prompt)

        logits = self.run_with_hooks(tokens, fwd_hook=[("hook_embed", patch_embed_fn)])

        probs = torch.softmax(logits[0][-1], dim=-1)

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
        prompt_tokens = self.to_tokens(prompt)
        seq_tokens = self.to_tokens(seq)

        if prompt_tokens and seq_tokens and prompt_tokens[0] == seq_tokens[0]:
            prompt_tokens, seq_tokens = prompt_tokens[1:], seq_tokens[1:]
        
        for i in range(len(prompt_tokens) - len(seq_tokens) + 1):
            if prompt_tokens[i:i + len(seq_tokens)] == seq_tokens:
                indices = list(range(i , i + len(seq_tokens)))
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

        def patch_emb_fn(activation: Tensor, hook):
            noise_tensor = torch.randn_like(activation[:, corrupt_span, :]) * noise

            activation[:, corrupt_span, :] += noise_tensor

            return activation

        return patch_emb_fn


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
        def restore_fn(act: Tensor, hook):
            if hook.name not in activation_record:
                return 
            
            clean = activation_record[hook.name]

            if act.dim < 3 or clean.dim < 3:
                return
            
            if clean.shape[0] != act.shape[0]:
                if clean.shape[0] == 1:
                    clean = clean.expand(act.shape[0], *clean.shape[1:])
                else:
                    return 
                
            if token_idx >= act.shape[1] or token_idx >= clean.shape[1]:
                return
            
            clean_slice = clean[:, token_idx, :].to(act.device, dtype=act.dtype)

            act[:, token_idx, :] = clean_slice

            return act

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
        hooks = []

        # 1️⃣ Always corrupt the embeddings
        hooks.append(("hook_embed", patch_embed_fn))

        # 2️⃣ Depending on what we're restoring:
        if patch_name == "resid_pre":
            # Restore at a specific layer's pre-residual input
            hooks.append((f"blocks.{layer}.hook_resid_pre", restore_fn))

        elif patch_name == "attn_out":
            # Restore at the attention output — possibly across a window of layers
            for l in range(layer, min(layer + window, self.cfg.n_layers)):
                hooks.append((f"blocks.{l}.hook_attn_out", restore_fn))

        elif patch_name == "mlp_post":
            # Restore at the MLP output — possibly across a window
            for l in range(layer, min(layer + window, self.cfg.n_layers)):
                hooks.append((f"blocks.{l}.hook_mlp_out", restore_fn))

        else:
            raise ValueError(f"Invalid patch_name '{patch_name}'")

        return hooks


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
        # ---- 1) Tokenize & basic sizes
        tokens = self.to_tokens(prompt)                # [B=1, T]
        seq_len = tokens.size(1)
        n_layers = self.cfg.n_layers

        # ---- 2) Find the source span to corrupt (token indices)
        corrupt_span = self.find_sequence_span(prompt, source)  # 1D tensor of positions

        # ---- 3) Build patch (embedding corruption) hook
        patch_embed_fn = self.get_patch_emb_fn(corrupt_span, noise=noise)

        # ---- 4) Target id for final-position probability
        target_id = self.get_target_id(target)

        # ---- 5) Corrupted baseline probability (run once)
        corrupted_probs = self.get_corrupted_probs(prompt, patch_embed_fn)  # [V]
        P_corrupt = corrupted_probs[0, target_id].item() if corrupted_probs.dim() == 2 else corrupted_probs[target_id].item()

        # ---- 6) Record clean activations (run once, clean)
        clean_record = self.record_clean_activations(prompt)

        # ---- 7) Allocate IE matrix
        ie = torch.zeros(seq_len - 1, n_layers)

        # We usually exclude the BOS at position 0; iterate tokens 1..(T-1)
        # These are the positions we attempt to restore one at a time.
        for token_idx in range(1, seq_len):
            # Restore function for this token position
            restore_fn = self.get_restore_fn(clean_record, token_idx)

            for layer in range(n_layers):
                # Build hooks: always corrupt embeddings + restore at chosen site/layer
                hooks = self.get_forward_hooks(
                    layer=layer,
                    patch_embed_fn=patch_embed_fn,
                    patch_name=patch_name,   # 'resid_pre' | 'attn_out' | 'mlp_post'
                    restore_fn=restore_fn,
                    window=window
                )

                # Run with the combined hooks
                logits = self.run_with_hooks(tokens, fwd_hooks=hooks)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                P_restored = probs[0, target_id].item()

                # Indirect Effect
                ie[token_idx - 1, layer] = P_restored - P_corrupt

        return ie



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