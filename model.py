"""
Model definitions for Vision Transformer variants with class-token attention, multi-class token transformers and token-dropout.
"""

import math
import random

import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class ConcreteGate(pl.LightningModule):
    """
    Stretched Concrete gate for differentiable sparsification.

    This module implements a gate based on the stretched Concrete (Gumbel-Sigmoid)
    distribution that can be applied to activations or weights in order to induce
    sparsity. It is typically used to softly mask attention weights or feature
    channels, with an associated L0-style regularization penalty.

    Parameters
    ----------
    shape : tuple[int, ...]
        Shape of the learnable gate parameter `log_a`. This should be broadcastable
        to the tensor you plan to gate. For example, to gate attention maps of
        shape `[batch, heads, tokens, tokens]` over the `heads` dimension, you can
        use `shape=(1, num_heads, 1, 1)`.
    device : str or torch.device
        Device where the gate parameters are expected to live. (Kept for
        compatibility with some usages; not strictly required inside the module.)
    temperature : float, optional
        Temperature parameter of the Concrete / Gumbel-Sigmoid distribution.
        Must be in `(0, 1]`. Lower values approximate hard (discrete) gates more
        closely but can make optimization more difficult.
    stretch_limits : tuple[float, float], optional
        Minimum and maximum values of the stretched Concrete distribution before
        being clipped to `[0, 1]`. The lower bound should be negative if L0-style
        regularization is desired, as in the original paper.
    l0_penalty : float, optional
        Coefficient for the L0 penalty encouraging sparsity (fewer open gates).
    eps : float, optional
        Small constant used for numerical stability when sampling and computing
        probabilities.
    hard : bool, optional
        If True, gates are binarized to `{0, 1}` in the forward pass, while the
        gradients are still computed through the continuous Concrete gates
        (straight-through estimator).
    local_rep : bool, optional
        If True, samples independent Concrete gates for each element in a batch
        using a noise tensor whose shape matches the incoming activation shape.
        If False, samples a single noise tensor with shape equal to `shape`.
    """

    def __init__(
        self,
        shape,
        device,
        temperature=0.33,
        stretch_limits=(-0.1, 1.1),
        l0_penalty=0.01,
        eps=1e-6,
        hard=False,
        local_rep=False,
    ):
        super().__init__()

        self.temperature, self.stretch_limits, self.eps = (
            temperature,
            stretch_limits,
            eps,
        )
        self.l0_penalty = l0_penalty
        self.hard, self.local_rep = hard, local_rep
        self.log_a = torch.nn.Parameter(
            torch.nn.init.xavier_uniform_(torch.empty(shape))
        )

    def forward(self, x):
        """
        Apply the Concrete gate to an input tensor.

        During training, random Concrete gates are sampled. At evaluation time,
        the deterministic mean gate is used.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to be modulated by the gate. Its shape must be
            broadcastable with the gate shape.

        Returns
        -------
        torch.Tensor
            Gated tensor with the same shape as `x`.
        """
        gates, _ = self.get_gates(shape=x.shape if self.local_rep else None)
        return x * gates

    def get_gates(self, shape=None):
        """
        Sample gate activations in the `[0, 1]` interval.

        Parameters
        ----------
        shape : tuple[int, ...], optional
            Shape of the noise tensor used for sampling during training. If
            omitted, defaults to `self.log_a.shape`.

        Returns
        -------
        clipped_concrete : torch.Tensor
            Gate activations in `[0, 1]` that will be applied to values.
        pre_clipped_concrete : torch.Tensor
            Same as `clipped_concrete` but before optional hard-binarization.
        """
        low, high = self.stretch_limits
        if self.training:
            shape = self.log_a.shape if shape is None else shape
            self.noise = torch.empty(shape).type_as(self.log_a)
            self.noise.uniform_(self.eps, 1.0 - self.eps)
            concrete = torch.sigmoid(
                (torch.log(self.noise) - torch.log(1 - self.noise) + self.log_a)
                / self.temperature
            )
        else:
            concrete = torch.sigmoid(self.log_a)
        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = torch.clamp(stretched_concrete, 0, 1)

        pre_clipped_concrete = clipped_concrete
        if self.hard:
            hard_concrete = torch.gt(clipped_concrete, 0.5).to(torch.float)
            clipped_concrete = (
                clipped_concrete + (hard_concrete - clipped_concrete).detach()
            )
        return clipped_concrete, pre_clipped_concrete

    def get_penalty(self, values=None, axis=None):
        """
        Compute the L0 penalty term associated with the gate.

        This computes the expected number of open gates (L0 norm) and multiplies
        it by the user-provided `l0_penalty`. If `values` are provided, the
        probability of being open is broadcast to match their shape.

        Parameters
        ----------
        values : torch.Tensor, optional
            Tensor that is being gated (e.g., activations or weights). Used only
            to broadcast the gate probabilities to the correct shape.
        axis : Any, optional
            Unused placeholder for interface compatibility.

        Returns
        -------
        torch.Tensor
            Scalar tensor representing the L0 regularization loss.
        """
        low, high = self.stretch_limits
        assert low < 0.0, (
            "p_gate_closed can be computed only if lower stretch limit is negative"
        )

        p_open = torch.sigmoid(
            self.log_a - self.temperature * torch.log(torch.tensor(-low / high))
        )
        p_open = torch.clamp(p_open, self.eps, 1.0 - self.eps)

        if values is not None:
            p_open += torch.zeros_like(values)  # broadcast shape to account for values
        l0_reg = self.l0_penalty * torch.sum(p_open)
        return torch.mean(l0_reg)

    def get_sparsity_rate(self):
        """
        Compute the fraction of currently active (non-zero) gates.

        Returns
        -------
        torch.Tensor
            Scalar tensor with the mean fraction of open gates across all
            elements of the gate tensor.
        """
        is_nonzero = torch.ne(self.get_gates()[0], 0.0)
        return torch.mean(is_nonzero.to(torch.float))


class Mlp(nn.Module):
    """
    Simple feed-forward network used inside transformer blocks.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """
    Multi-head self-attention with optional Concrete gating over attention maps.

    Parameters
    ----------
    dim : Embedding dimensionality
    num_heads : Number of attention heads.
    num_classes : Number of classes (used only for some attention visualizations).
    prune : If True, apply a `ConcreteGate` to the attention weights for pruning.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_classes=20,
        prune=False,
    ):
        super().__init__()
        self.prune = prune
        self.num_classes = num_classes
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attend = nn.Softmax(dim=-1)
        if self.prune:
            self.gate = ConcreteGate(
                (1, num_heads, 1, 1),
                device="cuda:0",
                hard=True,
                local_rep=False,
            )

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B x num_heads x N x N

        attn = self.attend(attn)
        attn = self.attn_drop(attn)

        weights = attn

        if self.prune:
            attn = self.gate(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, weights


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_classes=20,
        prune=False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            num_classes=num_classes,
            prune=prune,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        o, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(o)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class PatchEmbed(nn.Module):
    """
    Image to patch embedding via a convolutional projection.

    Parameters
    ----------
    img_size : int or tuple[int, int], optional
        Input image size (H, W). Can be a single int for square images.
    patch_size : int or tuple[int, int], optional
        Patch size (height, width). Determines convolution kernel and stride.
    in_chans : Number of input channels (e.g., 3 for RGB, 13 for Sentinel-2, etc.).
    embed_dim : Dimension of the output patch embeddings.
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class VisionTransformer(nn.Module):
    """
    Standard ViT backbone with optional attention pruning.
    mask_type : Any, optional
        Reserved argument for different masking strategies (unused here).
    prune : bool, optional
        If True, enable Concrete gate-based pruning of attention maps.
    return_att : bool, optional
        If True, return attention weights along with logits at inference.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=13,
        num_classes=8,
        embed_dim=768,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        mask_type=None,
        prune=False,
        return_att=False,
    ):
        super().__init__()

        self.return_att = return_att
        self.prune = prune
        self.num_classes = num_classes
        self.mask_type = mask_type
        self.num_features = self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_classes=num_classes,
                    prune=prune,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        Initialize module parameters.

        Linear layers are initialized with truncated normal weights and zero biases.
        LayerNorm weights are set to 1 and biases to 0.
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        """
        Resize positional encodings to match a given spatial resolution.
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    @torch.jit.ignore
    def no_weight_decay(self):
        """
        Parameters that should be excluded from weight decay.
        """
        return {"pos_embed", "cls_token"}

    def get_classifier(self):
        """
        Return the classifier head module.
        """
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        """
        Reset the classifier head for a new number of classes.
        """
        self.num_classes = num_classes
        self.head = (
            nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward_features(self, x, n):
        """
        Compute token embeddings and collect attention maps from the last `n` layers.
        Returns:
        cls_emb : torch.Tensor
            Class token embedding of shape `(B, C)`.
        attn_weights : list[torch.Tensor]
            List of attention maps from the last `n` blocks.
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], attn_weights

    def forward(self, x, n=12):
        """
        Forward pass through the Vision Transformer.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape `(B, C, H, W)`.
        n : int, optional
            Number of last layers from which to return attention maps.

        Returns
        -------
        torch.Tensor
            If `self.training` is True, returns logits of shape `(B, num_classes)`.
        tuple
            If `self.training` is False and `return_att` is True, returns
            `(logits, attn_weights)` where `attn_weights` is a list of attention
            maps from the last `n` layers.
        """
        x, attn_weights = self.forward_features(x, n)

        x = self.head(x)

        if self.training:
            return x
        else:
            return x, attn_weights


class MCTformerV2(VisionTransformer):
    """
    Multi-Class Token Transformer (MCTformerV2).

    This extends `VisionTransformer` by using one class token per class and an
    additional register token. It produces class-wise CAMs (class activation maps)
    and attention maps that can be used for weakly supervised localization.

    Parameters
    ----------
    *args :
        Positional arguments passed through to `VisionTransformer`.
    **kwargs :
        Keyword arguments passed through to `VisionTransformer`.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(
            self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head.apply(self._init_weights)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.register_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_classes + 1, self.embed_dim)
        )

        trunc_normal_(self.cls_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)

        self.sigmoid = nn.Sigmoid()
        self.depth = len(self.blocks)

        self.classifier_head_test = nn.Linear(
            self.embed_dim * self.num_classes, self.num_classes
        )

    def interpolate_pos_encoding(self, x, w, h):
        """
        Resize positional encodings for class tokens, patch tokens and register token.

        Returns
        -------
        torch.Tensor
            Positional encodings matching the token layout:
            `[class_tokens, patch_tokens, register_token]`.
        """
        npatch = x.shape[1] - self.num_classes - 1
        N = self.pos_embed.shape[1] - self.num_classes - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0 : self.num_classes]
        patch_pos_embed = self.pos_embed[:, self.num_classes : -1]
        reg_pos_embed = torch.unsqueeze(self.pos_embed[:, -1], 1)

        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(
                1, int(math.sqrt(N)), int(math.sqrt(N)), dim
            ).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert (
            int(w0) == patch_pos_embed.shape[-2]
            and int(h0) == patch_pos_embed.shape[-1]
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed, reg_pos_embed), dim=1)

    def forward_features(self, x, n=12):
        """
        Compute class-token and patch-token embeddings along with attention maps.

        Returns
        -------
        x_cls : torch.Tensor
            Class token embeddings of shape `(B, num_classes, C)`.
        x_patch : torch.Tensor
            Patch token embeddings of shape `(B, num_patches, C)`.
        attn_weights : list[torch.Tensor]
            List of attention maps from all transformer blocks.
        """
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        register_token = self.register_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x, register_token), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)
        x = self.pos_drop(x)
        attn_weights = []

        for i, blk in enumerate(self.blocks):
            x, weights_i = blk(x)
            attn_weights.append(weights_i)

        return x[:, 0 : self.num_classes], x[:, self.num_classes : -1], attn_weights

    def forward(self, x, return_att=False, n_layers=12, attention_type="fused"):
        w, h = x.shape[2:]
        x_cls, x_patch, attn_weights = self.forward_features(x)

        n, p, c = x_patch.shape
        if w != h:
            w0 = w // self.patch_embed.patch_size[0]
            h0 = h // self.patch_embed.patch_size[0]
            x_patch = torch.reshape(x_patch, [n, w0, h0, c])
        else:
            x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])

        x_patch = x_patch.permute([0, 3, 1, 2])

        x_patch = x_patch.contiguous()

        x_patch = self.head(x_patch)

        x_patch_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)

        attn_weights = torch.stack(attn_weights)  # depth * B * H * N * N
        attn_weights_tmp = attn_weights

        attn_weights = torch.mean(attn_weights, dim=2)  # depth * B * N * N

        feature_map = x_patch.detach().clone()  # B * C * H' * W'
        feature_map = F.relu(feature_map)

        n, c, h, w = feature_map.shape
        mtatt = (
            attn_weights[-n_layers:]
            .sum(0)[:, 0 : self.num_classes, self.num_classes : -1]
            .reshape([n, c, h, w])
        )

        if attention_type == "fused":
            cams = mtatt * feature_map  # B * C * H' * W'
        elif attention_type == "patchcam":
            cams = feature_map
        elif attention_type == "mct":
            cams = mtatt
        else:
            print("Error")
            cams = feature_map

        patch_attn = attn_weights[:, :, self.num_classes :, self.num_classes :]

        x_cls_logits = x_cls.mean(-1)

        if self.return_att:
            return x_cls, cams, patch_attn, attn_weights
        else:
            return x_cls, cams


class ViTWithTokenDropout(nn.Module):
    """
    Wrapper around MCTformerV2 that performs label-driven [CLS] token dropout.

    For each class token, if the corresponding class label is absent and a random
    draw is below `rate`, the class token is zeroed out (dropped).

    Parameters
    ----------
    patch_size : Patch size to use in the underlying MCTformerV2.
    prune : If True, enable Concrete-based pruning in the attention layers of MCTformerV2.
    in_chans : Number of input image channels.
    num_classes : Number of target classes.
    rate : Probability of dropping a class token when its corresponding label is 0.
    return_att : If True, return attention maps from MCTformerV2 along with predictions.
    hidden_dim : Embedding dimension used by MCTformerV2 (needed to size the classifier head).
    """

    def __init__(
        self,
        patch_size,
        prune,
        in_chans,
        num_classes,
        rate=0.5,
        return_att=False,
        hidden_dim=768,
    ):
        super().__init__()

        self.return_att = return_att
        # Initialize the Vision Transformer model
        self.mct = MCTformerV2(
            patch_size=patch_size,
            prune=prune,
            in_chans=in_chans,
            num_classes=num_classes,
            return_att=return_att,
        )

        # Store the number of classes
        self.num_classes = num_classes

        self.classifier_head_test = nn.Linear(
            hidden_dim * self.num_classes, self.num_classes
        )

        self.rate = rate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, labels=None):
        # Forward pass
        if self.return_att:
            features, mtatt, patch_attn, attn_weights = self.mct(x)
        else:
            features, mtatt = self.mct(x)

        token_features = features
        if labels is not None:
            for b in range(labels.shape[0]):
                for i in range(self.num_classes):
                    # Apply dropout to class tokens based on the absence of the class
                    if labels[b, i] == 0 and random.random() < self.rate:
                        # If the class is absent, "drop" the token by setting its feature to zeros
                        token_features[b, i, :] = torch.zeros_like(features[b, i, :])

        x_cls_logits = token_features.view(token_features.shape[0], -1)  # concat
        x_cls_logits = self.classifier_head_test(x_cls_logits)

        if self.return_att:
            return self.sigmoid(x_cls_logits), mtatt, patch_attn, attn_weights
        else:
            return self.sigmoid(x_cls_logits), mtatt
