"""Action memory, learned-query readout, and action head. Steps 2-4 of DECODER_BUILD.md.

Fatma hoca's design: a TRAINABLE copy of Track-On's transformer block, operating on robot
points, with one attention added -- into the scene memory.

    q = robot tokens at frame t                 (B, N_rob, D)
      -> self-attention across robot points     the gripper as one coherent body
      -> memory attention over the ACTION fifo  its own recent history, per point
      -> cross-attention into the SCENE memory  what the objects are doing        [NEW]
      -> the output is written into the action fifo

WHAT IS COPIED, AND WHAT IS NOT

`MHA_Block` is imported from Track-On rather than reimplemented, so the attention, the MLP
ratio and the norm placement are identical by construction instead of by inspection.

Track-On's block opens with `feature_attention`, cross-attending over the whole frame
feature map, because it does not know where the point is and must search for it. We do
know: FK gives the robot points' pixel positions exactly. Direct sampling replaced that
search upstream, so the replica has no feature attention and needs no frame features at
training time -- which is what makes the cached token stream sufficient and the 5.3 PB of
feature maps unnecessary (ACTION_DECODER.md section 4).

ORDERING

Memory attention comes BEFORE the scene cross-attention. Cross-attention is a
query-formation step: a token that already knows its own trajectory ("descending, fingers
closing") asks the scene a sharper question than one that only knows the current frame.

The consequential part is not the ordering itself but what gets WRITTEN to the FIFO. With
this order the stored token is scene-aware, so the next frame's memory attention sees
history that already contains object information, and that compounds over 24 frames. The
other order keeps the action memory robot-pure. It is a two-line ablation; this is the
default, not a settled fact.

CAUSALITY

The FIFO holds only past frames and is rolled after the output is produced, so frame t
never sees t+1. `checks/46_verify_replica.py` asserts this by perturbation rather than by
reading the code.
"""
from __future__ import annotations

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .paths import TRACK_ON
if str(TRACK_ON) not in sys.path:
    sys.path.insert(0, str(TRACK_ON))

from model.modules import MHA_Block  # noqa: E402  -- upstream, imported not copied

D_DEFAULT = 256


class ReplicaLayer(nn.Module):
    """One layer: points -> own history -> scene."""

    def __init__(self, D: int = D_DEFAULT, heads: int = 4):
        super().__init__()
        self.point_attention = MHA_Block(D, heads)    # across robot points
        self.memory_attention = MHA_Block(D, heads)   # per-point, over its own FIFO
        self.scene_attention = MHA_Block(D, heads)    # into the scene memory  [NEW]

    def forward(self, q, mem, mem_mask, scene_kv, scene_mask, point_mask, t_emb):
        """
        q          (B, N, D)        robot tokens at this frame
        mem        (B, N, M, D)     the action FIFO, oldest -> newest
        mem_mask   (B, N, M)        True = ignore (unfilled slot)
        scene_kv   (B, S, D)        scene memory, flattened over slots x points
        scene_mask (B, S)           True = ignore
        point_mask (B, N)           True = ignore (padded robot slot)
        t_emb      (1, M+1, D)      slot embedding, shared with the query position
        """
        B, N, Dm = q.shape
        M = mem.shape[2]

        # 1. across points -- the gripper as one body, not N independent trackers
        q = self.point_attention(q, q, q, mask=point_mask)

        # 2. per-point memory. Batch is B*N so each point attends only to ITS OWN history,
        #    exactly as Track-On does (`q_t.view(N, 1, D)`, trackon.py:502). Cross-point
        #    mixing already happened in step 1; doing it again here would blur identities.
        qkv = torch.cat([mem.reshape(B * N, M, Dm),
                         q.reshape(B * N, 1, Dm)], dim=1)          # (B*N, M+1, D)
        mask = torch.cat([mem_mask.reshape(B * N, M),
                          torch.zeros(B * N, 1, dtype=torch.bool, device=q.device)], 1)
        # Only the LAST slot's output is used, and everything after the attention in
        # MHA_Block -- residual, both LayerNorms, the 4x MLP -- is position-wise. So
        # passing one query position is bit-identical to passing all M+1 and discarding M
        # of them, at 1/25th the FFN and query-side attention cost. Measured: 17.6 ->
        # a fraction of that per batch, which is the difference between a trainable model
        # and a 7-hour epoch.
        #
        # Keys carry t_embedding, values do not -- Track-On's convention (trackon.py:505),
        # kept so the two blocks stay comparable.
        qk = qkv + t_emb
        out = self.memory_attention(qk[:, -1:], qk, qkv, mask=mask)  # (B*N, 1, D)
        q = out.reshape(B, N, Dm)

        # 3. into the scene
        q = self.scene_attention(q, scene_kv, scene_kv, mask=scene_mask)
        return q


class ActionMemory(nn.Module):
    """The replica stack plus its FIFO, rolled forward over a window.

    Training is RECURRENT on this side: the block writes its own output into its own
    memory, so frames cannot be processed in parallel. Track-On has the same property and
    trains on 48-frame windows with M=24 (`config/train.yaml`); we follow that.
    """

    def __init__(self, D: int = D_DEFAULT, heads: int = 4, layers: int = 3,
                 M: int = 24, scene_slots: int = 24, n_obj: int = 100,
                 checkpoint: bool = True):
        super().__init__()
        self.D, self.M, self.scene_slots = D, M, scene_slots
        # Gradient checkpointing over the frame loop. The rollout is 71 sequential steps,
        # and the per-point memory attention batches B*N sequences through MHA_Block's 4x
        # MLP over all M+1 slots -- of which only the last is used. Storing activations for
        # every frame OOMs a 16 GB card at B=8. Recomputing them in the backward pass costs
        # roughly 30% more compute for a large memory saving, which is the usual trade for
        # a recurrent rollout this long.
        self.checkpoint = checkpoint
        self.layers = nn.ModuleList([ReplicaLayer(D, heads) for _ in range(layers)])
        # slot embedding over the M memory positions plus the current query, mirroring
        # Track-On's `t_embedding` (trackon.py:56). Zero-init so it starts as a no-op.
        self.t_embedding = nn.Parameter(torch.zeros(1, M + 1, D))
        # the scene keys carry their own temporal position, otherwise a token from 24
        # frames ago is indistinguishable from the current one
        self.scene_slot_emb = nn.Parameter(torch.zeros(1, scene_slots, 1, D))

    def _scene_kv(self, scene_win):
        """(B, S_slots, n_obj, D) -> flattened keys (B, S_slots*n_obj, D)."""
        B, S, No, Dm = scene_win.shape
        kv = scene_win + self.scene_slot_emb[:, -S:]
        return kv.reshape(B, S * No, Dm)

    def forward(self, robot, scene, hist_valid, point_mask, T_win: int):
        """
        robot      (B, L, N, D)   L = (M-1) + T_win
        scene      (B, L, n_obj, D)
        hist_valid (B, L)         False where the stream was front-padded
        point_mask (B, N)         True = padded robot slot (ignore)
        -> (B, T_win, N, D) output tokens, one per supervised frame
        """
        B, L, N, Dm = robot.shape
        M = self.M
        dev = robot.device

        mem = torch.zeros(B, N, M, Dm, device=dev, dtype=robot.dtype)
        mem_mask = torch.ones(B, N, M, dtype=torch.bool, device=dev)  # all unfilled
        outs = []

        # Roll over the WHOLE stream, not just the supervised part. The first M-1 frames
        # are a warm-up: the block runs and fills its FIFO, but nothing is collected.
        #
        # An earlier version started at the first supervised frame with an empty memory,
        # which meant the robot history the dataloader already loads was never read on the
        # action side (it was used only for the scene). `46_verify_replica.py` caught it:
        # scrambling the past changed nothing, so the memory was not live. Warming up is
        # strictly better -- every supervised frame then has a FULL memory, and the
        # warm-up loss masking that DECODER_BUILD.md step 5 anticipated is no longer
        # needed for the action side.
        for t in range(L):
            q = robot[:, t]                     # (B, N, D)

            lo = max(0, t - self.scene_slots + 1)
            sc = scene[:, lo:t + 1]             # (B, <=slots, n_obj, D) -- past only
            sv = hist_valid[:, lo:t + 1]        # (B, <=slots)
            kv = self._scene_kv(sc)
            S = sc.shape[1]
            sm = (~sv).unsqueeze(-1).expand(B, S, sc.shape[2]).reshape(B, -1)
            # a padded robot slot is never a valid key anywhere
            sm = sm | point_mask.new_zeros(B, kv.shape[1])

            # Front-padding is not an observation. Do not run it through attention:
            # when every scene key is masked, upstream MHA_Block zeros the attention
            # OUTPUT but its backward can still produce NaNs (torch 2.4.1 on CPU).
            # Process only live examples; their current scene frame guarantees at
            # least one valid scene key. Invalid FIFO writes remain masked below.
            live = hist_valid[:, t]
            all_live = bool(live.all())
            if all_live:
                layer_args = (mem, mem_mask, kv, sm, point_mask, self.t_embedding)
            else:
                q = q[live]
                layer_args = (mem[live], mem_mask[live], kv[live], sm[live],
                              point_mask[live], self.t_embedding)
            if q.shape[0] > 0:
                for layer in self.layers:
                    if self.checkpoint and self.training:
                        q = torch.utils.checkpoint.checkpoint(
                            layer, q, *layer_args, use_reentrant=False)
                    else:
                        q = layer(q, *layer_args)
            if not all_live:
                full_q = robot.new_zeros(B, N, Dm)
                full_q[live] = q
                q = full_q

            if t >= M - 1:
                outs.append(q)

            # Roll AFTER producing the output, so frame t never sees t+1. A slot is valid
            # only if the point is real AND the stream frame was not front-padding.
            write_mask = point_mask | (~hist_valid[:, t]).unsqueeze(1)   # (B, N)
            mem = torch.cat([mem[:, :, 1:], q.unsqueeze(2)], dim=2)
            mem_mask = torch.cat([mem_mask[:, :, 1:],
                                  write_mask.unsqueeze(-1)], dim=2)

        return torch.stack(outs, dim=1)         # (B, T_win, N, D)


class ActionReadout(nn.Module):
    """Reduce robot points to k action tokens independently at each frame.

    Learned queries cross-attend to ActionMemory's output using Track-On's attention
    block (attention + residual, LayerNorm, FFN + residual, LayerNorm). No point-index
    embeddings are added, so reordering the points and their mask leaves the readout
    unchanged in eval mode, up to floating-point reduction order.

    Start with k=1. Larger k provides distinct query slots, but future-action semantics
    still require the corresponding targets and head; this module only emits features.
    """

    def __init__(self, D: int = D_DEFAULT, heads: int = 4, k: int = 1):
        super().__init__()
        if k < 1:
            raise ValueError("k must be at least 1")
        self.D, self.k = D, k
        self.action_queries = nn.Embedding(k, D)
        self.attention = MHA_Block(D, heads)

    def forward(self, robot_out: torch.Tensor, robot_pad: torch.Tensor) -> torch.Tensor:
        """
        robot_out (B, T_win, N_rob, D)   output of ActionMemory
        robot_pad (B, N_rob) bool        True = padding, ignored as an attention key
        returns   (B, T_win, k, D)       features for the future action head

        PolicyWindows currently calls its REAL-point mask `robot_pad`: invert that
        loader mask once before passing it to both ActionMemory and this readout.
        Each example must contain at least one real robot point.
        """
        if robot_out.ndim != 4 or robot_out.shape[-1] != self.D:
            raise ValueError(f"robot_out must have shape (B, T_win, N_rob, {self.D})")
        B, T, N, Dm = robot_out.shape
        if robot_pad.shape != (B, N) or robot_pad.dtype != torch.bool:
            raise ValueError("robot_pad must be boolean with shape (B, N_rob)")
        if N == 0 or robot_pad.all(dim=1).any():
            raise ValueError("each example must have at least one real robot point")

        # Mask before projection too: even NaNs in unused slots must not contaminate
        # keys/values or their gradients. Key masking also removes their softmax mass.
        kv = robot_out.masked_fill(robot_pad[:, None, :, None], 0).reshape(B * T, N, Dm)
        mask = robot_pad[:, None, :].expand(B, T, N).reshape(B * T, N)
        queries = self.action_queries.weight.unsqueeze(0).expand(B * T, self.k, Dm)
        out = self.attention(queries, kv, kv, mask=mask)
        return out.reshape(B, T, self.k, Dm)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Decode the first two COLUMNS of R, matching actions.rot_to_6d.

    Uses differentiable Gram-Schmidt for nondegenerate inputs. Half/bfloat16 inputs
    are decoded in float32; float64 stays float64. At a zero first vector use +x;
    for a collinear second vector choose the coordinate axis least aligned with the
    first. These exceptional fallbacks produce proper rotations but are not smooth
    at the degeneracy boundary. The head initializes well away from that boundary.
    """
    if d6.ndim < 1 or d6.shape[-1] != 6 or not d6.is_floating_point():
        raise ValueError("d6 must be a floating tensor with final dimension 6")
    if d6.dtype in (torch.float16, torch.bfloat16):
        d6 = d6.float()
    eps = 1e-6
    a1, a2 = d6[..., :3], d6[..., 3:]
    n1 = torch.linalg.vector_norm(a1, dim=-1, keepdim=True)
    x_axis = torch.zeros_like(a1)
    x_axis[..., 0] = 1
    b1 = torch.where(n1 > eps, a1 / n1.clamp_min(eps), x_axis)
    a2_orth = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    n2 = torch.linalg.vector_norm(a2_orth, dim=-1, keepdim=True)
    axis = F.one_hot(b1.abs().argmin(dim=-1), num_classes=3).to(d6.dtype)
    fallback = axis - (b1 * axis).sum(-1, keepdim=True) * b1
    b2 = F.normalize(torch.where(n2 > eps, a2_orth, fallback), dim=-1, eps=eps)
    # Re-orthogonalize after subtraction, which can lose precision when a1/a2 are
    # nearly collinear. This preserves the Gram-Schmidt orientation in regular cases.
    b3 = F.normalize(torch.linalg.cross(b1, b2, dim=-1), dim=-1, eps=eps)
    b2 = torch.linalg.cross(b3, b1, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


class ActionHead(nn.Module):
    """Map each action token to body-frame delta pose and next-state gripper logit.

    Three linear projections share the readout features. Translation is emitted in
    metres, rotation in unstandardized 6D, and gripper as an unrestricted logit (no
    sigmoid in training). Shared group scales belong to the loss, not this head.

    Small nonzero weights keep initial motion near zero/identity while allowing the
    very first backward pass to reach the readout and action memory. With unit-scale
    features the initial position standard deviation is about 1 mm per coordinate.
    """

    def __init__(self, D: int = D_DEFAULT):
        super().__init__()
        if D < 1:
            raise ValueError("D must be positive")
        self.D = D
        self.position = nn.Linear(D, 3)
        self.rotation = nn.Linear(D, 6)
        self.gripper = nn.Linear(D, 1)
        for layer, output_std in ((self.position, 1e-3), (self.rotation, 1e-3),
                                  (self.gripper, 1e-2)):
            nn.init.normal_(layer.weight, std=output_std / D**0.5)
            nn.init.zeros_(layer.bias)
        with torch.no_grad():
            self.rotation.bias.copy_(self.rotation.bias.new_tensor([1, 0, 0, 0, 1, 0]))

    def forward(self, action_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        in:  action_tokens (B, T_win, k, D), from ActionReadout
        out: delta_pos_body    (B, T_win, k, 3), metres
             delta_rot6d_body  (B, T_win, k, 6), first two rotation columns
             delta_rotmat_body (B, T_win, k, 3, 3), decoded proper rotation
             gripper_logit     (B, T_win, k, 1), next observed open-state logit

        Keep the k dimension even for k=1. Apply sigmoid to gripper_logit only when
        probabilities are needed; it is not a recorded control-command prediction.
        """
        if action_tokens.ndim != 4 or action_tokens.shape[-1] != self.D:
            raise ValueError(f"action_tokens must have shape (B, T_win, k, {self.D})")
        d6 = self.rotation(action_tokens)
        return {
            "delta_pos_body": self.position(action_tokens),
            "delta_rot6d_body": d6,
            "delta_rotmat_body": rotation_6d_to_matrix(d6),
            "gripper_logit": self.gripper(action_tokens),
        }
