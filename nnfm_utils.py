# Adapted from https://github.com/Kai-46/ARF-svox2

"""
BSD 2-Clause License

Copyright (c) 2021, the ARF and Plenoxels authors
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch

def match_colors_for_image_set(image_set, style_img):
    """
    image_set: [N, H, W, 3]
    style_img: [H, W, 3]
    """
    sh = image_set.shape
    image_set = image_set.view(-1, 3)
    style_img = style_img.view(-1, 3).to(image_set.device)

    mu_c = image_set.mean(0, keepdim=True)
    mu_s = style_img.mean(0, keepdim=True)

    cov_c = torch.matmul((image_set - mu_c).transpose(1, 0), image_set - mu_c) / float(image_set.size(0))
    cov_s = torch.matmul((style_img - mu_s).transpose(1, 0), style_img - mu_s) / float(style_img.size(0))

    u_c, sig_c, _ = torch.svd(cov_c)
    u_s, sig_s, _ = torch.svd(cov_s)

    u_c_i = u_c.transpose(1, 0)
    u_s_i = u_s.transpose(1, 0)

    scl_c = torch.diag(1.0 / torch.sqrt(torch.clamp(sig_c, 1e-8, 1e8)))
    scl_s = torch.diag(torch.sqrt(torch.clamp(sig_s, 1e-8, 1e8)))

    tmp_mat = u_s @ scl_s @ u_s_i @ u_c @ scl_c @ u_c_i
    tmp_vec = mu_s.view(1, 3) - mu_c.view(1, 3) @ tmp_mat.T

    image_set = image_set @ tmp_mat.T + tmp_vec.view(1, 3)
    image_set = image_set.contiguous().clamp_(0.0, 1.0).view(sh)

    color_tf = torch.eye(4).float().to(tmp_mat.device)
    color_tf[:3, :3] = tmp_mat
    color_tf[:3, 3:4] = tmp_vec.T
    return image_set, color_tf

def argmin_cos_distance(a, b, center=False, neg_s_flat=None):
    """
    a: [b, c, hw],
    b: [b, c, h2w2]

    neg_s_flat: [b,c,h3w3] for neg NNFM
    """
    with torch.no_grad():
        if center:
            a = a - a.mean(2, keepdims=True)
            b = b - b.mean(2, keepdims=True)
            if neg_s_flat is not None:
                neg_s_flat = neg_s_flat - neg_s_flat.mean(2, keepdims=True)

        a_norm = ((a * a).sum(1, keepdims=True) + 1e-8).sqrt()  # global normalize across channel
        a = a / (a_norm + 1e-8)
        b_norm = ((b * b).sum(1, keepdims=True) + 1e-8).sqrt() # global normalize across channel
        b = b / (b_norm + 1e-8)
        if neg_s_flat is not None:
            neg_s_norm = ((neg_s_flat * neg_s_flat).sum(1, keepdims=True) + 1e-8).sqrt() # global normalize across channel
            neg_s_flat = neg_s_flat / (neg_s_norm + 1e-8)
            b = torch.cat([b, neg_s_flat], dim=-1)  # [b, c, h2w2 + h3w3]

        z_best = []
        loop_batch_size = int(1e8 / b.shape[-1])
        for i in range(0, a.shape[-1], loop_batch_size): # over some dimension of generated image spatial dim
            a_batch = a[..., i : i + loop_batch_size]

            d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b) # [1, loop_batch_size, h2w2]

            z_best_batch = torch.argmin(d_mat, 2)
            z_best.append(z_best_batch)
        z_best = torch.cat(z_best, dim=-1) # [1, hw]

    return z_best

def nn_feat_replace(a, b, neg_s_feats=None):
    """
    return feature from generated image a, with NN feature replaced by 
    features from b(style)
    """
    n, c, h, w = a.size()
    n2, c, h2, w2 = b.size()

    assert (n == 1) and (n2 == 1)

    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone() # [n2, c, h2w2]
    if neg_s_feats is not None:
        n3, c, h3, w3 = neg_s_feats.size()
        neg_s_flat = neg_s_feats.view(n3, c, -1)
        neg_s_ref = neg_s_flat
        merged_ref = torch.cat([b_ref, neg_s_ref], dim=-1)  # [1, c, h2w2 + h3w3]

    z_new = []
    for i in range(n):
        if neg_s_feats is None:
            z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1]) # [1, hw]
            z_best = z_best.unsqueeze(1).repeat(1, c, 1) # [1, C, hw]
            feat = torch.gather(b_ref, 2, z_best)  # [1, C, hw]
        else:
            z_best = argmin_cos_distance(a_flat[i : i + 1], b_flat[i : i + 1], neg_s_flat=neg_s_flat[i:i+1]) # [1, hw]
            z_best = z_best.unsqueeze(1).repeat(1, c, 1) # [1, C, hw]
            feat = torch.gather(merged_ref, 2, z_best)  # [1, C, hw]
        z_new.append(feat)

    z_new = torch.cat(z_new, 0)
    z_new = z_new.view(n, c, h, w)
    return z_new

def cos_loss(a, b):
    a_norm = (a * a).sum(1, keepdims=True).sqrt()
    b_norm = (b * b).sum(1, keepdims=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    cos_d = 1.0 - cossim
    return cos_d.mean()

_nnfm_block_indices = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
_nnfm_blocks = [2,]
_nnfm_blocks.sort()
_nnfm_all_layers = []
for block in _nnfm_blocks:
    _nnfm_all_layers += _nnfm_block_indices[block]

def nnfm_block_indices():
     return _nnfm_block_indices

def nnfm_blocks():
     return _nnfm_blocks

def nnfm_all_layers():
     return _nnfm_all_layers

def calculate_nnfm_loss(
	gen_features, #outputs,
	style_features, #styles, # [1, C, H, W]
):
	ix_map = {}
	for a, b in enumerate(_nnfm_all_layers):
		ix_map[b] = a

	loss = 0.0
	for block in _nnfm_blocks:
		layers = _nnfm_block_indices[block]
		x_feats = torch.cat([gen_features[ix_map[ix]] for ix in layers], 1)
		s_feats = torch.cat([style_features[ix_map[ix]] for ix in layers], 1)

		target_feats = nn_feat_replace(x_feats, s_feats)
		loss += cos_loss(x_feats, target_feats)
			
	return loss