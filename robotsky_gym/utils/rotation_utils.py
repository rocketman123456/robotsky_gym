import functools
from typing import Literal, Tuple, Union
import torch
from torch import Tensor
import numpy as np

_use_quat_wxyz = True


def set_quat_convention(wxyz_or_xyzw: Literal["wxyz", "xyzw"]):
    global _use_quat_wxyz
    if wxyz_or_xyzw == "wxyz":
        _use_quat_wxyz = True
    else:
        _use_quat_wxyz = False


def get_quat_convention():
    if _use_quat_wxyz:
        return "wxyz"
    else:
        return "xyzw"


def _apply_quat_convention(func):
    def wrapper(*args, **kwargs):
        if "use_quat_wxyz" not in kwargs or kwargs["use_quat_wxyz"] is None:
            kwargs["use_quat_wxyz"] = _use_quat_wxyz
        return func(*args, **kwargs)

    return wrapper


@_apply_quat_convention
@torch.jit.script
def quat_apply(a, b, use_quat_wxyz: bool = None):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    if use_quat_wxyz:
        xyz = a[:, 1:]
        w = a[:, :1]
    else:
        xyz = a[:, :3]
        w = a[:, 3:]
    t = xyz.cross(b, dim=-1) * 2
    return (b + w * t + xyz.cross(t, dim=-1)).view(shape)


@_apply_quat_convention
@torch.jit.script
def quat_rotate(q, v, use_quat_wxyz: bool = None):
    shape = q.shape
    if use_quat_wxyz:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    else:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@_apply_quat_convention
@torch.jit.script
def quat_rotate_inverse(q, v, use_quat_wxyz: bool = None):
    shape = q.shape
    if use_quat_wxyz:
        q_w = q[:, 0]
        q_vec = q[:, 1:]
    else:
        q_w = q[:, -1]
        q_vec = q[:, :3]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def copysign(a: float, b: Tensor) -> Tensor:
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@_apply_quat_convention
@torch.jit.script
def quat_to_euler_xyz(q: Tensor, use_quat_wxyz: bool = None) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Returns:
        roll, pitch, yaw
    """
    if use_quat_wxyz:
        qx, qy, qz, qw = 1, 2, 3, 0
    else:
        qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return (
        (roll + torch.pi) % (2 * torch.pi) - torch.pi,
        (pitch + torch.pi) % (2 * torch.pi) - torch.pi,
        yaw % (2 * torch.pi),
    )
    # return torch.fmod((roll + torch.pi) % (2 * torch.pi) - torch.pi, 2*np.pi), torch.fmod(pitch, 2*np.pi), torch.fmod(yaw, 2*np.pi)
    # return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@_apply_quat_convention
@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw, use_quat_wxyz: bool = None):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    if use_quat_wxyz:
        return torch.stack([qw, qx, qy, qz], dim=-1)
    else:
        return torch.stack([qx, qy, qz, qw], dim=-1)


@_apply_quat_convention
@torch.jit.script
def quat_mul(a, b, use_quat_wxyz: bool = None):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    if use_quat_wxyz:
        x1, y1, z1, w1 = a[:, 1], a[:, 2], a[:, 3], a[:, 0]
        x2, y2, z2, w2 = b[:, 1], b[:, 2], b[:, 3], b[:, 0]
    else:
        x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]

    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    if use_quat_wxyz:
        quat = torch.stack([w, y, z], dim=-1).view(shape)
    else:
        quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@_apply_quat_convention
@torch.jit.script
def quat_conjugate(a, use_quat_wxyz: bool = None):
    shape = a.shape
    a = a.reshape(-1, 4)
    if use_quat_wxyz:
        return torch.cat((a[:, :1], -a[:, 1:]), dim=-1).view(shape)
    else:
        return torch.cat((-a[:, :3], a[:, 3:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@_apply_quat_convention
@torch.jit.script
def quat_from_angle_axis(angle, axis, use_quat_wxyz: bool = None):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    if use_quat_wxyz:
        return quat_unit(torch.cat([w, xyz], dim=-1))
    else:
        return quat_unit(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@_apply_quat_convention
@torch.jit.script
def quat_from_rot_vec(rot_vec, use_quat_wxyz: bool = None):
    """
    Convert 3-D rotation vectors to quaternion.
    This function avoids the discontinuity when rot_vec = 0.
    """
    # angle = ||rot_vec||
    # theta = angle / 2
    theta = rot_vec.norm(p=2, dim=-1).unsqueeze(-1) * 0.5
    # axis = rot_vec / angle
    # xyz = axis * sin(theta) = rot_vec * sin(theta) / theta / 2
    sin_theta_over_theta = torch.special.sinc(theta / np.pi)
    xyz = rot_vec * sin_theta_over_theta / 2
    w = theta.cos()
    if use_quat_wxyz:
        return quat_unit(torch.cat([w, xyz], dim=-1))
    else:
        return quat_unit(torch.cat([xyz, w], dim=-1))


@_apply_quat_convention
@torch.jit.script
def quat_to_rot_vec(q, use_quat_wxyz: bool = None):
    """
    Convert quaternion to 3-D rotation vectors.
    This function avoids the discontinuity when rotation angle = PI.
    """
    if use_quat_wxyz:
        q_positive = torch.sign(q[..., 0]).unsqueeze(-1) * q
        xyz = q_positive[..., 1:]
        w = q_positive[..., 0]
    else:
        q_positive = torch.sign(q[..., 3]).unsqueeze(-1) * q
        xyz = q_positive[..., :3]
        w = q_positive[..., 3]
    sin_theta = xyz.norm(p=2, dim=-1)
    cos_theta = w
    theta = torch.atan2(sin_theta, cos_theta)
    # 0 <= theta <= PI / 2
    sin_theta_over_theta = torch.special.sinc(theta / np.pi)
    rot_vec = xyz / sin_theta_over_theta.unsqueeze(-1) * 2
    return rot_vec


@_apply_quat_convention
@torch.jit.script
def quat_to_matrix(q: torch.Tensor, use_quat_wxyz: bool = None) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    This function is copied from pytorch3d/transforms/rotation_conversions.py
    and modified.
    pytorch3d: https://github.com/facebookresearch/pytorch3d

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if use_quat_wxyz:
        r, i, j, k = torch.unbind(q, -1)
    else:
        i, j, k, r = torch.unbind(q, -1)
    two_s = 2.0 / (q * q).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(q.shape[:-1] + (3, 3))


@torch.jit.script
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


@_apply_quat_convention
@torch.jit.script
def matrix_to_quaternion(matrix: torch.Tensor, use_quat_wxyz: bool = None) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    This function is copied from pytorch3d/transforms/rotation_conversions.py
    and modified.
    pytorch3d: https://github.com/facebookresearch/pytorch3d

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
                1.0 + m00 + m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m10 + m01, m02 + m20, m21 - m12], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 + m01, q_abs[..., 1] ** 2, m12 + m21, m02 - m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m20 + m02, m21 + m12, q_abs[..., 2] ** 2, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, m02 - m20, m10 - m01, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[torch.nn.functional.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    q_positive = torch.sign(out[..., 3]).unsqueeze(-1) * out

    if use_quat_wxyz:
        q_positive = torch.cat((q_positive[..., 3:], q_positive[..., :3]), dim=-1)

    return q_positive


@_apply_quat_convention
@torch.jit.script
def quat_from_two_vectors(u, v, use_quat_wxyz: bool = None):
    """
    Find rotation q to rotate vector u to the direction of vector v.

    I found the algorithm on https://raw.org/proof/quaternion-from-two-vectors
    """
    d = (u * v).sum(dim=-1, keepdim=True)  # dot
    c = torch.cross(u, v, dim=-1)
    if use_quat_wxyz:
        q = torch.cat(
            (
                d + torch.sqrt(d.square() + c.square().sum(dim=-1, keepdim=True)),
                c,
            ),
            dim=-1,
        )
    else:
        q = torch.cat(
            (
                c,
                d + torch.sqrt(d.square() + c.square().sum(dim=-1, keepdim=True)),
            ),
            dim=-1,
        )
    return normalize(q)


@torch.jit.script
def matrix_to_rep6d(rot_mat):
    """
    Convert rotation matrix to a 6-D continuous representation of SO(3).

    This methods comes from:
    ZHOU Y., BARNES C., LU J., YANG J., LI H.,
    On the Continuity of Rotation Representations in Neural Networks,
    https://arxiv.org/abs/1812.07035.
    """
    return rot_mat[..., :, :2].flatten(start_dim=-2)


@torch.jit.script
def rep6d_to_matrix(rep_6d):
    """
    Convert the 6-D continuous representation of SO(3) to rotation matrix.

    This methods comes from:
    ZHOU Y., BARNES C., LU J., YANG J., LI H.,
    On the Continuity of Rotation Representations in Neural Networks,
    https://arxiv.org/abs/1812.07035.
    """
    row_1 = rep_6d[..., 0::2]
    row_2 = rep_6d[..., 1::2]
    row_1 = normalize(row_1)
    row_2 = normalize(row_2 - (row_1 * row_2).sum(dim=-1, keepdim=True) * row_1)
    row_3 = torch.cross(row_1, row_2, dim=-1)
    return torch.stack((row_1, row_2, row_3), dim=-1)


@torch.jit.script
def normalize_rep6d(rep_6d):
    """
    Normalize the 6-D continuous representation of SO(3).

    6-D continuous representation of SO(3) from:
    ZHOU Y., BARNES C., LU J., YANG J., LI H.,
    On the Continuity of Rotation Representations in Neural Networks,
    https://arxiv.org/abs/1812.07035.
    """
    row_1 = rep_6d[..., 0::2]
    row_2 = rep_6d[..., 1::2]
    row_1 = normalize(row_1)
    row_2 = normalize(row_2 - (row_1 * row_2).sum(dim=-1, keepdim=True) * row_1)
    return torch.stack((row_1, row_2), dim=-1).flatten(start_dim=-2)


@torch.jit.script
def angle_diff(angle_1: torch.Tensor, angle_2: torch.Tensor) -> torch.Tensor:
    """
    Return the difference of two scalar angles.
    The result is in range [- PI, PI]
    """
    diff = angle_1 - angle_2
    diff_sgn = torch.sign(diff)
    diff = torch.fmod(diff + np.pi * diff_sgn, 2 * np.pi) - np.pi * diff_sgn
    return diff
