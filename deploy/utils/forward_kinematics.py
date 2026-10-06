##
#
# Forward kinematics helpers for the G1 robot in MuJoCo.
#
# After mj_step (or an explicit mj_kinematics call) every body/site xpos and
# xmat are already up-to-date in MjData — these helpers just read those
# values and express them in convenient reference frames.
#
# NOTE: call mujoco.mj_kinematics(model, data) before using these during
# initialisation (before the first mj_step).  During the control loop
# mj_step keeps everything current automatically.
#
##

import numpy as np

# ---------------------------------------------------------------------------
# Named sites / bodies used as end-effectors
# ---------------------------------------------------------------------------

#: MuJoCo site names defined on the wrist-yaw links (see g1_27dof_deploy_tennis.xml)
SITE_LEFT_PALM  = "left_palm"
SITE_RIGHT_PALM = "right_palm"

#: Most-distal body on each arm (parent of the palm sites)
BODY_LEFT_WRIST  = "left_wrist_yaw_link"
BODY_RIGHT_WRIST = "right_wrist_yaw_link"


# ---------------------------------------------------------------------------
# World-frame accessors
# ---------------------------------------------------------------------------

def left_palm_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the left palm site.  Shape (3,), float32."""
    return mj_data.site(SITE_LEFT_PALM).xpos.astype(np.float32).copy()


def right_palm_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the right palm site.  Shape (3,), float32."""
    return mj_data.site(SITE_RIGHT_PALM).xpos.astype(np.float32).copy()


def left_wrist_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the left wrist-yaw body origin.  Shape (3,), float32."""
    return mj_data.body(BODY_LEFT_WRIST).xpos.astype(np.float32).copy()


def right_wrist_pos_world(mj_data) -> np.ndarray:
    """World-frame position of the right wrist-yaw body origin.  Shape (3,), float32."""
    return mj_data.body(BODY_RIGHT_WRIST).xpos.astype(np.float32).copy()


# ---------------------------------------------------------------------------
# Frame-relative accessors
# ---------------------------------------------------------------------------

def pos_in_frame(
    pos_world: np.ndarray,
    R_frame: np.ndarray,
    p_frame: np.ndarray,
) -> np.ndarray:
    """
    Express a world-frame position in a given reference frame.

    Parameters
    ----------
    pos_world : (3,) world-frame position
    R_frame   : (3, 3) rotation matrix of the reference frame (columns = frame axes)
    p_frame   : (3,) world-frame origin of the reference frame

    Returns
    -------
    (3,) position expressed in the reference frame, float32
    """
    return (R_frame.T @ (pos_world - p_frame)).astype(np.float32)


def left_palm_in_frame(
    mj_data,
    R_frame: np.ndarray,
    p_frame: np.ndarray,
) -> np.ndarray:
    """Left palm position expressed in the given reference frame.  Shape (3,)."""
    return pos_in_frame(left_palm_pos_world(mj_data), R_frame, p_frame)


def right_palm_in_frame(
    mj_data,
    R_frame: np.ndarray,
    p_frame: np.ndarray,
) -> np.ndarray:
    """Right palm position expressed in the given reference frame.  Shape (3,)."""
    return pos_in_frame(right_palm_pos_world(mj_data), R_frame, p_frame)


# ---------------------------------------------------------------------------
# Relative hand geometry
# ---------------------------------------------------------------------------

def left_wrt_right_world(mj_data) -> np.ndarray:
    """
    Vector from the right palm to the left palm in the world frame.

    This captures the relative bimanual configuration independent of
    pelvis pose, and is useful as a goal when expressed in a body frame.

    Returns
    -------
    (3,) float32
    """
    return left_palm_pos_world(mj_data) - right_palm_pos_world(mj_data)


def left_wrt_right_in_frame(
    mj_data,
    R_frame: np.ndarray,
) -> np.ndarray:
    """
    Vector from the right palm to the left palm, expressed in the given frame.

    Only the orientation of the reference frame matters (not its origin), so
    only R_frame is required.

    Parameters
    ----------
    R_frame : (3, 3) rotation matrix of the reference frame

    Returns
    -------
    (3,) float32
    """
    return (R_frame.T @ left_wrt_right_world(mj_data)).astype(np.float32)


# ---------------------------------------------------------------------------
# Pelvis-frame convenience wrappers
# ---------------------------------------------------------------------------

def pelvis_frame(mj_data):
    """
    Return (R_pelvis, p_pelvis) — the current pelvis rotation matrix and
    world-frame origin — ready to pass to the frame helpers above.

    Uses the ``pelvis`` body which is always present in the G1 models.
    """
    from utils.math_utils import quat_to_rotation_matrix  # local import to avoid circular deps
    p = mj_data.body("pelvis").xpos.astype(np.float32).copy()
    R = quat_to_rotation_matrix(mj_data.body("pelvis").xquat.astype(np.float32))
    return R, p


def left_palm_in_pelvis(mj_data) -> np.ndarray:
    """Left palm position in the pelvis frame.  Shape (3,)."""
    R, p = pelvis_frame(mj_data)
    return left_palm_in_frame(mj_data, R, p)


def right_palm_in_pelvis(mj_data) -> np.ndarray:
    """Right palm position in the pelvis frame.  Shape (3,)."""
    R, p = pelvis_frame(mj_data)
    return right_palm_in_frame(mj_data, R, p)


def left_wrt_right_in_pelvis(mj_data) -> np.ndarray:
    """
    Vector from the right palm to the left palm, expressed in the pelvis frame.
    Shape (3,).
    """
    R, _ = pelvis_frame(mj_data)
    return left_wrt_right_in_frame(mj_data, R)


# ---------------------------------------------------------------------------
# Pickup-task helper  (both goals in one call)
# ---------------------------------------------------------------------------

def pickup_fk_goals(
    mj_data,
    R_pelvis: np.ndarray,
    p_pelvis: np.ndarray,
    otherhand_offset: np.ndarray,
) -> np.ndarray:
    """
    Compute the FK-derived second goal vector for the box-pickup task.

    Matches the training observation exactly (commands.py ``command`` property):

        target_position_w  =  right_palm_world  +  R_pelvis @ otherhand_offset
        observation        =  R_pelvis.T @ (target_position_w - pelvis_pos)
                           =  right_palm_in_pelvis  +  otherhand_offset

    where ``otherhand_offset`` is ``target_pos_mean`` from the motion config
    (the grip-width offset in the anchor/pelvis frame, e.g. ``[0, 0.25, 0]``).

    This is the target position for the *left* palm expressed in the pelvis
    frame — a point defined dynamically relative to the *live* right palm.

    Parameters
    ----------
    mj_data          : live MuJoCo MjData (xpos current after mj_step)
    R_pelvis         : (3, 3) current pelvis rotation matrix
    p_pelvis         : (3,)   current pelvis world-frame position
    otherhand_offset : (3,)   grip-width offset in pelvis/anchor frame;
                              read from ``pickup_otherhand_position.vector``
                              in the deploy config YAML

    Returns
    -------
    (3,) float32
        Second goal observation: right-palm position in pelvis frame plus the
        grip-width offset.  Pass directly as ``goal_targets[3:6]``.
    """
    rp_w = right_palm_pos_world(mj_data)
    right_in_pelvis = (R_pelvis.T @ (rp_w - p_pelvis)).astype(np.float32)
    return right_in_pelvis + np.asarray(otherhand_offset, dtype=np.float32)
