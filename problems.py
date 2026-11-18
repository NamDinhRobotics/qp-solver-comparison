"""
QP problem generators for legged-robot control.

Two canonical families, matching the two pipeline stages discussed in the paper:

  (A) srbd_mpc  -- convex single-rigid-body-dynamics MPC for a quadruped,
                   in both the *sparse OCP* form and the *condensed dense* form.
                   Follows the formulation of Di Carlo et al. (IROS 2018).

  (B) wbc_id    -- whole-body inverse dynamics for a quadruped in stance,
                   a small dense QP solved at high rate.

Everything is generated from an explicit, fixed random seed so the instances
are reproducible on any machine.
"""

import numpy as np
import scipy.sparse as sp

GRAV = 9.81


# --------------------------------------------------------------------------
# (A) SRBD MPC
# --------------------------------------------------------------------------
def _srbd_matrices(mass, inertia_diag, foot_pos, dt, yaw=0.0):
    """Discrete-time SRBD matrices for one knot point.

    State x = [theta(3), p(3), omega(3), v(3), g] in R^13.
    Input u = [f_1(3), ..., f_4(3)] in R^12.
    """
    nx, nu = 13, 12
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                   [np.sin(yaw),  np.cos(yaw), 0.0],
                   [0.0,          0.0,         1.0]])
    I_world = Rz @ np.diag(inertia_diag) @ Rz.T
    I_inv = np.linalg.inv(I_world)

    Ac = np.zeros((nx, nx))
    Ac[0:3, 6:9] = Rz.T          # theta_dot  = Rz^T omega
    Ac[3:6, 9:12] = np.eye(3)    # p_dot      = v
    Ac[11, 12] = 1.0             # v_z picks up gravity through the extra state

    Bc = np.zeros((nx, nu))
    for i in range(4):
        r = foot_pos[i]
        skew = np.array([[0.0, -r[2], r[1]],
                         [r[2], 0.0, -r[0]],
                         [-r[1], r[0], 0.0]])
        Bc[6:9, 3 * i:3 * i + 3] = I_inv @ skew
        Bc[9:12, 3 * i:3 * i + 3] = np.eye(3) / mass

    A = np.eye(nx) + Ac * dt
    B = Bc * dt
    return A, B


def _friction_block(mu, fz_min, fz_max):
    """Per-foot linearised friction pyramid + normal-force bounds (6 x 3)."""
    C = np.array([
        [1.0,  0.0, -mu],
        [-1.0, 0.0, -mu],
        [0.0,  1.0, -mu],
        [0.0, -1.0, -mu],
        [0.0,  0.0,  1.0],
        [0.0,  0.0, -1.0],
    ])
    h = np.array([0.0, 0.0, 0.0, 0.0, fz_max, -fz_min])
    return C, h


def srbd_mpc(horizon=10, dt=0.03, mass=12.0, mu=0.6,
             fz_min=5.0, fz_max=250.0, form="sparse", seed=0, reg=1e-6):
    """Build one SRBD MPC QP.

    form="sparse"    : decision vector [x_1..x_N, u_0..u_{N-1}], dynamics as
                       equality constraints -> large, block-banded, sparse.
    form="condensed" : decision vector [u_0..u_{N-1}], states eliminated
                       -> small, fully dense.

    Returns a dict with P, q, G, h, A, b (A,b may be None) and metadata.
    """
    rng = np.random.default_rng(seed)
    nx, nu, N = 13, 12, horizon

    # nominal stance geometry of a mid-size quadruped
    foot_pos = np.array([[0.19, 0.11, -0.30],
                         [0.19, -0.11, -0.30],
                         [-0.19, 0.11, -0.30],
                         [-0.19, -0.11, -0.30]])
    inertia = np.array([0.07, 0.26, 0.24])

    A_d, B_d = _srbd_matrices(mass, inertia, foot_pos, dt)

    x0 = np.zeros(nx)
    x0[12] = -GRAV
    x0[3:6] = np.array([0.0, 0.0, 0.30])
    x0[0:3] = rng.normal(0.0, 0.03, 3)      # small attitude error
    x0[9:12] = rng.normal(0.0, 0.05, 3)     # small velocity error

    # reference: track a constant height and a forward velocity
    xref = np.tile(x0, (N, 1))
    for k in range(N):
        xref[k, 0:3] = 0.0
        xref[k, 3:6] = np.array([0.4 * dt * (k + 1), 0.0, 0.30])
        xref[k, 9:12] = np.array([0.4, 0.0, 0.0])
        xref[k, 12] = -GRAV

    # the gravity state carries no tracking weight; `reg` keeps the Hessian
    # strictly positive definite, which several dual active-set solvers require
    q_state = np.array([20., 20., 5., 10., 10., 50., 0.5, 0.5, 0.2, 1., 1., 1., 0.])
    Qx = np.diag(q_state) + reg * np.eye(nx)
    Ru = 1e-5 * np.eye(nu)

    Cf, hf = _friction_block(mu, fz_min, fz_max)
    Cu = np.kron(np.eye(4), Cf)            # 24 x 12
    hu = np.tile(hf, 4)                    # 24

    if form == "sparse":
        nz = N * nx + N * nu
        P = sp.block_diag([sp.kron(sp.eye(N), sp.csc_matrix(Qx)),
                           sp.kron(sp.eye(N), sp.csc_matrix(Ru))], format="csc")
        qv = np.concatenate([(-Qx @ xref.reshape(-1, nx).T).T.reshape(-1),
                             np.zeros(N * nu)])

        # dynamics:  x_{k+1} - A x_k - B u_k = 0  (x_0 is data)
        rows = []
        rhs = []
        for k in range(N):
            row = sp.lil_matrix((nx, nz))
            row[:, k * nx:(k + 1) * nx] = sp.eye(nx)
            if k > 0:
                row[:, (k - 1) * nx:k * nx] = -A_d
            row[:, N * nx + k * nu:N * nx + (k + 1) * nu] = -B_d
            rows.append(row)
            rhs.append(A_d @ x0 if k == 0 else np.zeros(nx))
        Aeq = sp.vstack(rows, format="csc")
        beq = np.concatenate(rhs)

        Gin = sp.hstack([sp.csc_matrix((N * 24, N * nx)),
                         sp.kron(sp.eye(N), sp.csc_matrix(Cu))], format="csc")
        hin = np.tile(hu, N)
        meta = dict(name=f"srbd_mpc_sparse_N{N}", n=nz, n_eq=Aeq.shape[0],
                    n_in=Gin.shape[0], structure="sparse OCP", reg=reg)
        return dict(P=P, q=qv, G=Gin, h=hin, A=Aeq, b=beq, meta=meta)

    elif form == "condensed":
        # x = Sx x0 + Su u    with u = [u_0..u_{N-1}]
        Sx = np.zeros((N * nx, nx))
        Su = np.zeros((N * nx, N * nu))
        Apow = np.eye(nx)
        for k in range(N):
            Apow = Apow @ A_d if k > 0 else A_d.copy()
            Sx[k * nx:(k + 1) * nx, :] = Apow
            for j in range(k + 1):
                blk = np.linalg.matrix_power(A_d, k - j) @ B_d
                Su[k * nx:(k + 1) * nx, j * nu:(j + 1) * nu] = blk

        Qbar = np.kron(np.eye(N), Qx)
        Rbar = np.kron(np.eye(N), Ru)
        P = Su.T @ Qbar @ Su + Rbar
        P = 0.5 * (P + P.T)
        err = Sx @ x0 - xref.reshape(-1)
        qv = Su.T @ Qbar @ err

        Gin = np.kron(np.eye(N), Cu)
        hin = np.tile(hu, N)
        meta = dict(name=f"srbd_mpc_condensed_N{N}", n=N * nu, n_eq=0,
                    n_in=Gin.shape[0], structure="dense condensed", reg=reg)
        return dict(P=sp.csc_matrix(P), q=qv, G=sp.csc_matrix(Gin), h=hin,
                    A=None, b=None, meta=meta,
                    P_dense=P, G_dense=Gin)
    else:
        raise ValueError(form)


# --------------------------------------------------------------------------
# (B) Whole-body inverse dynamics
# --------------------------------------------------------------------------
def wbc_id(n_joints=12, n_contacts=4, mu=0.6, fz_min=5.0, fz_max=250.0,
           seed=0, infeasible=False):
    """Quadruped whole-body inverse-dynamics QP in stance.

    Decision vector z = [qddot (6 + n_joints), f (3 * n_contacts)].
    Equality : floating-base dynamics (6 rows) + contact acceleration (3*nc).
    Inequality: friction pyramid + normal-force bounds, joint-torque limits.
    Cost     : track a desired base acceleration + regularise forces.
    """
    rng = np.random.default_rng(seed)
    nv = 6 + n_joints
    nf = 3 * n_contacts
    n = nv + nf

    # a plausible, well-conditioned mass matrix
    L = rng.normal(0.0, 1.0, (nv, nv))
    M = L @ L.T + nv * np.eye(nv)
    nle = rng.normal(0.0, 5.0, nv)

    Jc = rng.normal(0.0, 1.0, (nf, nv))        # stacked contact Jacobian
    dJdq = rng.normal(0.0, 0.5, nf)

    # --- cost: || S qddot - a_des ||^2_W  +  eps ||f||^2
    a_des = rng.normal(0.0, 1.0, nv)
    W = np.diag(np.concatenate([10.0 * np.ones(6), np.ones(n_joints)]))
    P = np.zeros((n, n))
    P[:nv, :nv] = W
    P[nv:, nv:] = 1e-4 * np.eye(nf)
    q = np.zeros(n)
    q[:nv] = -W @ a_des

    # --- equality: floating-base rows of  M qddot + h = S^T tau + Jc^T f
    #     the 6 unactuated rows contain no tau
    Aeq = np.zeros((6 + nf, n))
    Aeq[:6, :nv] = M[:6, :]
    Aeq[:6, nv:] = -Jc[:, :6].T
    beq = np.concatenate([-nle[:6], -dJdq])
    Aeq[6:, :nv] = Jc                     # Jc qddot + dJdq = 0  (no contact slip)

    # --- inequality: friction pyramid per foot
    Cf, hf = _friction_block(mu, fz_min, fz_max)
    Gin = np.zeros((6 * n_contacts + 2 * n_joints, n))
    hin = np.zeros(6 * n_contacts + 2 * n_joints)
    for i in range(n_contacts):
        Gin[6 * i:6 * (i + 1), nv + 3 * i:nv + 3 * (i + 1)] = Cf
        hin[6 * i:6 * (i + 1)] = hf

    # --- torque limits, written on the actuated rows
    tau_max = 33.5
    r0 = 6 * n_contacts
    Tact = np.zeros((n_joints, n))
    Tact[:, :nv] = M[6:, :]
    Tact[:, nv:] = -Jc[:, 6:].T
    Gin[r0:r0 + n_joints, :] = Tact
    hin[r0:r0 + n_joints] = tau_max - nle[6:]
    Gin[r0 + n_joints:, :] = -Tact
    hin[r0 + n_joints:] = tau_max + nle[6:]

    if infeasible:
        # demand a normal force both above fz_max and below a larger bound
        extra = np.zeros((1, n))
        extra[0, nv + 2] = -1.0            # -f_z <= -(fz_max + 50)  => f_z >= fz_max+50
        Gin = np.vstack([Gin, extra])
        hin = np.concatenate([hin, [-(fz_max + 50.0)]])

    meta = dict(name="wbc_id" + ("_infeasible" if infeasible else ""),
                n=n, n_eq=Aeq.shape[0], n_in=Gin.shape[0],
                structure="dense WBC")
    return dict(P=sp.csc_matrix(P), q=q, G=sp.csc_matrix(Gin), h=hin,
                A=sp.csc_matrix(Aeq), b=beq, meta=meta,
                P_dense=P, G_dense=Gin, A_dense=Aeq)
