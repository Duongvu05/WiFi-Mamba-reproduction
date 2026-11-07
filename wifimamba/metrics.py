"""Person-in-WiFi 3D evaluation (millimetres).

Protocol: the predicted person count decides how many slots are kept (top confidences), the Hungarian
algorithm only pairs kept slots with ground-truth people for scoring, and missed or extra people are
reported separately. All errors are in millimetres (coordinates are stored in metres).
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

# 14-keypoint skeleton, used as a soft anatomical regulariser.
# The 14 bones of the joint order of the Person-in-WiFi 3D labels (the bone lengths are rigid on the train labels, the variation of every length is under 10 percent).
# 0 head, 1 neck, 2 and 3 shoulders, 4 and 6 elbows, 8 and 10 wrists, 5 and 7 hips, 9 and 11 knees, 12 and 13 ankles, the arms 2-4-8 and 3-6-10, the legs 5-9-12 and 7-11-13.
BONES = ((0, 1), (1, 2), (1, 3), (2, 4), (4, 8), (3, 6), (6, 10), (1, 5), (1, 7), (5, 7), (5, 9), (9, 12), (7, 11), (11, 13))


def compute_similarity_transform(source, target):
    """Procrustes alignment (scale, rotation, translation) of source onto target. [J, 3]."""
    mu_s, mu_t = source.mean(axis=0, keepdims=True), target.mean(axis=0, keepdims=True)
    s0, t0 = source - mu_s, target - mu_t
    if (s0 ** 2).sum() < 1e-8:
        return source
    u, _, vt = np.linalg.svd(s0.T @ t0)
    r = u @ vt
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = u @ vt
    scale = np.linalg.norm(t0) / (np.linalg.norm(s0 @ r) + 1e-8)
    return scale * (source - mu_s) @ r + mu_t


def hungarian_pairs(gt, pred_sel):
    """gt [n,J,3], pred_sel [m,J,3] -> list of (gt_idx, pred_idx), min(n,m) pairs."""
    cost = np.linalg.norm(gt[:, None] - pred_sel[None], axis=-1).mean(axis=-1)
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def person_metrics(gt_p, pred_p):
    """Per-person metrics in mm. Returns (normalised distance per joint, (MPJPE, PA-MPJPE, h, v, d) in mm)."""
    gt_p, pred_p = gt_p * 1000.0, pred_p * 1000.0
    torso = np.linalg.norm(gt_p[6] - gt_p[4])
    dist = np.linalg.norm(pred_p - gt_p, axis=-1)
    pa = np.linalg.norm(compute_similarity_transform(pred_p, gt_p) - gt_p, axis=-1)
    mm = (dist.mean(), pa.mean(), np.abs(pred_p[:, 0] - gt_p[:, 0]).mean(), np.abs(pred_p[:, 1] - gt_p[:, 1]).mean(), np.abs(pred_p[:, 2] - gt_p[:, 2]).mean())
    return dist / torso, mm


def select_and_match(pred, logits, n_hat, y, mask):
    """Yield (n, frame, matched pairs, kept slot poses) with the e2e protocol."""
    counts = mask.sum(1).astype(int)
    for n in (1, 2, 3):
        for i in np.flatnonzero(counts == n):
            m = int(n_hat[i])
            ps = pred[i][np.argsort(-logits[i])[:m]]
            m = len(ps)  # slots actually kept
            yield n, i, (hungarian_pairs(y[i, :n], ps) if m else []), ps, m


def metrics(pred, logits, counts, y, masks):
    """Matched MPJPE (mm) with missed/extra accounting, plus root versus root-relative error."""
    result = {}
    for n in (1, 2, 3):
        rows = np.flatnonzero(masks.sum(1) == n)
        errors, translation_sq, centered_sq, joint_sq, matched, total = [], [], [], [], 0, 0
        for i in rows:
            take = int(counts[i])
            pp = pred[i, np.argsort(-logits[i])[:take]]
            gt = y[i, masks[i] > 0]
            total += take
            if take:
                a, b = linear_sum_assignment(np.linalg.norm(pp[:, None] - gt[None], axis=-1).mean(-1))
                delta = pp[a] - gt[b]
                errors.extend(np.linalg.norm(delta, axis=-1).mean(-1).tolist())
                shift = delta.mean(1)
                translation_sq.extend(np.sum(shift ** 2, axis=-1).tolist())
                centered_sq.extend(np.sum((delta - shift[:, None]) ** 2, axis=-1).mean(-1).tolist())
                joint_sq.extend(np.sum(delta ** 2, axis=-1).mean(-1).tolist())
                matched += len(a)
        result[f"n{n}"] = dict(
            mpjpe_mm=float(np.mean(errors) * 1000) if errors else None, frames=len(rows), matched=matched,
            missed=len(rows) * n - matched, extra=total - matched, recall=matched / (len(rows) * n),
            precision=matched / max(total, 1), count_accuracy=float(np.mean(counts[rows] == n)),
            centroid_rmse_mm=float(np.sqrt(np.mean(translation_sq)) * 1000) if matched else None,
            centered_pose_rmse_mm=float(np.sqrt(np.mean(centered_sq)) * 1000) if matched else None,
            centroid_squared_error_fraction=float(np.sum(translation_sq) / max(np.sum(joint_sq), 1e-12)) if matched else None)
    return result


def full_metrics(pred, logits, n_hat, y, mask):
    """MPJPE, PA-MPJPE, MPJDLE (h, v, d), PCK@30/20/10 and count/recall/precision per person count."""
    out = {}
    for n in (1, 2, 3):
        mm, dn, matched, n_pred = [], [], 0, 0
        idx = np.flatnonzero(mask.sum(1).astype(int) == n)
        for nn_, i, pairs, ps, m in select_and_match(pred, logits, n_hat, y, mask):
            if nn_ != n:
                continue
            n_pred += m
            matched += len(pairs)
            for g, p in pairs:
                d, r = person_metrics(y[i, g], ps[p])
                mm.append(r)
                dn.append(d)
        o = dict(samples=int(len(idx)))
        if not mm:  # every person missed
            out[f"n{n}"] = o | dict(mpjpe_mm=None, count_acc=round(100 * float((n_hat[idx] == n).mean()), 2), recall=0.0, precision=0.0)
            continue
        mm, dn = np.array(mm), np.concatenate(dn)
        for k, name in enumerate(("mpjpe_mm", "pa_mpjpe_mm", "mpjdle_h_mm", "mpjdle_v_mm", "mpjdle_d_mm")):
            o[name] = round(float(mm[:, k].mean()), 2)
        for t, name in ((0.3, "pck30"), (0.2, "pck20"), (0.1, "pck10")):
            o[name] = round(100 * float((dn <= t).mean()), 2)
        o["count_acc"] = round(100 * float((n_hat[idx] == n).mean()), 2)
        o["recall"] = round(100 * matched / (len(idx) * n), 2)
        o["precision"] = round(100 * matched / max(n_pred, 1), 2)
        out[f"n{n}"] = o
    return out


def greedy_match(real, pred):
    """Nearest-neighbour matching of the official Person-in-WiFi 3D code (opera/datasets/wifi_pose.py, calc_mpjpe).
    real [n,J,3], pred [m,J,3] in metres. The 50 threshold is in metres, so there is none. The distance to the
    ground truth chooses the prediction, so with many predictions this is a best-of-m selection by the labels."""
    dist = np.linalg.norm(real[:, None] - pred[None], axis=-1).mean(-1)
    corres = -np.ones(len(real), dtype=int)
    occupied = np.zeros(len(pred), dtype=bool)
    d = dist.copy()
    while d.size and d.min() < 50:
        for r, c in zip(*np.nonzero(d == d.min())):
            d[r, c] = 50
            if corres[r] >= 0 or occupied[c]:
                continue
            corres[r], occupied[c] = c, True
    keep = corres >= 0
    return real[keep], pred[corres[keep]]


def greedy_metrics(P, y, mask, keep=None):
    """Greedy protocol over the first `keep` predictions of each frame (all when None). Averages matched people only."""
    out = {}
    counts = mask.sum(1).astype(int)
    for n in (1, 2, 3):
        mm, total = [], 0
        for i in np.flatnonzero(counts == n):
            gt, pr = greedy_match(y[i, :n], P[i] if keep is None else P[i][:keep])
            total += n
            for g, p in zip(gt * 1000.0, pr * 1000.0):
                mm.append((np.linalg.norm(p - g, axis=-1).mean(), np.linalg.norm(compute_similarity_transform(p, g) - g, axis=-1).mean(),
                           np.abs(p[:, 0] - g[:, 0]).mean(), np.abs(p[:, 1] - g[:, 1]).mean(), np.abs(p[:, 2] - g[:, 2]).mean()))
        mm = np.array(mm)
        out[f"n{n}"] = dict(zip(("mpjpe_mm", "pa_mpjpe_mm", "mpjdle_h_mm", "mpjdle_v_mm", "mpjdle_d_mm"), np.round(mm.mean(0), 1).tolist()), matched=len(mm), of=total)
    return out
