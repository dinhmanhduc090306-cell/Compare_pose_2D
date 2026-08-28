import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os
import torch.nn.functional as F

part_indices = {
    'left_arm':  [11, 12, 13],
    'right_arm': [14, 15, 16],
    'left_leg':  [4, 5, 6],
    'right_leg': [1, 2, 3],
    'torso':     [0, 7, 8, 9, 10],
}

def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value

def mpjpe_cal(predicted, target):

    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))

def n_mpjpe(predicted, target):
    """
    Normalized MPJPE (scale only), adapted for input shape (batch_size, frame, joints, 3).
    """
    assert predicted.shape == target.shape, "Predicted and target shapes must match"

    norm_predicted = torch.mean(torch.sum(predicted ** 2, dim=-1, keepdim=True), dim=-2, keepdim=True)

    norm_target = torch.mean(torch.sum(target * predicted, dim=-1, keepdim=True), dim=-2, keepdim=True)

    scale = norm_target / norm_predicted

    return mpjpe_cal(scale * predicted, target)

def loss_velocity(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))

def depth_uncertainty_loss(pred_depth, pred_sigma, gt_depth):
    """
    计算深度不确定性损失
    :param pred_depth: (b, f, j) - 预测的深度均值 μ
    :param pred_sigma: (b, f, j) - 预测的深度标准差 σ
    :param gt_depth: (b, f, j) - 真实深度 z
    """

    pred_sigma = F.softplus(pred_sigma) + 1e-6
    loss = ((gt_depth - pred_depth) ** 2) / (2 * pred_sigma ** 2) + torch.log(pred_sigma)

    if torch.any(loss < 0):
        print("Negative loss detected. pred_sigma values:")
        print(pred_sigma)
    return loss.mean()

def nig_nll(gamma, v, alpha, beta, y):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    return nll.mean()

def nig_var(gamma, v, alpha, beta, y):
    var = torch.abs(beta / (v * (alpha - 1)))
    return var.mean()

def nig_reg(gamma, v, alpha, _beta, y):
    reg = (y - gamma).abs() * (2 * v + alpha)
    return reg.mean()

def evidential_regression(dist_params, y, lamb=1.0):
    L1 = nig_var(*dist_params, y)
    L2 = nig_reg(*dist_params, y)
    return L1 + lamb * L2

def test_calculation(predicted, target, action, error_sum, data_type, subject, valid_mask=None):
    error_sum = mpjpe_by_action_p1(predicted, target, action, error_sum, valid_mask)
    error_sum = mpjpe_by_action_p2(predicted, target, action, error_sum, valid_mask)

    return error_sum

def mpjpe_by_action_p1(predicted, target, action, action_error_sum, valid_mask=None):
    assert predicted.shape == target.shape

    num = predicted.size(0)

    dist = torch.mean(
        torch.norm(
            predicted - target,
            dim=len(target.shape) - 1
        ),
        dim=len(target.shape) - 2
    )

    dist_joints = torch.mean(
        torch.norm(
            predicted - target,
            dim=len(target.shape) - 1
        ),
        dim=len(target.shape) - 3
    )

    def resolve_name(act_name):

        # Make sure action is a normal Python string
        if isinstance(act_name, bytes):
            act_name = act_name.decode("utf-8")

        act_name = str(act_name)

        # --------------------------------------------------------
        # 1. Exact match
        # --------------------------------------------------------

        if act_name in action_error_sum:
            return act_name

        # --------------------------------------------------------
        # 2. H36M style: "Walking 1", "Eating 2", etc.
        # --------------------------------------------------------

        end_idx = act_name.find(' ')

        if end_idx != -1:

            candidate = act_name[:end_idx]

            if candidate in action_error_sum:
                return candidate

        # --------------------------------------------------------
        # 3. AP3D/custom style:
        #    progressively remove underscore components
        # --------------------------------------------------------

        parts = act_name.split('_')

        while len(parts) > 0:

            candidate = '_'.join(parts)

            if candidate in action_error_sum:
                return candidate

            parts = parts[:-1]

        # --------------------------------------------------------
        # 4. Custom dataset action
        #
        # Example:
        #     default
        #
        # If it doesn't exist, create it.
        # --------------------------------------------------------

        if act_name not in action_error_sum:

            action_error_sum[act_name] = {
                'p1': AccumLoss(),
                'p1_joints': AccumLoss(),
                'p2': AccumLoss(),
                'p2_joints': AccumLoss()
            }

        return act_name


    # ============================================================
    # SINGLE ACTION IN BATCH
    # ============================================================

    if len(set(list(action))) == 1:

        action_name = resolve_name(action[0])

        if valid_mask is not None:

            valid_num = sum(valid_mask)

            if valid_num > 0:

                dist_masked = dist[valid_mask]

                dist_joints_masked = dist_joints[valid_mask]

                action_error_sum[action_name]['p1'].update(
                    torch.mean(dist_masked).item() * valid_num,
                    valid_num
                )

                action_error_sum[action_name]['p1_joints'].update(
                    torch.mean(
                        dist_joints_masked,
                        dim=0
                    ).cpu().numpy() * valid_num,
                    valid_num
                )

        else:

            action_error_sum[action_name]['p1'].update(
                torch.mean(dist).item() * num,
                num
            )

            action_error_sum[action_name]['p1_joints'].update(
                torch.mean(
                    dist_joints,
                    dim=0
                ).cpu().numpy() * num,
                num
            )


    # ============================================================
    # MULTIPLE ACTIONS IN BATCH
    # ============================================================

    else:

        for i in range(num):

            if valid_mask is not None and not valid_mask[i]:
                continue

            action_name = resolve_name(action[i])

            action_error_sum[action_name]['p1'].update(
                dist[i].item(),
                1
            )

            action_error_sum[action_name]['p1_joints'].update(
                dist_joints[i].cpu().numpy(),
                1
            )

    return action_error_sum
    
def mpjpe_by_action_p2(predicted, target, action, action_error_sum, valid_mask=None):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    pred = predicted.detach().cpu().numpy().reshape(-1, predicted.shape[-2], predicted.shape[-1])
    gt = target.detach().cpu().numpy().reshape(-1, target.shape[-2], target.shape[-1])
    dist = p_mpjpe(pred, gt)

    def resolve_name(act_name):
        # 1. Try space-based splitting (H3.6M style)
        end_idx = act_name.find(' ')
        if end_idx != -1:
            candidate = act_name[:end_idx]
            if candidate in action_error_sum:
                return candidate
        # 2. Try progressive underscore splitting (AP3D style)
        if act_name not in action_error_sum:
            parts = act_name.split('_')
            while len(parts) > 0:
                candidate = '_'.join(parts)
                if candidate in action_error_sum:
                    return candidate
                parts = parts[:-1]
        return act_name

    if len(set(list(action))) == 1:
        action_name = resolve_name(action[0])
        if valid_mask is not None:
            valid_num = sum(valid_mask)
            if valid_num > 0:
                dist_masked = dist[valid_mask]
                action_error_sum[action_name]['p2'].update(np.mean(dist_masked) * valid_num, valid_num)
        else:
            action_error_sum[action_name]['p2'].update(np.mean(dist) * num, num)
    else:
        for i in range(num):
            if valid_mask is not None and not valid_mask[i]:
                continue
            action_name = resolve_name(action[i])
            action_error_sum[action_name]['p2'].update(dist[i], 1)

    return action_error_sum

def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape

    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)

    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0 ** 2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0 ** 2, axis=(1, 2), keepdims=True))

    normX = np.where(normX == 0, 1e-10, normX)
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY
    t = muX - a * np.matmul(muY, R)

    predicted_aligned = a * np.matmul(predicted, R) + t

    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=len(target.shape) - 2)

def define_actions(action):
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all" or action == '*':
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]

def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]:
                          {'p1': AccumLoss(), 'p2': AccumLoss(), 'p1_joints': AccumLoss()}
                      for i in range(len(actions))})
    return error_sum

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var

def print_error(data_type, action_error_sum, is_train):
    mean_error_p1, mean_error_p2 = print_error_action(action_error_sum, is_train)

    return mean_error_p1, mean_error_p2

def print_error_action(action_error_sum, is_train):
    mean_error_each = {'p1': 0.0, 'p2': 0.0, 'p1_joints': np.zeros(17)}
    mean_error_all = {'p1': AccumLoss(), 'p2': AccumLoss(), 'p1_joints': AccumLoss()}

    if is_train == 0:
        print("{0:=^20} {1:=^8} {2:=^8} {3:=^8} {4:=^8} {5:=^8} {6:=^8} {7:=^8} {8:=^8}".format(
            "Action", "Shoulder", "Elbow", "Wrist", "Hip", "Knee", "Ankle", "MPJPE", "P-MPJPE"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<20} ".format(action), end="")

        mean_error_each['p1'] = action_error_sum[action]['p1'].avg
        mean_error_all['p1'].update(mean_error_each['p1'], 1)

        mean_error_each['p2'] = action_error_sum[action]['p2'].avg
        mean_error_all['p2'].update(mean_error_each['p2'], 1)

        if hasattr(action_error_sum[action]['p1_joints'].avg, '__len__'):
            mean_error_each['p1_joints'] = action_error_sum[action]['p1_joints'].avg
        else:
            mean_error_each['p1_joints'] = np.zeros(17)
            
        mean_error_all['p1_joints'].update(mean_error_each['p1_joints'], 1)

        if is_train == 0:
            joints = mean_error_each['p1_joints']
            shoulder = (joints[11] + joints[14]) / 2.0
            elbow = (joints[12] + joints[15]) / 2.0
            wrist = (joints[13] + joints[16]) / 2.0
            hip = (joints[1] + joints[4]) / 2.0
            knee = (joints[2] + joints[5]) / 2.0
            ankle = (joints[3] + joints[6]) / 2.0
            print("{0:>8.2f} {1:>8.2f} {2:>8.2f} {3:>8.2f} {4:>8.2f} {5:>8.2f} {6:>8.2f} {7:>8.2f}".format(
                shoulder, elbow, wrist, hip, knee, ankle, mean_error_each['p1'], mean_error_each['p2']))

    if is_train == 0:
        avg_joints = mean_error_all['p1_joints'].avg
        if not hasattr(avg_joints, '__len__'):
            avg_joints = np.zeros(17)
        shoulder = (avg_joints[11] + avg_joints[14]) / 2.0
        elbow = (avg_joints[12] + avg_joints[15]) / 2.0
        wrist = (avg_joints[13] + avg_joints[16]) / 2.0
        hip = (avg_joints[1] + avg_joints[4]) / 2.0
        knee = (avg_joints[2] + avg_joints[5]) / 2.0
        ankle = (avg_joints[3] + avg_joints[6]) / 2.0
        print("{0:<20} {1:>8.2f} {2:>8.2f} {3:>8.2f} {4:>8.2f} {5:>8.2f} {6:>8.2f} {7:>8.2f} {8:>8.2f}".format(
            "Average", shoulder, elbow, wrist, hip, knee, ankle, mean_error_all['p1'].avg, mean_error_all['p2'].avg))

    return mean_error_all['p1'].avg, mean_error_all['p2'].avg

def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))

    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name

def save_model_step1_best(previous_name, save_dir, epoch, data_threshold, model):
    """
    保存 Step1（joint training）最优模型，命名沿用原始规则，增加 step1 前缀。
    """
    file_path = '%s/step1_model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    if previous_name and os.path.exists(previous_name):
        os.remove(previous_name)
    torch.save(model.state_dict(), file_path)
    return file_path

def save_model_step2_best(previous_name, save_dir, epoch, data_threshold, model):
    """
    保存 Step2（meta training）最优模型，命名沿用原始规则，增加 step2 前缀。
    """
    file_path = '%s/step2_model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    if previous_name and os.path.exists(previous_name):
        os.remove(previous_name)
    torch.save(model.state_dict(), file_path)
    return file_path

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))

def define_adaptive_weight():
    train_list = ['S1', 'S5', 'S6', 'S7', 'S8']
    actions = ['Directions 1', 'Directions', 'Discussion 1', 'Discussion', 'Eating 2', 'Eating', 'Greeting 1',
               'Greeting', 'Phoning 1', 'Phoning', 'Posing 1', 'Posing', 'Purchases 1', 'Purchases', 'Sitting 1',
               'Sitting 2', 'SittingDown 2', 'SittingDown', 'Smoking 1', 'Smoking', 'Photo 1', 'Photo', 'Waiting 1',
               'Waiting', 'Walking 1', 'Walking', 'WalkDog 1', 'WalkDog', 'WalkTogether 1', 'WalkTogether',
               'Directions 2', 'Discussion 2', 'Discussion 3', 'Eating 1', 'Greeting 2', 'Photo 2', 'Sitting',
               'SittingDown 1', 'Waiting 2', 'Posing 2', 'Waiting 3', 'Phoning 2', 'Walking 2', 'WalkTogether 2']
    act_dict = {}
    for act in actions:
        act_dict[act] = np.ones(6400)

    adaptive_weight = {}
    for sbj in train_list:
        adaptive_weight[sbj] = act_dict

    return adaptive_weight

def get_adaptive_weight(adaptive_weight, subject, action, start, end):
    N = len(subject)
    weights = torch.zeros((1, N))
    for idx in range(N):
        weights[0, idx] = sum(adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]]) / (end[idx] - start[idx])
    return weights

def fil_ex(se, min=0.1, max=0.9):
    se_num = se.shape[0]
    se_sort = np.sort(se)
    se_sort_file = se_sort[int(se_num * min):int(se_num * max)]
    mean, var = np.mean(se_sort_file), np.sqrt(np.var(se_sort_file))

    if var < 2:
        return mean, var
    else:
        return mean, torch.tensor(1e-10)

def update_adaptive_weight(adaptive_weight, subject, action, start, end, loss_batch):
    N = len(subject)
    loss_batch_for_mean_var = loss_batch.detach().cpu().numpy()
    mean, var = fil_ex(loss_batch_for_mean_var, min=0.05,
                       max=0.95)
    for idx in range(N):
        temp_weight = torch.exp(-(loss_batch[idx] - mean) * var).detach().cpu().numpy()
        adaptive_weight[subject[idx]][action[idx]][start[idx]:end[idx]] *= temp_weight

    return adaptive_weight, mean, var

def compute_body_part_loss(pred, gt):
    """
    pred, gt: [B, 17, 3] - 3D关键点
    part_indices: dict of body parts -> list of joint indices
    loss_fn: e.g., MSELoss()
    """

    loss = 0.0
    for part, idx in part_indices.items():
        pred_part = pred[:, idx, :]
        gt_part = gt[:, idx, :]
        loss += mpjpe_cal(pred_part, gt_part)
    return loss
