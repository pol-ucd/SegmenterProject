def compute_iou(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_dice(pred, target, smooth=1e-6):
    pred = pred.flatten()
    target = target.flatten()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def compute_metrics(pred_masks, masks):
    pred_binary = (pred_masks > 0.5).float()

    iou = compute_iou(pred_binary, masks).item()
    dice = compute_dice(pred_binary, masks).item()

    tp = (pred_binary * masks).sum()
    fp = (pred_binary * (1 - masks)).sum()
    fn = ((1 - pred_binary) * masks).sum()

    precision = (tp / (tp + fp + 1e-6)).item()
    recall = (tp / (tp + fn + 1e-6)).item()

    return {'iou': iou, 'dice': dice, 'precision': precision, 'recall': recall}
