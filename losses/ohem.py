import torch


def ohem_loss(rate, base_crit, cls_pred, cls_target):

    batch_size = cls_pred.size(0) 
    ohem_cls_loss = base_crit(cls_pred, cls_target)
    if rate==1:
        return ohem_cls_loss.sum()
    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min((sorted_ohem_loss.size())[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss * batch_size
