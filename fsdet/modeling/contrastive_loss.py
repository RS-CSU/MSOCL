import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.layers import cat



class ContrastiveHead(nn.Module):
    """MLP head for contrastive representation learning, https://arxiv.org/abs/2003.04297
    Args:
        dim_in (int): dimension of the feature intended to be contrastively learned
        feat_dim (int): dim of the feature to calculated contrastive loss

    Return:
        feat_normalized (tensor): L-2 normalized encoded feature,
            so the cross-feature dot-product is cosine similarity (https://arxiv.org/abs/2004.11362)
    """
    def __init__(self, dim_in, feat_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, feat_dim),
            nn.BatchNorm1d(feat_dim)
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        feat = self.head(x)
        # feat_normalized = F.normalize(feat, dim=1)
        return feat

class ContrastLoss(nn.Module):
    def __init__(self, temperature=0.2, iou_thres=0.4):
        super().__init__()
        self.temperature = temperature
        self.iou_threshold = iou_thres

    def Cos_Distance(self, box_feat, crop_feat):  # negative cosine similarity
        z = crop_feat.detach()  # stop gradient
        p = F.normalize(box_feat, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
#        dis = -(p * z).sum(dim=1).mean()
        # dis = F.pairwise_distance(p, z).mean()
        cos_dis = -F.cosine_similarity(p, z, dim=1).mean()
        # return torch.exp(cos_dis)
        return cos_dis

    # # 使用简单的相似性度量
    # def forward(self,box_cls_feat_con, crop_feat_con, batch_size, ious):
    #     # z = crop_feat_con.detach()
    #     L_MS = []
    #     for crop_feat in crop_feat_con:
    #         z_expand = cat([torch.index_select(crop_feat, 0, torch.tensor(val).cuda()) for val in range(batch_size) for i in range(256)],dim=0)
    #         L = 0
    #         for b in range(batch_size):
    #             b_ious = ious[b*256: b*256 + 256]
    #             pos_index = b_ious >= 0.4
    #             neg_index = b_ious < 0.4
    #             b_z = z_expand[b*256: b*256 + 256]
    #             z_pos = b_z[pos_index]
    #             z_neg = b_z[neg_index]
    #             b_box = box_cls_feat_con[b*256: b*256 + 256]
    #             box_pos = b_box[pos_index]
    #             box_neg = b_box[neg_index]
    #             sim_pos = self.Cos_Distance(box_pos, z_pos)
    #             sim_neg = self.Cos_Distance(box_neg, z_neg)
    #             L_batch = torch.exp((sim_pos - sim_neg))
    #             # L_batch = torch.exp(sim_pos)
    #             L += L_batch
    #         L_MS.append(L)
    #         # pass
    #     Loss = min(L_MS)/batch_size
    #     # L = self.Cos_Distance(box_cls_feat_con, z_expand)
    #     return Loss
    """
    # 采用infoNCE Loss进行损失计算
    def forward(self, box_cls_feat_con, crop_feat_con, batch_size, ious):
        # z = crop_feat_con.detach()
        # print(len(crop_feat_con))
        L_MS = []
        for crop_feat in crop_feat_con:
            z_expand = cat(
                [torch.index_select(crop_feat, 0, torch.tensor(val).cuda()) for val in range(batch_size) for i in
                 range(256)], dim=0)
            # L = 0
            # b_ious = ious[b * 256: b * 256 + 256]
            pos_index = ious >= 0.4
            neg_index = ious < 0.4
            # b_z = z_expand[b * 256: b * 256 + 256]
            z_pos = z_expand[pos_index]
            z_neg = z_expand[neg_index]
            # b_box = box_cls_feat_con[b * 256: b * 256 + 256]
            box_pos = box_cls_feat_con[pos_index]
            box_neg = box_cls_feat_con[neg_index]
            sim_pos = self.Cos_Distance(box_pos, z_pos)
            sim_neg = self.Cos_Distance(box_neg, z_neg)
            # print(sim_pos.shape)
            pos = torch.exp(torch.div(sim_pos, self.temperature))
            neg = torch.exp(torch.div(sim_neg, self.temperature))
            L = -torch.log(torch.div(pos, (pos + neg)))
            # for b in range(batch_size):
            #     b_ious = ious[b * 256: b * 256 + 256]
            #     pos_index = b_ious >= 0.4
            #     neg_index = b_ious < 0.4
            #     b_z = z_expand[b * 256: b * 256 + 256]
            #     z_pos = b_z[pos_index]
            #     z_neg = b_z[neg_index]
            #     b_box = box_cls_feat_con[b * 256: b * 256 + 256]
            #     box_pos = b_box[pos_index]
            #     box_neg = b_box[neg_index]
            #     # 采用cos距离
            #     sim_pos = self.Cos_Distance(box_pos, z_pos)
            #     sim_neg = self.Cos_Distance(box_neg, z_neg)
            #     # print(sim_pos.shape)
            #     pos = torch.exp(torch.div(sim_pos, self.temperature))
            #     neg = torch.exp(torch.div(sim_neg, self.temperature))
            #     # pos = torch.exp(torch.div(torch.bmm(box_pos.view(box_pos.shape[0],1,box_pos.shape[1]),
            #     #                                     z_pos.detach().view(z_pos.shape[0],z_pos.shape[1],1)).view(box_pos.shape[0]), self.temperature))
            #     # neg = torch.sum(torch.exp(torch.div(torch.mm(box_neg,torch.t(z_neg.detach())), self.temperature)), dim=1).mean()
            #     L_batch = -torch.log(torch.div(pos, (pos + neg)))
            #     # L_batch = torch.exp((sim_pos - sim_neg))
            #     # L_batch = torch.exp(sim_pos)
            #     L += L_batch
            L_MS.append(L)
            # pass
        Loss = min(L_MS)
        # L = self.Cos_Distance(box_cls_feat_con, z_expand)
        return Loss
    """

    # CMSP Loss
    def forward(self, box_cls_feat_con, crop_feat_con, batch_size, ious):
        # z = crop_feat_con.detach()
        # print(len(crop_feat_con))
        L_MS = []
        for crop_feat in crop_feat_con:
            z_expand = cat(
                [torch.index_select(crop_feat, 0, torch.tensor(val).cuda()) for val in range(batch_size) for i in
                 range(256)], dim=0)
            L = 0
            for b in range(batch_size):
                b_ious = ious[b * 256: b * 256 + 256]
                pos_index = b_ious >= self.iou_threshold
                neg_index = b_ious < self.iou_threshold
                b_z = z_expand[b * 256: b * 256 + 256]
                z_pos = b_z[pos_index]
                z_neg = b_z[neg_index]
                b_box = box_cls_feat_con[b * 256: b * 256 + 256]
                box_pos = b_box[pos_index]
                box_neg = b_box[neg_index]
                # cos distance
                sim_pos = self.Cos_Distance(box_pos, z_pos)
                sim_neg = self.Cos_Distance(box_neg, z_neg)
                # print(sim_pos.shape)
                pos = torch.exp(torch.div(sim_pos, self.temperature))
                neg = torch.exp(torch.div(sim_neg, self.temperature))
                # pos = torch.exp(torch.div(torch.bmm(box_pos.view(box_pos.shape[0],1,box_pos.shape[1]),
                #                                     z_pos.detach().view(z_pos.shape[0],z_pos.shape[1],1)).view(box_pos.shape[0]), self.temperature))
                # neg = torch.sum(torch.exp(torch.div(torch.mm(box_neg,torch.t(z_neg.detach())), self.temperature)), dim=1).mean()
                L_batch = -torch.log(torch.div(pos, (pos + neg)))
                # L_batch = torch.exp((sim_pos - sim_neg))
                # L_batch = torch.exp(sim_pos)
                L += L_batch
            L_MS.append(L)
            # pass
        Loss = min(L_MS)/batch_size
        # L = self.Cos_Distance(box_cls_feat_con, z_expand)
        return Loss

