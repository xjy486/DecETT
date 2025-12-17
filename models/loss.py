# 修改 models/loss.py 中的 DecETT_Loss 类
import torch
import torch.nn as nn
class DecETT_Loss(nn.Module):
    def __init__(self, args):
        super(DecETT_Loss, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.cosine_loss = nn.CosineEmbeddingLoss()
        
        # [修改点 4]：使用 MSELoss 计算重构误差 (||x' - x||^2)
        self.mse_loss = nn.MSELoss() 

    def forward(self, outputs, y, x_tls_indices, x_tun_indices):
        # 解包 (注意这里包含了 drl.py 新返回的 emb_tls/tun)
        (pred_tls, pred_tun, 
         rec_tls, rec_tun, 
         rec_cross_tls, rec_cross_tun,
         pred_adv_tls, pred_adv_tun, 
         con_tls_hn, con_tun_hn,
         target_emb_tls, target_emb_tun) = outputs

        # 1. Classification Loss (L_ASC)
        loss_asc = self.criterion(pred_tls, y) + self.criterion(pred_tun, y)
        
        # 2. Adversarial Loss (L_PSM)
        loss_psm = self.criterion(pred_adv_tls, y) + self.criterion(pred_adv_tun, y)

        # 3. Self Reconstruction Loss (L_SRC)
        # 计算重构向量与原始 Embedding 向量的 MSE
        loss_src = self.mse_loss(rec_tls, target_emb_tls) + \
                   self.mse_loss(rec_tun, target_emb_tun)
        
        # 4. Cross Reconstruction Loss (L_CPD)
        # 交叉重构的目标依然是原始输入的 Embedding
        loss_cpd = self.mse_loss(rec_cross_tls, target_emb_tls) + \
                   self.mse_loss(rec_cross_tun, target_emb_tun)

        # 5. Semantic Alignment Loss (L_ASA)
        target_ones = torch.ones(con_tls_hn.size(0)).to(con_tls_hn.device)
        loss_asa = self.cosine_loss(con_tls_hn, con_tun_hn, target_ones)
        
        # 总损失
        total_loss = loss_asc + loss_psm + loss_src + loss_cpd + loss_asa
        
        return total_loss, loss_asc, loss_psm, loss_src, loss_cpd, loss_asa