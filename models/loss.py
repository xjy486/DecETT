import torch
import torch.nn as nn

class DecETT_Loss(nn.Module):
    def __init__(self, args):
        super(DecETT_Loss, self).__init__()
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, pred_tls, pred_tun, rec_tls, rec_tun, pred_adv_tls, pred_adv_tun, y, x_tls, x_tun):
        # Classification Loss
        loss_cls = self.criterion(pred_tls, y) + self.criterion(pred_tun, y)
        
        # Reconstruction Loss
        # x_tls/tun are indices (batch, seq_len)
        # rec_tls/tun are logits (batch, vocab_size) ? No, (batch, 2*max_packet_len+1)
        # Wait, reconstruction target is the input sequence? 
        # In `drl.py`, `rec_tls` output size is `2*max_packet_len+1`.
        # This suggests it's predicting a single token or a distribution over the packet length vocabulary?
        # The input `x_tls` is a sequence of packet lengths.
        # But `rec_tls` output is (batch, vocab_size). It seems to be reconstructing a global representation or just one step?
        # Let's look at `drl.py` again.
        # `rec_tls` is a Sequential(Linear, SELU, Linear). It outputs a single vector of size vocab_size.
        # `x_tls` has shape (batch, seq_len).
        # If it's reconstructing the *input*, it might be an AutoEncoder style where it reconstructs the *bag of words* or just the *original input* if it was a single feature?
        # But `x_tls` is a sequence.
        # Maybe it reconstructs the *sum* or *mean*?
        # Or maybe the "reconstruction" is actually predicting the *next* packet?
        # Given the output shape (batch, vocab_size), it looks like it predicts *one* thing.
        # If the paper says "reconstruction", and the output is vocab size, maybe it's predicting the *distribution* of packet lengths?
        # Or maybe `x_tls` is not a sequence but a single value?
        # `dataset.py`: `x_tls` comes from `feature_extractor`. `seq_features.DRLPacketSizeSequence` suggests it's a sequence.
        # `drl.py`: `x_tls = X[:, 0, :]`. `embedding(x_tls...)`.
        # So `x_tls` is definitely a sequence.
        
        # Hypothesis: The reconstruction target is the *original sequence* but the decoder only outputs *one* step?
        # That doesn't make sense for full sequence reconstruction.
        # However, `rec_tls` in `drl.py` is `Linear(..., 2*max_packet_len+1)`.
        # This output dimension matches the vocabulary size (packet lengths).
        # Perhaps it tries to predict *which* packet lengths are present (Multi-label)?
        # Or maybe it's just a simple auxiliary task.
        
        # Let's assume it tries to reconstruct the *input sequence* but the implementation in `drl.py` (which I didn't write, it was there) only has a simple Linear layer.
        # Wait, `Decoder` class in `drl.py` has a `BiGRU`.
        # `dec_output, dec_hn_tls = self.decoder(concat)`
        # `rec_tls_feat` comes from `self.dec`.
        # `rec_tls` takes `rec_tls_feat.squeeze(1)`.
        # This implies `rec_tls_feat` has seq_len=1.
        # So the decoder only runs for 1 step.
        
        # If the decoder only runs for 1 step, what can it reconstruct?
        # Maybe the *first* packet? Or the *last*?
        # Or maybe it's not reconstruction in the sequence-to-sequence sense.
        
        # Let's check `run.sh`. It has `--recon_loss`.
        # If I look at `dataset.py`, `x_tls` is `torch.tensor(X_tls, dtype=torch.float)`.
        # Wait, `drl.py` casts `X` to `torch.long`.
        # `x_tls = X[:, 0, :]`.
        
        # Let's assume the target is simply the input `x_tls`.
        # But `x_tls` is (batch, 200).
        # `rec_tls` is (batch, 6001).
        # We can't compute CrossEntropy between (batch, 6001) and (batch, 200).
        
        # Maybe the target is not `x_tls`?
        # Is there any other target?
        
        # Let's look at the paper title "Dual Decouple-based Semantic Enhancement".
        # Maybe it reconstructs a "statistical feature" vector?
        
        # Alternative: The code I see in `drl.py` for `rec_tls` might be incomplete or I misunderstood it.
        # `self.rec_tls = nn.Sequential(...)`
        # It outputs `2*max_packet_len+1`.
        # This is exactly the vocabulary size of packet lengths (range -3000 to 3000).
        # So it predicts a probability distribution over packet lengths.
        # Maybe it predicts the *distribution* of packet lengths in the flow?
        # In that case, we can treat `x_tls` as a bag-of-words (bag-of-packets) and minimize KL divergence or Cross Entropy against the empirical distribution.
        
        # Let's try to implement a "Bag of Packets" reconstruction loss.
        # Target: Normalized frequency of each packet length in `x_tls`.
        # Pred: Softmax of `rec_tls`.
        
        loss_rec = 0
        if self.args.recon_loss:
            # Create Bag-of-Packets target
            # x_tls: (batch, seq_len). Values are in [-max_len, max_len].
            # We need to map them to [0, 2*max_len].
            # In `drl.py`: `emb_tls = self.embedding(x_tls + abs(self.max_packet_len))`
            # So indices are `x_tls + 3000`.
            
            target_tls = torch.zeros_like(rec_tls)
            target_tun = torch.zeros_like(rec_tun)
            
            # Scatter add to create counts
            # We need to iterate or use scatter.
            # x_tls_indices = x_tls + self.args.max_packet_len
            # target_tls.scatter_add_(1, x_tls_indices, torch.ones_like(x_tls_indices).float())
            # Normalize to get distribution
            # target_tls = target_tls / target_tls.sum(dim=1, keepdim=True)
            
            # However, doing this inside forward might be slow.
            # But it's the most logical interpretation given the output shape.
            
            x_tls_idx = (x_tls + self.args.max_packet_len).long()
            x_tun_idx = (x_tun + self.args.max_packet_len).long()
            
            # Ensure indices are within range
            x_tls_idx = torch.clamp(x_tls_idx, 0, 2*self.args.max_packet_len)
            x_tun_idx = torch.clamp(x_tun_idx, 0, 2*self.args.max_packet_len)

            target_tls.scatter_add_(1, x_tls_idx, torch.ones_like(x_tls_idx, dtype=torch.float))
            target_tun.scatter_add_(1, x_tun_idx, torch.ones_like(x_tun_idx, dtype=torch.float))
            
            # Normalize to get probability distribution
            target_tls = target_tls / (target_tls.sum(dim=1, keepdim=True) + 1e-8)
            target_tun = target_tun / (target_tun.sum(dim=1, keepdim=True) + 1e-8)
            
            # KL Divergence or Cross Entropy?
            # nn.CrossEntropyLoss expects class indices, not prob distributions (unless soft targets supported in newer pytorch, but let's be safe).
            # We can use KLDivLoss.
            # Input to KLDivLoss should be LogSoftmax.
            
            log_probs_tls = torch.log_softmax(rec_tls, dim=1)
            log_probs_tun = torch.log_softmax(rec_tun, dim=1)
            
            loss_rec = nn.KLDivLoss(reduction='batchmean')(log_probs_tls, target_tls) + \
                       nn.KLDivLoss(reduction='batchmean')(log_probs_tun, target_tun)

        # Adversarial Loss
        # We want the attribute encoder to NOT contain class info.
        # But GRL reverses the gradient.
        # So we just train the adversary to classify correctly, and the GRL will take care of the rest (making the encoder fail).
        loss_adv = self.criterion(pred_adv_tls, y) + self.criterion(pred_adv_tun, y)
        
        total_loss = loss_cls + loss_rec + loss_adv
        
        return total_loss, loss_cls, loss_rec, loss_adv
