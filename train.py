import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loguru import logger as logging
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def train(args, model, train_loader, valid_loader, criterion):
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    model_save_path = os.path.join('saved_models', args.model_file)
    os.makedirs('saved_models', exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_asc = 0
        total_psm = 0
        total_src = 0
        total_cpd = 0
        total_asa = 0
        
        start_time = time.time()
        
        for i, (X, y) in enumerate(train_loader):
            X, y = X.to(args.device), y.to(args.device)
            
            optimizer.zero_grad()
            
            # Forward
            # pred_tls, pred_tun, rec_tls, rec_tun, pred_adv_tls, pred_adv_tun
            outputs = model(X)
            pred_tls, pred_tun, rec_tls, rec_tun, rec_cross_tls, rec_cross_tun, pred_adv_tls, pred_adv_tun, con_tls_hn, con_tun_hn, emb_tls, emb_tun = outputs
            
            # Loss
            # We need x_tls and x_tun for reconstruction loss
            x_tls = X[:, 0, :]
            x_tun = X[:, 1, :]
            
            loss, loss_asc, loss_psm, loss_src, loss_cpd, loss_asa = criterion(
                outputs,
                y, x_tls, x_tun
            )
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_asc += loss_asc.item()
            total_psm += loss_psm.item()
            total_src += loss_src.item()
            total_cpd += loss_cpd.item()
            total_asa += loss_asa.item()
            
            if (i + 1) % 10 == 0 and args.verbose:
                logging.info(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], '
                             f'Loss: {loss.item():.4f}, '
                             f'ASC: {loss_asc.item():.4f}, '
                             f'PSM: {loss_psm.item():.4f}, '
                             f'SRC: {loss_src.item():.4f}, '
                             f'CPD: {loss_cpd.item():.4f}, '
                             f'ASA: {loss_asa.item():.4f}')
        
        epoch_time = time.time() - start_time
        avg_loss = total_loss / len(train_loader)
        logging.info(f'Epoch [{epoch+1}/{args.epochs}] Finished. Time: {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}')
        
        # Validation
        if valid_loader:
            val_acc = evaluate(args, model, valid_loader)
            logging.info(f'Validation Accuracy: {val_acc:.4f}')
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model, model_save_path)
                logging.info(f'Saved best model to {model_save_path}')
        else:
            # Save last model if no validation
            torch.save(model, model_save_path)
            logging.info(f'Saved model to {model_save_path}')

def evaluate(args, model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(args.device), y.to(args.device)
            pred_tls, pred_tun = model(X)
            
            # Combine predictions (e.g., sum logits)
            pred = pred_tls + pred_tun
            pred_cls = torch.argmax(pred, dim=1)
            
            all_preds.extend(pred_cls.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return accuracy_score(all_labels, all_preds)
