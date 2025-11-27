
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from utils.helpers import set_seed, save_checkpoint
from utils.datasets import SketchyDataset
from utils.transforms import default_transforms
from models.freq_encoder import DCTFrequencyEncoder
from models.spatial_encoder import SpatialEncoder
from models.cross_domain_transformer import CrossDomainTransformer
from models.diffusion_fusion import DiffusionLatentFusion
import os

def build_model(cfg, device):
    freq = DCTFrequencyEncoder(in_ch=1, emb_dim=cfg['model']['freq_embedding_dim'])
    spatial = SpatialEncoder(out_dim=cfg['model']['freq_embedding_dim'])
    cdt = CrossDomainTransformer(dim=cfg['model']['freq_embedding_dim'])
    diffusion = DiffusionLatentFusion(latent_dim=cfg['model']['latent_dim'], timesteps=cfg.get('model',{}).get('ddpm_timesteps',50), device=device)
    return dict(freq=freq, spatial=spatial, cdt=cdt, diffusion=diffusion)

def main(args):
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(cfg, device)
    for k,m in model.items():
        m.to(device)
    # dataset skeleton
    ds = SketchyDataset(csv_file=os.path.join(cfg['data']['root'],'sketchy_mini','split_train.csv'),
                        root=os.path.join(cfg['data']['root'],'sketchy_mini'),
                        transform=default_transforms())
    loader = DataLoader(ds, batch_size=cfg['training']['batch_size'], shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(
        list(model['freq'].parameters()) + list(model['spatial'].parameters()) + list(model['cdt'].parameters()) + list(model['diffusion'].parameters()),
        lr=cfg['training']['lr']
    )
    lambda_ddpm = cfg.get('training',{}).get('lambda_ddpm', 0.1)
    for epoch in range(cfg['training']['epochs']):
        model['freq'].train(); model['spatial'].train(); model['cdt'].train(); # diffusion used for loss
        total_loss = 0.0
        for sketch, img, label in loader:
            sketch = sketch.to(device)
            img = img.to(device)
            # forward
            f_emb = model['freq'](sketch)
            s_emb = model['spatial'](img)
            f_align, s_align = model['cdt'](f_emb, s_emb)
            # Triplet loss with simple in-batch mining
            # anchor: sketch embedding (f_align), positive: image embedding with same label, negative: image embedding with different label
            batch_size = f_align.size(0)
            # prepare positive and negative indices in-batch
            pos_idx = [-1] * batch_size
            neg_idx = [-1] * batch_size
            labels_list = label.tolist()
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and labels_list[j] == labels_list[i]:
                        pos_idx[i] = j
                        break
                for k in range(batch_size):
                    if labels_list[k] != labels_list[i]:
                        neg_idx[i] = k
                        break
            # collect triplets
            anchors = []
            positives = []
            negatives = []
            for i in range(batch_size):
                if pos_idx[i] >= 0 and neg_idx[i] >= 0:
                    anchors.append(f_align[i])
                    positives.append(s_align[pos_idx[i]])
                    negatives.append(s_align[neg_idx[i]])
            if len(anchors) > 0:
                anchors = torch.stack(anchors)
                positives = torch.stack(positives)
                negatives = torch.stack(negatives)
                triplet_loss_fn = torch.nn.TripletMarginLoss(margin=0.2, p=2)
                recon_loss = triplet_loss_fn(anchors, positives, negatives)
            else:
                # fallback to simple MSE if no valid triplets in batch
                recon_loss = ((f_align - s_align).pow(2).sum(dim=1)).mean()
            # ddpm loss on fused latent (we use average of f_align and s_align as latent)
            latent = (f_align + s_align) / 2.0
            ddpm_loss = model['diffusion'].ddpm_loss(latent)
            loss = recon_loss + lambda_ddpm * ddpm_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss={total_loss/len(loader):.4f} (recon={recon_loss.item():.4f}, ddpm={ddpm_loss.item():.4f})")
        # saving small checkpoint
        os.makedirs(cfg['logging']['output_dir'], exist_ok=True)
        save_checkpoint({'epoch': epoch+1, 'state_dict': {k: v.state_dict() for k,v in model.items()}},
                        os.path.join(cfg['logging']['output_dir'], 'checkpoint_last.pth'))
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()
    main(args)
