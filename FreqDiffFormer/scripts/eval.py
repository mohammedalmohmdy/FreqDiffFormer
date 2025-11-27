import argparse
import torch
import yaml
from utils.datasets import SketchyDataset
from utils.transforms import default_transforms
from models.freq_encoder import DCTFrequencyEncoder
from models.spatial_encoder import SpatialEncoder
from models.cross_domain_transformer import CrossDomainTransformer
import os
from sklearn.metrics import average_precision_score

def load_models(cfg, ckpt_path):
    freq = DCTFrequencyEncoder(in_ch=1, emb_dim=cfg['model']['freq_embedding_dim'])
    spatial = SpatialEncoder(out_dim=cfg['model']['freq_embedding_dim'])
    cdt = CrossDomainTransformer(dim=cfg['model']['freq_embedding_dim'])
    # load checkpoint if present
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location='cpu')
        st = ck['state_dict']
        # simplistic load (keys must match)
        for k, m in [('freq', freq), ('spatial', spatial), ('cdt', cdt)]:
            try:
                m.load_state_dict(st[k])
            except Exception as e:
                print("Warning loading", k, e)
    return freq, spatial, cdt

def extract_embeddings(model_tuple, loader, device='cpu'):
    freq, spatial, cdt = model_tuple
    freq.to(device); spatial.to(device); cdt.to(device)
    emb_list = []
    label_list = []
    with torch.no_grad():
        for sketch, img, label in loader:
            sketch = sketch.to(device); img = img.to(device)
            f = freq(sketch)
            s = spatial(img)
            fa, sa = cdt(f, s)
            emb = (fa + sa) / 2.0
            emb_list.append(emb.cpu())
            label_list.append(label)
    import torch
    emb = torch.cat(emb_list, dim=0)
    labels = torch.cat(label_list, dim=0)
    return emb.numpy(), labels.numpy()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--checkpoint', default='outputs/checkpoint_last.pth')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ds = SketchyDataset(csv_file=os.path.join(cfg['data']['root'],'sketchy','split_val.csv'),
                        root=os.path.join(cfg['data']['root'],'sketchy'),
                        transform=default_transforms())
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    freq, spatial, cdt = load_models(cfg, args.checkpoint)
    emb, labels = extract_embeddings((freq, spatial, cdt), loader)
    print("Embeddings shape:", emb.shape)
