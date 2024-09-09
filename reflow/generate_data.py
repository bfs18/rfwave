import torch

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from pathlib import Path

from rfwave.dataset import VocosDataModule, DataConfig, VocosDataset
from inference_voc import create_instance, load_model, load_config


torch.set_float32_matmul_precision('high')


def sample_ode(reflow, mel, encodec_bandwidth_id=None, N=100):
    traj = []  # to store the trajectory
    dt = 1. / N

    bandwidth_id = torch.tile(torch.arange(reflow.num_bands, device=mel.device), (mel.size(0),))

    if reflow.wave:
        nf = mel.shape[2] if reflow.head.padding == "same" else (mel.shape[2] - 1)
        r = torch.randn([mel.shape[0], reflow.head.hop_length * nf], device=mel.device)
        rf = reflow.stft(r)
        z0 = reflow.get_joint_subband(rf)
    else:
        r = torch.randn([mel.shape[0], reflow.head.n_fft + 2, mel.shape[2]], device=mel.device)
        z0 = reflow.get_joint_subband(r)
    z0 = z0.reshape(z0.size(0) * reflow.num_bands, z0.size(1) // reflow.num_bands, z0.size(2))

    mel = torch.repeat_interleave(mel, reflow.num_bands, 0)
    z = z0.detach()
    fs = (z.size(0) // reflow.num_bands, z.size(1) * reflow.num_bands, z.size(2))
    ss = z.shape
    for i in range(N):
        t = torch.ones(z.size(0)) * i / N
        if reflow.cfg:
            mel_ = torch.cat([mel, torch.ones_like(mel) * mel.mean(dim=(2,), keepdim=True)], dim=0)
            (z_, t_, bandwidth_id_) = [torch.cat([v] * 2, dim=0) for v in (z, t, bandwidth_id)]
            pred = reflow.get_pred(z_, t_.to(mel.device), mel_, bandwidth_id_, encodec_bandwidth_id)
            pred, uncond_pred = torch.chunk(pred, 2, dim=0)
            pred = uncond_pred + reflow.guidance_scale * (pred - uncond_pred)
        else:
            pred = reflow.get_pred(z, t.to(mel.device), mel, bandwidth_id, encodec_bandwidth_id)
        if reflow.wave:
            pred = reflow.place_joint_subband(pred.reshape(fs))
            pred = reflow.stft(reflow.istft(pred))
            pred = reflow.get_joint_subband(pred).reshape(ss)
            z = z.detach() + pred * dt
        else:
            z = z.detach() + pred * dt
        if i == N - 1:
            traj.append(z.detach())
        traj = [reflow.place_joint_subband(tt.reshape(tt.size(0) // reflow.num_bands, -1, tt.size(2)))
                for tt in traj]
        traj = [reflow.get_wave(tt) for tt in traj]
    return r, traj[0]


def generate_data(exp, dataloader, num_pairs, save_dir, device):
    i = 0
    e = 0
    dataiter = iter(dataloader)
    print("epoch 0")
    while i < num_pairs:
        print(f"sample {i}...")
        try:
            batch = next(dataiter)
        except StopIteration:
            e += 1
            print(f"epoch {e}")
            dataiter = iter(dataloader)
            batch = next(dataiter)
        features = exp.feature_extractor(batch.to(device))
        with torch.no_grad():
            z0, z1 = sample_ode(exp.reflow, features)
        for j in range(z0.size(0)):
            save_fp = Path(save_dir) / f'sample_{i:0>7d}_{j:0>3d}.th'
            torch.save({'z0': z0[j].cpu(), 'z1': z1[j].cpu(),
                        'mel': features[j].cpu()}, save_fp.as_posix())
        i += 1


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--num_pairs', type=int, default=100000)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp = load_model(args.model_dir, device=device, last=True)
    config = load_config(Path(args.model_dir) / 'config.yaml')
    data_config = DataConfig(**config['data']['init_args']['train_params'])
    dataset = VocosDataset(data_config, train=True)
    dataloader = DataLoader(
        dataset, batch_size=data_config.batch_size, num_workers=data_config.num_workers,
        shuffle=True, pin_memory=True)

    Path(args.save_dir).mkdir(exist_ok=True)
    generate_data(exp, dataloader, args.num_pairs, args.save_dir, device)
