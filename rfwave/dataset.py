import numpy as np
import torch
import torchaudio
import kaldiio
import random

from dataclasses import dataclass
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
from rfwave.bucket import DynamicBucketingDataset, DynamicBucketingSampler
from torch.nn import functional as F

torch.set_num_threads(1)


def get_num_tokens(tokens, padding_value=0):
    # take 1 pad in to consideration.
    num_tokens = (tokens != padding_value).sum(1)
    num_tokens += (tokens[:, -1] == padding_value)
    return num_tokens


def get_exp_length(num_tokens, token_exp_scale):
    max_val = num_tokens.max()
    num_tokens = torch.where(num_tokens == max_val, num_tokens, num_tokens - 1)
    length = torch.round(num_tokens * token_exp_scale).long()
    return length


def get_exp_scale(num_tokens, length):
    max_val = num_tokens.max()
    token_exp_scale = torch.where(
        num_tokens == max_val, length.float() / num_tokens.float(), length.float() / (num_tokens.float() - 1))
    return token_exp_scale


@dataclass
class DataConfig:
    filelist_path: str
    batch_size: int
    num_workers: int
    sampling_rate: int = 24000
    num_samples: int = 65280
    cache: bool = False
    task: str = "voc"
    hop_length: int = None
    padding: str = None
    phoneset: str = None
    segment: bool = True
    min_context: int = 50
    max_context: int = 300
    max_duration: float = 100
    max_cuts: int = 32
    num_buckets: int = 20
    drop_last: bool = False
    quadratic_duration: Optional[float] = None
    filter_max_duration: Optional[float] = None
    random_batch_every_epoch: bool = False


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        if cfg.task == "voc":
            if cfg.segment:
                if cfg.filelist_path.endswith(".scp"):
                    dataset = ArkDatasetSegment(cfg, train=train)
                else:
                    dataset = VocosDatasetSegment(cfg, train=train)
            else:
                assert not cfg.filelist_path.endswith(".scp"), \
                    "filelist should not be in .scp format when using segment input"
                dataset = VocosDataset(cfg, train=train)
            collate_fn = voc_collate
        elif cfg.task == "tts":
            assert not cfg.segment
            if cfg.min_context > 0:
                dataset = TTSCtxDataset(cfg, train=train)
                collate_fn = tts_ctx_collate
            else:
                dataset = TTSDataset(cfg, train=train)
                collate_fn = tts_collate
        elif cfg.task == "dur":
            dataset = DurDataset(cfg, train=train)
            collate_fn = dur_collate
        elif cfg.task == "e2e":
            dataset = E2ETTSCtxDataset(cfg, train=train)
            collate_fn = e2e_tts_ctx_collate
        else:
            raise ValueError(f"Unknown task: {cfg.task}")
        if cfg.segment:
            dataloader = DataLoader(
                dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train,
                pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        else:
            batch_sampler = DynamicBucketingSampler(dataset, random_batch_every_epoch=cfg.random_batch_every_epoch)
            dataloader = DataLoader(
                dataset, batch_sampler=batch_sampler, num_workers=cfg.num_workers,
                pin_memory=True, collate_fn=collate_fn, persistent_workers=True)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


class VocosDatasetSegment(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "voc"
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self._cache = dict() if getattr(cfg, 'cache', False) else None

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int):
        audio_path = self.filelist[index]
        fn = Path(audio_path).name
        if self._cache is None or fn not in self._cache:
            y, sr = torchaudio.load(audio_path)
            if self._cache is not None:
                self._cache[fn] = (y, sr)
        else:
            y, sr = self._cache[fn]
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            start = 0
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            start = 0
            y = y[:, : self.num_samples]
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        return y[0], torch.tensor(start), self.num_samples


class VocosDataset(DynamicBucketingDataset):
    def __init__(self, cfg: DataConfig, train: bool):
        # inheritage make sure use the same filelist.
        super(VocosDataset, self).__init__(
            filelist_path=cfg.filelist_path,
            max_duration=cfg.max_duration,
            max_cuts=cfg.max_cuts,
            num_buckets=cfg.num_buckets,
            shuffle=train,
            drop_last=cfg.drop_last,
            quadratic_duration=cfg.quadratic_duration,
            filter_max_duration=cfg.filter_max_duration)
        assert cfg.task == "voc"
        assert cfg.hop_length is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.train = train
        self.hop_length = cfg.hop_length
        self._cache = dict() if getattr(cfg, 'cache', False) else None

    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int):
        audio_path = self.filelist[index]
        fn = Path(audio_path).name
        if self._cache is None or fn not in self._cache:
            y, sr = torchaudio.load(audio_path)
            if self._cache is not None:
                self._cache[fn] = (y, sr)
        else:
            y, sr = self._cache[fn]
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        y = y[:, :y.size(1) // self.hop_length * self.hop_length]
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        start = 0
        num_samples = y.shape[1]
        return y[0], torch.tensor(start), num_samples


def load_ark_scp(scp_fp):
    ark_dict = kaldiio.load_scp(scp_fp, max_cache_fd=32)
    return ark_dict


class ArkDatasetSegment(torch.utils.data.Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "voc"
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        random.seed(123456)
        self.ark_dict = load_ark_scp(cfg.filelist_path)
        self.keys = sorted(list(self.ark_dict.keys()))
        random.shuffle(self.keys)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        k = self.keys[index]
        sr, y = self.ark_dict[k]  # soundfile read, [t, channels]
        y = torch.from_numpy(y.T.astype('float32'))
        if y.ndim == 1:
            y = y[None, :]
        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)
        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
        if y.size(-1) < self.num_samples:
            pad_length = self.num_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.num_samples + 1)
            y = y[:, start: start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        return y[0], start, self.num_samples


def expand_token_by_alignment(tokens, durations, phoneset):
    phone2id = dict([(p, i) for i, p in enumerate(phoneset)])
    ids = [phone2id[tk] for tk in tokens]
    exp_ids = []
    for id, dur in zip(ids, durations):
        exp_ids.extend([id] * dur)
    return torch.tensor(exp_ids, dtype=torch.long)


def upsample_to_gt_len(tokens, audio_len, hop_length, padding):
    gt_len = int(audio_len / hop_length) + (1 if padding == "center" else 0)
    tokens = tokens.to(torch.float32)[None, None, :]
    up_tokens = F.interpolate(tokens, size=(gt_len,), mode='nearest')[0, 0]
    up_tokens = up_tokens.to(torch.long)
    return up_tokens


def upsample_durations(durations, audio_len, hop_length, padding):
    gt_len = int(audio_len / hop_length) + (1 if padding == "center" else 0)
    durations = F.pad(durations, [1, 0])
    cs_durations = torch.cumsum(durations, dim=0)
    us_durations = cs_durations.float() * gt_len / cs_durations[-1]
    us_durations = torch.round(us_durations).long()
    return us_durations[1:] - us_durations[:-1]


class TTSDatasetSegment(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "tts"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None
        self.gain = -3.

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        k, audio_fp, alignment_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(alignment_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            durations = (alignment['durations'] if isinstance(alignment['durations'], torch.Tensor)
                         else torch.tensor(alignment['durations']))
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            up_durations = upsample_durations(durations, y.size(1), self.hop_length, self.padding)
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{self.gain:.2f}"]])
            if self._cache is not None:
                self._cache[k] = (y, token_ids, up_durations)
        else:
            y, token_ids, up_durations = self._cache[k]

        num_frames = self.num_samples // self.hop_length + (1 if self.padding == "center" else 0)
        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        token_ids = token_ids.detach().clone()
        up_durations = up_durations.detach().clone()
        cs_durations = torch.cumsum(up_durations, 0)

        if y.size(-1) < self.num_samples + self.hop_length:
            repeats = np.ceil((self.num_samples + self.hop_length) / y.size(-1)).astype(np.int64)
            y = y.repeat(1, repeats)
            token_ids = token_ids.repeat(repeats)
            up_durations = up_durations.repeat(repeats)
            cs_durations = torch.cumsum(up_durations, 0)

        total_frames = y.size(-1) // self.hop_length
        assert total_frames - num_frames + 1 > 0, (
            f"y length {y.size(-1)}, total_frames {total_frames}, num_frames {num_frames}")
        if self.train:
            start_frame = np.random.randint(low=0, high=total_frames - num_frames + 1)
            start = start_frame * self.hop_length
            end_frame = start_frame + num_frames
            y = y[:, start: start + self.num_samples]
            start_phone_idx = torch.searchsorted(cs_durations, start_frame, right=True)
            end_phone_idx = torch.searchsorted(cs_durations, end_frame, right=False)
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]
            start_frame = 0
            end_frame = start_frame + num_frames
            start_phone_idx = 0
            end_phone_idx = torch.searchsorted(cs_durations, end_frame, right=False)

        token_ids = token_ids[start_phone_idx: end_phone_idx + 1]
        durations = up_durations[start_phone_idx: end_phone_idx + 1].detach().clone()
        if end_phone_idx != start_phone_idx:
            first_num_frames = cs_durations[start_phone_idx] - start_frame
            last_num_frames = end_frame - cs_durations[end_phone_idx - 1]
            durations[0] = first_num_frames
            durations[-1] = last_num_frames
        else:
            durations[0] = end_frame - start_frame

        # gain = np.random.uniform(-1, -6) if self.train else -3
        # y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        assert torch.sum(durations) == num_frames, (
            f"{k}, sum durations {torch.sum(durations)}, num frames: {num_frames}, "
            f"start_phone_idx: {start_phone_idx}, start_frame: {start_frame}, durations: {durations}")
        return y[0], (token_ids, durations, start_phone_idx, start_frame)


class TTSDataset(DynamicBucketingDataset):
    def __init__(self, cfg: DataConfig, train: bool):
        # inheritage make sure use the same filelist.
        super(TTSDataset, self).__init__(
            filelist_path=cfg.filelist_path,
            max_duration=cfg.max_duration,
            max_cuts=cfg.max_cuts,
            num_buckets=cfg.num_buckets,
            shuffle=train,
            drop_last=cfg.drop_last,
            quadratic_duration=cfg.quadratic_duration,
            filter_max_duration=cfg.filter_max_duration)
        assert cfg.task == "tts"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        self.num_tokens = [int(fl.split('|')[3]) for fl in self.filelist]
        self.sampling_rate = cfg.sampling_rate
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None
        self.gain = -3.

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        k, audio_fp, alignment_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(alignment_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            durations = (alignment['durations'] if isinstance(alignment['durations'], torch.Tensor)
                         else torch.tensor(alignment['durations']))
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            up_durations = upsample_durations(durations, y.size(1), self.hop_length, self.padding)
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{self.gain:.2f}"]])
            if self._cache is not None:
                self._cache[k] = (y, token_ids, up_durations)
        else:
            y, token_ids, up_durations = self._cache[k]

        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        num_frames = y.size(1) // self.hop_length + (1 if self.padding == "center" else 0)
        token_ids = token_ids.detach().clone()
        durations = up_durations.detach().clone()

        start_frame = 0
        start_phone_idx = 0

        # gain = np.random.uniform(-1, -6) if self.train else -3
        # y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        assert torch.sum(durations) == num_frames, (
            f"{k}, sum durations {torch.sum(durations)}, num frames: {num_frames}, "
            f"start_phone_idx: {start_phone_idx}, start_frame: {start_frame}, durations: {durations}")
        return y[0], (token_ids, durations, start_phone_idx, start_frame)


class TTSCtxDatasetSegment(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "tts"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None
        self.min_context = cfg.min_context
        self.max_context = cfg.max_context
        np.random.seed(1234)
        self.gain = -3.

    def __len__(self):
        return len(self.filelist)

    def get_token_ids_durations(self, cs_durations, token_ids, up_durations, start_frame, end_frame):
        start_phone_idx = torch.searchsorted(cs_durations, start_frame, right=True) if start_frame > 0 else 0
        end_phone_idx = torch.searchsorted(cs_durations, end_frame, right=False)
        seg_token_ids = token_ids[start_phone_idx: end_phone_idx + 1].detach().clone()
        seg_durations = up_durations[start_phone_idx: end_phone_idx + 1].detach().clone()
        if end_phone_idx != start_phone_idx:
            first_num_frames = cs_durations[start_phone_idx] - start_frame
            last_num_frames = end_frame - cs_durations[end_phone_idx - 1]
            seg_durations[0] = first_num_frames
            seg_durations[-1] = last_num_frames
        else:
            seg_durations[0] = end_frame - start_frame
        return seg_token_ids, seg_durations, start_phone_idx, end_phone_idx

    def __getitem__(self, index):
        k, audio_fp, alignment_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(alignment_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            durations = (alignment['durations'] if isinstance(alignment['durations'], torch.Tensor)
                         else torch.tensor(alignment['durations']))
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            up_durations = upsample_durations(durations, y.size(1), self.hop_length, self.padding)
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{self.gain:.2f}"]])
            if self._cache is not None:
                self._cache[k] = (y, token_ids, up_durations)
        else:
            y, token_ids, up_durations = self._cache[k]

        num_frames = self.num_samples // self.hop_length + (1 if self.padding == "center" else 0)
        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        token_ids = token_ids.detach().clone()
        up_durations = up_durations.detach().clone()
        cs_durations = torch.cumsum(up_durations, 0)

        if y.size(-1) < self.num_samples + self.hop_length + self.min_context * self.hop_length:
            # deal with short sentence carefully
            if cs_durations.size(0) < 2:
                repeats = np.ceil((self.num_samples + self.hop_length) / y.size(-1)).astype(np.int64)
                y = y.repeat(1, repeats)
                token_ids = token_ids.repeat(repeats)
                up_durations = up_durations.repeat(repeats)
                cs_durations = torch.cumsum(up_durations, 0)
                start = 0
                start_frame = 0
                end_frame = num_frames
                ctx_start_frame = 0
                ctx_n_frame = self.min_context
                y_seg = y[:, : self.num_samples]
            else:
                half_frames = cs_durations[-1] // 2
                hf = torch.searchsorted(cs_durations, half_frames, right=False)
                hf = hf - 1 if hf == cs_durations.size(0) - 1 else hf
                hf = hf if torch.sum(up_durations[:hf]) > torch.sum(up_durations[hf + 1]) else hf + 1
                half_frames = torch.sum(up_durations[:hf])
                lo_up_duration = up_durations[:hf]
                hi_up_duration = up_durations[hf:]
                lo_y = y[:, :half_frames * self.hop_length]
                hi_y = y[:, half_frames * self.hop_length:]
                lo_token_ids = token_ids[:hf]
                hi_token_ids = token_ids[hf:]
                repeats = np.ceil(
                    (self.num_samples + self.hop_length) / min(lo_y.size(-1), hi_y.size(-1))).astype(np.int64)
                lo_y = lo_y.repeat(1, repeats)
                hi_y = hi_y.repeat(1, repeats)
                lo_up_duration = lo_up_duration.repeat(repeats)
                hi_up_duration = hi_up_duration.repeat(repeats)
                lo_token_ids = lo_token_ids.repeat(repeats)
                hi_token_ids = hi_token_ids.repeat(repeats)
                y = torch.cat([lo_y, hi_y], dim=-1)
                up_durations = torch.cat([lo_up_duration, hi_up_duration], dim=0)
                token_ids = torch.cat([lo_token_ids, hi_token_ids], dim=0)
                cs_durations = torch.cumsum(up_durations, dim=0)
                if np.random.uniform() < 0.5:
                    start = 0
                    start_frame = 0
                    end_frame = num_frames
                    y_seg = y[:, :self.num_samples]
                    ctx_start_frame = torch.sum(lo_up_duration)
                else:
                    start_frame = torch.sum(lo_up_duration)
                    start = start_frame * self.hop_length
                    end_frame = num_frames + start_frame
                    y_seg = y[:, start: start + self.num_samples]
                    ctx_start_frame = 0
                ctx_n_frame = self.min_context
        else:
            total_frames = y.size(-1) // self.hop_length
            assert total_frames - num_frames + 1 > 0, (
                f"y length {y.size(-1)}, total_frames {total_frames}, num_frames {num_frames}")
            if self.train:
                start_frame = np.random.randint(low=0, high=total_frames - num_frames + 1)
                start = start_frame * self.hop_length
                end_frame = start_frame + num_frames
                y_seg = y[:, start: start + self.num_samples]
            else:
                # During validation, take always the first segment for determinism
                y_seg = y[:, : self.num_samples]
                start = 0
                start_frame = 0
                end_frame = start_frame + num_frames

            # get context
            if start_frame > self.min_context:
                max_context = np.minimum(start_frame, self.max_context)
                ctx_n_frame = np.random.randint(self.min_context, max_context)
                ctx_start_frame = np.random.randint(0, start_frame - ctx_n_frame)
            elif total_frames - (start_frame + num_frames) > self.min_context:
                max_context = np.minimum(total_frames - (start_frame + num_frames), self.max_context)
                ctx_n_frame = np.random.randint(self.min_context, max_context)
                ctx_start_frame = np.random.randint(start_frame + num_frames, total_frames - ctx_n_frame)
            else:
                # ctx_start_frame = 0
                # ctx_n_frame = self.min_context
                if start_frame > total_frames - (start_frame + num_frames):
                    ctx_start_frame = 0
                    ctx_n_frame = start_frame
                else:
                    ctx_start_frame = start_frame + num_frames
                    ctx_n_frame = total_frames - ctx_start_frame

        seg_token_ids, seg_durations, start_phone_idx, _ = self.get_token_ids_durations(
            cs_durations, token_ids, up_durations, start_frame, end_frame)

        # get context
        ctx_start = ctx_start_frame * self.hop_length
        ctx_end = (ctx_start_frame + ctx_n_frame) * self.hop_length
        y_ctx = y[:, ctx_start: ctx_end]
        ctx_n_frame = ctx_n_frame + 1 if self.padding == 'center' else ctx_n_frame

        ctx_token_ids, ctx_durations, *_ = self.get_token_ids_durations(
            cs_durations, token_ids, up_durations, ctx_start_frame, ctx_start_frame + ctx_n_frame)

        # gain = np.random.uniform(-1, -6) if self.train else -3
        # y_seg, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     y_seg, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        # y_ctx, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     y_ctx, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        assert torch.sum(seg_durations) == num_frames, (
            f"{k}, sum durations {torch.sum(seg_durations)}, num frames: {num_frames}, "
            f"start_phone_idx: {start_phone_idx}, start_frame: {start_frame}, durations: {seg_durations}")
        assert torch.sum(ctx_durations) == ctx_n_frame, (
            f"{k}, ctx sum durations {torch.sum(ctx_durations)}, ctx num_frames: {ctx_n_frame}")
        return y_seg[0], (seg_token_ids, seg_durations, start_phone_idx, start_frame, num_frames,
                          y_ctx[0], ctx_start_frame, ctx_n_frame, ctx_token_ids, ctx_durations)


class TTSCtxDataset(DynamicBucketingDataset):
    def __init__(self, cfg: DataConfig, train: bool):
        super(TTSCtxDataset, self).__init__(
            filelist_path=cfg.filelist_path,
            max_duration=cfg.max_duration,
            max_cuts=cfg.max_cuts,
            num_buckets=cfg.num_buckets,
            shuffle=train,
            drop_last=cfg.drop_last,
            quadratic_duration=cfg.quadratic_duration,
            filter_max_duration=cfg.filter_max_duration)
        assert cfg.task == "tts"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None
        self.min_context = cfg.min_context
        self.max_context = cfg.max_context
        np.random.seed(1234)
        self.gain = -3.

    def __len__(self):
        return len(self.filelist)

    def get_token_ids_durations(self, cs_durations, token_ids, up_durations, start_frame, end_frame):
        start_phone_idx = torch.searchsorted(cs_durations, start_frame, right=True) if start_frame > 0 else 0
        end_phone_idx = torch.searchsorted(cs_durations, end_frame, right=False)
        seg_token_ids = token_ids[start_phone_idx: end_phone_idx + 1].detach().clone()
        seg_durations = up_durations[start_phone_idx: end_phone_idx + 1].detach().clone()
        if end_phone_idx != start_phone_idx:
            first_num_frames = cs_durations[start_phone_idx] - start_frame
            last_num_frames = end_frame - cs_durations[end_phone_idx - 1]
            seg_durations[0] = first_num_frames
            seg_durations[-1] = last_num_frames
        else:
            seg_durations[0] = end_frame - start_frame
        return seg_token_ids, seg_durations, start_phone_idx, end_phone_idx

    def __getitem__(self, index):
        k, audio_fp, alignment_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(alignment_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            durations = (alignment['durations'] if isinstance(alignment['durations'], torch.Tensor)
                         else torch.tensor(alignment['durations']))
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            up_durations = upsample_durations(durations, y.size(1), self.hop_length, self.padding)
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{self.gain:.2f}"]])
            if self._cache is not None:
                self._cache[k] = (y, token_ids, up_durations)
        else:
            y, token_ids, up_durations = self._cache[k]

        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        token_ids = token_ids.detach().clone()
        up_durations = up_durations.detach().clone()
        num_frames = y.size(1) // self.hop_length + (1 if self.padding == "center" else 0)
        cs_durations = torch.cumsum(up_durations, 0)

        if y.size(-1) > self.min_context * self.hop_length * 2:
            max_context = np.minimum(num_frames // 2, self.max_context)
            ctx_n_frame = np.random.randint(self.min_context, max_context)
            ctx_start_frame = np.random.randint(0, num_frames - ctx_n_frame - 1)
        else:
            ctx_n_frame = num_frames // 2 - 1
            ctx_start_frame = 0 if np.random.rand() < 0.5 else ctx_n_frame

        # get context
        ctx_start = ctx_start_frame * self.hop_length
        ctx_end = (ctx_start_frame + ctx_n_frame) * self.hop_length
        assert ctx_end <= y.size(1)
        y_ctx = y[:, ctx_start: ctx_end]
        ctx_n_frame = ctx_n_frame + 1 if self.padding == 'center' else ctx_n_frame

        ctx_token_ids, ctx_durations, *_ = self.get_token_ids_durations(
            cs_durations, token_ids, up_durations, ctx_start_frame, ctx_start_frame + ctx_n_frame)

        # gain = np.random.uniform(-1, -6) if self.train else -3
        # y, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     y, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        # y_ctx, _ = torchaudio.sox_effects.apply_effects_tensor(
        #     y_ctx, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        return y[0], (token_ids, up_durations, 0, 0, num_frames,
                      y_ctx[0], ctx_start_frame, ctx_n_frame, ctx_token_ids, ctx_durations)


class E2ETTSCtxDataset(DynamicBucketingDataset):
    def __init__(self, cfg: DataConfig, train: bool):
        super(E2ETTSCtxDataset, self).__init__(
            filelist_path=cfg.filelist_path,
            max_duration=cfg.max_duration,
            max_cuts=cfg.max_cuts,
            num_buckets=cfg.num_buckets,
            shuffle=train,
            drop_last=cfg.drop_last,
            quadratic_duration=cfg.quadratic_duration,
            filter_max_duration=cfg.filter_max_duration)
        assert cfg.task == "e2e"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.sampling_rate = cfg.sampling_rate
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None
        self.min_context = cfg.min_context
        self.max_context = cfg.max_context
        self.gain = -3.

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        k, audio_fp, phone_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(phone_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{self.gain:.2f}"]])
            if self._cache is not None:
                self._cache[k] = (y, token_ids)
        else:
            y, token_ids = self._cache[k]

        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        num_frames = y.size(1) // self.hop_length + (1 if self.padding == "center" else 0)
        token_ids = token_ids.detach().clone()
        exp_scale = num_frames / token_ids.size(0)

        if y.size(-1) > self.min_context * self.hop_length * 2:
            max_context = np.minimum(num_frames // 2, self.max_context)
            ctx_n_frame = np.random.randint(self.min_context, max_context)
            ctx_start_frame = np.random.randint(0, num_frames - ctx_n_frame - 1)
        else:
            ctx_n_frame = num_frames // 2 - 1
            ctx_start_frame = 0 if np.random.rand() < 0.5 else ctx_n_frame

        # get context
        ctx_start = ctx_start_frame * self.hop_length
        ctx_end = (ctx_start_frame + ctx_n_frame) * self.hop_length
        y_ctx = y[:, ctx_start: ctx_end]
        ctx_n_frame = ctx_n_frame + 1 if self.padding == 'center' else ctx_n_frame

        assert ctx_start_frame + ctx_n_frame <= num_frames
        return y[0], (token_ids, len(token_ids), y_ctx[0], ctx_start_frame, ctx_n_frame, exp_scale)


class DurDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "dur"
        assert cfg.phoneset is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.num_tokens = [int(fl.split('|')[3]) for fl in self.filelist]
        self.train = train
        phoneset = torch.load(cfg.phoneset)
        self.phoneset = ["_PAD_"] + phoneset
        self.phone2id = dict([(p, i) for i, p in enumerate(self.phoneset)])
        self._cache = dict() if getattr(cfg, 'cache', False) else None

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, index):
        k, audio_fp, alignment_fp, *_ = self.filelist[index].split("|")
        if self._cache is None or k not in self._cache:
            alignment = torch.load(alignment_fp, map_location="cpu")
            token_ids = torch.tensor([self.phone2id[str(tk)] for tk in alignment['tokens']])
            durations = (alignment['durations'] if isinstance(alignment['durations'], torch.Tensor)
                         else torch.tensor(alignment['durations']))
            if self._cache is not None:
                self._cache[k] = (token_ids, durations)
        else:
            token_ids, durations = self._cache[k]
        token_ids = token_ids.detach().clone()
        durations = durations.detach().clone()
        return token_ids, durations


def voc_collate(data):
    y_lens = [d[0].size(0) for d in data]
    max_y_len = max(y_lens)
    y = torch.zeros([len(data), max_y_len], dtype=torch.float)
    for i, d in enumerate(data):
        y[i, :d[0].size(0)] = d[0]
    start = torch.tensor([d[1] for d in data])
    length = torch.tensor([d[2] for d in data])
    return y, start, length


def tts_collate(data):
    y_lens = [d[0].size(0) for d in data]
    max_y_len = max(y_lens)
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    num_frames = [torch.sum(ti[1]) for ti in token_info]
    max_num_phone = max(num_phones)  # + 1 an extra phone for expanded padding
    max_num_frame = max(num_frames)
    y = torch.zeros([len(data), max_y_len], dtype=torch.float)
    token_ids = torch.zeros([len(data), max_num_phone], dtype=torch.long)
    durations = torch.zeros([len(data), max_num_phone], dtype=torch.long)
    for i, (ti, d, _, _) in enumerate(token_info):
        y[i, :y_lens[i]] = data[i][0]
        token_ids[i, :ti.size(0)] = ti
        durations[i, :ti.size(0)] = d
        # token_ids[i, -1] = 0  # for expanded padding
        # durations[i, -1] = max_num_frame - torch.sum(d)
    start_phone_idx = torch.tensor([ti[2] for ti in token_info])
    start_frame = torch.tensor([ti[3] for ti in token_info])
    return y, [token_ids, durations, start_phone_idx, start_frame]


def tts_ctx_collate(data):
    y_lens = [d[0].size(0) for d in data]
    max_y_len = max(y_lens)
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    num_frames = [torch.sum(ti[1]) for ti in token_info]
    max_num_phones = max(num_phones)  # + 1 an extra phone for expanded padding
    max_num_frame = max(num_frames)
    y = torch.zeros([len(data), max_y_len], dtype=torch.float)
    y_ctx = [ti[5] for ti in token_info]
    max_ctx_len = max([yc.size(0) for yc in y_ctx])
    num_ctx_phones = [ti[8].size(0) for ti in token_info]
    max_ctx_num = max(num_ctx_phones)
    token_ids = torch.zeros([len(data), max_num_phones], dtype=torch.long)
    durations = torch.zeros([len(data), max_num_phones], dtype=torch.long)
    y_ctx_pad = torch.zeros([len(data), max_ctx_len], dtype=torch.float)
    ctx_token_ids = torch.zeros([len(data), max_ctx_num], dtype=torch.long)
    ctx_durations = torch.zeros([len(data), max_ctx_num], dtype=torch.long)
    for i, (ti, d, _, _, _, ctx, _, _, cti, cd) in enumerate(token_info):
        y[i, :y_lens[i]] = data[i][0]
        token_ids[i, :ti.size(0)] = ti
        durations[i, :ti.size(0)] = d
        # token_ids[i, -1] = 0  # for expanded padding
        # durations[i, -1] = max_num_frame - torch.sum(d)
        y_ctx_pad[i, :ctx.size(0)] = ctx
        ctx_token_ids[i, :cti.size(0)] = cti
        ctx_durations[i, :cd.size(0)] = cd
    start_phone_idx = torch.tensor([ti[2] for ti in token_info])
    start_frame = torch.tensor([ti[3] for ti in token_info])
    num_frames = torch.tensor([ti[4] for ti in token_info])
    ctx_start_frame = torch.tensor([ti[6] for ti in token_info])
    ctx_n_frame = torch.tensor([ti[7] for ti in token_info])
    # assert torch.all(durations[:, :-1].sum(1) == num_frames)
    assert torch.all(durations.sum(1) == num_frames)
    return y, [token_ids, durations, start_phone_idx, start_frame, num_frames, y_ctx_pad,
               ctx_start_frame, ctx_n_frame, ctx_token_ids, ctx_durations]


def e2e_tts_ctx_collate(data):
    y_lens = [d[0].size(0) for d in data]
    max_y_len = max(y_lens)
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    max_num = max(num_phones)
    y = torch.zeros([len(data), max_y_len], dtype=torch.float)
    y_ctx = [ti[2] for ti in token_info]
    max_ctx_len = max([y.size(0) for y in y_ctx])
    token_ids = torch.zeros([len(data), max_num], dtype=torch.long)
    y_ctx_pad = torch.zeros([len(data), max_ctx_len], dtype=torch.float)
    for i, (ti, _, ctx, *_) in enumerate(token_info):
        y[i, :y_lens[i]] = data[i][0]
        token_ids[i, :ti.size(0)] = ti
        y_ctx_pad[i, :ctx.size(0)] = ctx
    num_tokens_ = torch.tensor([ti[1] for ti in token_info])
    num_tokens = get_num_tokens(token_ids)
    assert num_tokens_.sum() + len(data) == num_tokens.sum() + (token_ids[:, -1] != 0).sum()
    ctx_start = torch.tensor([ti[3] for ti in token_info])
    ctx_n_frame = torch.tensor([ti[4] for ti in token_info])
    exp_scale_ = torch.tensor([ti[5] for ti in token_info])
    #TODO: a special case, not correct. #tok + 1 = max #tok
    special_exp_scale = torch.round(num_tokens_ * exp_scale_) / (num_tokens_ + 1)
    exp_scale = torch.where(num_tokens_ == num_tokens_.max() - 1, special_exp_scale, exp_scale_)
    # assert torch.all(torch.tensor([yl // 256 + 1 for yl in y_lens]) == get_exp_length(num_tokens, exp_scale)), \
    #     (f'num_tokens_: {num_tokens_}, num_tokens:{num_tokens}, frames: {[yl // 256 + 1 for yl in y_lens]}, '
    #      f'exp_scal1:{torch.round(exp_scale_ * num_tokens_).long()} '
    #      f'exp_scale2: {get_exp_length(num_tokens, exp_scale)}')
    return y, [token_ids, num_tokens, y_ctx_pad, ctx_start, ctx_n_frame, exp_scale]


def dur_collate(data):
    num_phones = [len(d[0]) for d in data]
    max_num_phone = max(num_phones)
    token_ids = torch.zeros([len(data), max_num_phone], dtype=torch.long)
    durations = torch.zeros([len(data), 1, max_num_phone], dtype=torch.float)
    for i, (ti, d) in enumerate(data):
        token_ids[i, :ti.size(0)] = ti
        durations[i, 0, :ti.size(0)] = d
    return token_ids, durations


def test_tts_segment():
    from matplotlib import pyplot as plt
    from rfwave.feature_extractors import MelSpectrogramFeatures
    import torch.nn.functional as F
    import matplotlib
    import soundfile as sf
    matplotlib.use('TkAgg')
    LINE_COLORS = ['w', 'r', 'orange', 'k', 'cyan', 'm', 'b', 'lime', 'g', 'brown', 'navy']

    def spec_to_figure(spec, vmin=None, vmax=None, title='', f0s=None, dur_info=None):
        if isinstance(spec, torch.Tensor):
            spec = spec.cpu().numpy()
        H = spec.shape[1]
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
        if dur_info is not None:
            assert isinstance(dur_info, dict)
            txt = dur_info['txt']
            dur_gt = dur_info['dur_gt']
            if isinstance(dur_gt, torch.Tensor):
                dur_gt = dur_gt.cpu().numpy()
            dur_gt = np.cumsum(dur_gt).astype(int)
            for i in range(len(dur_gt)):
                shift = (i % 8) + 1
                plt.text(dur_gt[i], shift * 4, txt[i], ha='right')
                plt.vlines(dur_gt[i], 0, H // 2, colors='b')  # blue is gt
            plt.xlim(0, dur_gt[-1])
            if 'dur_pred' in dur_info:
                dur_pred = dur_info['dur_pred']
                if isinstance(dur_pred, torch.Tensor):
                    dur_pred = dur_pred.cpu().numpy()
                dur_pred = np.cumsum(dur_pred).astype(int)
                for i in range(len(dur_pred)):
                    shift = (i % 8) + 1
                    plt.text(dur_pred[i], H + shift * 4, txt[i], ha='right')
                    plt.vlines(dur_pred[i], H, H * 1.5, colors='r')  # red is pred
                plt.xlim(0, max(dur_gt[-1], dur_pred[-1]))
        if f0s is not None:
            ax = plt.gca()
            ax2 = ax.twinx()
            if not isinstance(f0s, dict):
                f0s = {'f0': f0s}
            for i, (k, f0) in enumerate(f0s.items()):
                if isinstance(f0, torch.Tensor):
                    f0 = f0.cpu().numpy()
                ax2.plot(f0, label=k, c=LINE_COLORS[i], linewidth=1, alpha=0.5)
            ax2.set_ylim(0, 1000)
            ax2.legend()
        return fig

    cfg = DataConfig(
        filelist_path="/data/corpus/LJSpeech-1.1/synta_filelist.valid",
        sampling_rate=22050,
        num_samples=65280,
        batch_size=8,
        num_workers=0,
        cache=True,
        task="tts",
        hop_length=256,
        padding="center",
        phoneset="/data/corpus/LJSpeech-1.1/synta_phoneset.th",
    )

    mel_extractor = MelSpectrogramFeatures(22050)
    dataset = TTSCtxDatasetSegment(cfg, train=False)
    id2phone = dict((i, str(p)) for i, p in enumerate(dataset.phoneset))
    d = dataset[0]
    y = d[0]
    mel = mel_extractor(y.unsqueeze(0))[0]
    tokens = [id2phone[i.item()] for i in d[1][0]]
    durs = d[1][1]
    cumsum_durs = torch.cumsum(F.pad(durs, (1, 0)), dim=0)
    cumsum_durs = cumsum_durs * mel.size(1) / cumsum_durs[-1]
    cumsum_durs = cumsum_durs.round()
    durs = cumsum_durs[1:] - cumsum_durs[:-1]
    dur_info = {
        "txt": tokens,
        "dur_gt": durs.numpy()
    }
    print(tokens)
    print(durs)
    spec_to_figure(mel.T, dur_info=dur_info)
    sf.write('test.wav', y.numpy(), 22050, 'PCM_16')
    plt.show()

    data = [dataset[0], dataset[1]]
    tts_ctx_collate(data)
    for d in dataset:
        print(d[0].shape)


def test_dur():
    dur_cfg = DataConfig(
        filelist_path="/data/corpus/LJSpeech-1.1/synta_filelist.valid",
        batch_size=8,
        num_workers=0,
        cache=True,
        task="dur",
        phoneset="/data/corpus/LJSpeech-1.1/synta_phoneset.th",
    )
    dur_dataset = DurDataset(dur_cfg, train=False)
    print(dur_dataset[0])
    data = [dur_dataset[0], dur_dataset[1]]
    dur_collate(data)


def test_tts_sentence():
    cfg = DataConfig(
        filelist_path="/data1/corpus/LJSpeech-1.1/synta_filelist.valid",
        sampling_rate=22050,
        num_samples=65280,
        batch_size=8,
        num_workers=0,
        cache=True,
        task="tts",
        hop_length=256,
        padding="center",
        phoneset="/data1/corpus/LJSpeech-1.1/synta_phoneset.th",
    )
    dataset = TTSDataset(cfg, train=True)
    sampler = DynamicBucketingSampler(dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=tts_collate)
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))


def test_tts_ctx_sentence():
    cfg = DataConfig(
        filelist_path="/data1/corpus/LJSpeech-1.1/synta_filelist.valid",
        sampling_rate=22050,
        num_samples=65280,
        batch_size=8,
        num_workers=0,
        cache=True,
        task="tts",
        hop_length=256,
        padding="center",
        phoneset="/data1/corpus/LJSpeech-1.1/synta_phoneset.th",
    )
    dataset = TTSCtxDataset(cfg, train=True)
    sampler = DynamicBucketingSampler(dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=tts_ctx_collate)
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))


def test_tts_e2e():
    cfg = DataConfig(
        filelist_path="/Users/liupeng/wd_disk/dataset/LJSpeech-1.1/synta_filelist.train",
        sampling_rate=22050,
        num_samples=65280,
        batch_size=8,
        num_workers=0,
        cache=True,
        task="e2e",
        hop_length=256,
        padding="center",
        phoneset="/Users/liupeng/wd_disk/dataset/LJSpeech-1.1/synta_phoneset.th",
    )
    dataset = E2ETTSCtxDataset(cfg, train=True)
    sampler = DynamicBucketingSampler(dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=e2e_tts_ctx_collate)
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))


def test_voc():
    cfg = DataConfig(
        filelist_path="/Users/liupeng/wd_disk/dataset/LJSpeech-1.1/wav_filelist.train",
        sampling_rate=22050,
        batch_size=1,
        num_workers=0,
        cache=False,
        task="voc",
        hop_length=256,
        padding="center",
    )
    dataset = VocosDataset(cfg, train=True)
    sampler = DynamicBucketingSampler(dataset)
    dataloader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=voc_collate)
    for i, batch in enumerate(dataloader):
        print('batch idx', i, 'batch size', batch[0].size(0))


if __name__ == '__main__':
    # test_voc()
    test_tts_e2e()
