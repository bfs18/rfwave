import numpy as np
import torch
import torchaudio
import kaldiio
import random
import torch.nn.functional as F
import types
import sys
import torch.distributed as dist

from dataclasses import dataclass
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

torch.set_num_threads(1)


@dataclass
class DataConfig:
    filelist_path: str
    batch_size: int
    num_workers: int
    sampling_rate: int = 24000
    num_samples: int = 65280
    cache: bool = True
    task: str = "voc"
    hop_length: int = None
    padding: str = None
    phoneset: str = None
    segment: bool = True
    min_context: int = 50
    max_context: int = 300


class VocosDataModule(LightningDataModule):
    def __init__(self, train_params: DataConfig, val_params: DataConfig):
        super().__init__()
        self.train_config = train_params
        self.val_config = val_params

    def _get_dataloder(self, cfg: DataConfig, train: bool):
        if cfg.task == "voc":
            if cfg.filelist_path.endswith(".scp"):
                dataset = ArkDataset(cfg, train=train)
            else:
                dataset = VocosDataset(cfg, train=train)
            collate_fn = None
        elif cfg.task == "tts":
            if cfg.segment:
                if cfg.min_context > 0:
                    print("!!using context!!")
                    dataset = TTSCtxDatasetSegment(cfg, train=train)
                    collate_fn = tts_ctx_collate_segment
                else:
                    dataset = TTSDatasetSegment(cfg, train=train)
                    collate_fn = tts_collate_segment
            else:
                dataset = TTSDataset(cfg, train=train)
                collate_fn = tts_collate
        elif cfg.task == "dur":
            dataset = DurDataset(cfg, train=train)
            collate_fn = dur_collate
        elif cfg.task == "e2e":
            dataset = E2ETTSCtxDataset(cfg, train=train)
            collate_fn = e2e_tts_ctx_collate_segment
        else:
            raise ValueError(f"Unknown task: {cfg.task}")
        if cfg.task == 'tts' and not cfg.segment:
            if dist.is_initialized():
                num_replicas = dist.get_world_size()
                rank = dist.get_rank()
            else:
                num_replicas = 1
                rank = 0
            print("get_dataloader", rank, 'of', num_replicas)
            # required_batch_size_multiple should be multiple of gpu device count
            batch_sampler = batch_by_size(
                range(len(dataset)), dataset.num_tokens, max_tokens=750*num_replicas,
                max_sentences=12*num_replicas, required_batch_size_multiple=num_replicas)
            if train:
                np.random.shuffle(batch_sampler)
            if num_replicas > 1:
                batch_sampler = [x[rank::num_replicas] for x in batch_sampler if len(x) % num_replicas == 0]
            dataloader = DataLoader(
                dataset, num_workers=cfg.num_workers, batch_sampler=batch_sampler,
                pin_memory=True, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(
                dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=train,
                pin_memory=True, collate_fn=collate_fn)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.train_config, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloder(self.val_config, train=False)


def _is_batch_full(batch, num_tokens, max_tokens, max_sentences):
    if len(batch) >= max_sentences or num_tokens >= max_tokens:
        return True
    else:
        return False


def batch_by_size(
        indices, dataset_num_tokens, max_tokens=None, max_sentences=None,
        required_batch_size_multiple=1):
    """
    Yield mini-batches of indices bucketed by size. Batches may contain
    sequences of different lengths.

    Args:
        indices (List[int]): ordered list of dataset indices
        dataset_num_tokens (callable): function that returns the number of tokens at
            a given index
        max_tokens (int, optional): max number of tokens in each batch
            (default: None).
        max_sentences (int, optional): max number of sentences in each
            batch (default: None).
        required_batch_size_multiple (int, optional): require batch size to
            be a multiple of N (default: 1).
    """
    max_tokens = max_tokens if max_tokens is not None else sys.maxsize
    max_sentences = max_sentences if max_sentences is not None else sys.maxsize
    bsz_mult = required_batch_size_multiple

    if isinstance(indices, types.GeneratorType):
        indices = np.fromiter(indices, dtype=np.int64, count=-1)

    sorted_index_num = sorted([(i, n) for i, n in zip(indices, dataset_num_tokens)],
                              key=lambda tup: tup[1])

    sample_len = 0
    sample_lens = []
    batch = []
    batches = []
    for idx, num_tokens in sorted_index_num:
        sample_lens.append(num_tokens)
        sample_len = max(sample_len, num_tokens)

        assert sample_len <= max_tokens, (
            "sentence at index {} of size {} exceeds max_tokens "
            "limit of {}!".format(idx, sample_len, max_tokens)
        )
        batch_num_tokens = (len(batch) + 1) * sample_len

        if _is_batch_full(batch, batch_num_tokens, max_tokens, max_sentences):
            mod_len = max(
                bsz_mult * (len(batch) // bsz_mult),
                len(batch) % bsz_mult,
            )
            batches.append(batch[:mod_len])
            batch = batch[mod_len:]
            sample_lens = sample_lens[mod_len:]
            sample_len = max(sample_lens) if len(sample_lens) > 0 else 0
        batch.append(idx)
    if len(batch) > 0:
        batches.append(batch)
    return batches


class VocosDataset(Dataset):
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
        return y[0], torch.tensor(start)


def load_ark_scp(scp_fp):
    ark_dict = kaldiio.load_scp(scp_fp, max_cache_fd=32)
    return ark_dict


class ArkDataset(torch.utils.data.Dataset):
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
            y = y[:, start : start + self.num_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.num_samples]
        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, sr, [["norm", f"{gain:.2f}"]])
        return y[0]


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

        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        assert torch.sum(durations) == num_frames, (
            f"{k}, sum durations {torch.sum(durations)}, num frames: {num_frames}, "
            f"start_phone_idx: {start_phone_idx}, start_frame: {start_frame}, durations: {durations}")
        return y[0], (token_ids, durations, start_phone_idx, start_frame)


class TTSDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "tts"
        assert cfg.hop_length is not None
        assert cfg.phoneset is not None
        assert cfg.padding is not None
        with open(cfg.filelist_path) as f:
            self.filelist = f.read().splitlines()
        self.num_tokens = [int(fl.split('|')[3]) for fl in self.filelist]
        self.sampling_rate = cfg.sampling_rate
        self.num_samples = cfg.num_samples
        self.train = train
        self.hop_length = cfg.hop_length
        self.padding = cfg.padding
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
            y, sr = torchaudio.load(audio_fp)
            if y.size(0) > 1:
                # mix to mono
                y = y.mean(dim=0, keepdim=True)
            if sr != self.sampling_rate:
                y = torchaudio.functional.resample(y, orig_freq=sr, new_freq=self.sampling_rate)
            up_durations = upsample_durations(durations, y.size(1), self.hop_length, self.padding)
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

        gain = np.random.uniform(-1, -6) if self.train else -3
        y, _ = torchaudio.sox_effects.apply_effects_tensor(y, self.sampling_rate, [["norm", f"{gain:.2f}"]])
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

        # get tokens
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

        # get context
        ctx_start = ctx_start_frame * self.hop_length
        ctx_end = (ctx_start_frame + ctx_n_frame) * self.hop_length
        y_ctx = y[:, ctx_start: ctx_end]
        ctx_n_frame = ctx_n_frame + 1 if self.padding == 'center' else ctx_n_frame
        ctx_start_phone_idx = torch.searchsorted(cs_durations, ctx_start_frame, right=True) if start_frame > 0 else 0
        ctx_end_phone_idx = torch.searchsorted(cs_durations, ctx_start_frame + ctx_n_frame, right=False)
        ctx_token_ids = token_ids[ctx_start_phone_idx: ctx_end_phone_idx + 1].detach().clone()
        ctx_durations = up_durations[ctx_start_phone_idx: ctx_end_phone_idx + 1].detach().clone()
        if ctx_end_phone_idx != ctx_start_phone_idx:
            first_num_frames = cs_durations[ctx_start_phone_idx] - ctx_start_frame
            last_num_frames = ctx_start_frame + ctx_n_frame - cs_durations[ctx_end_phone_idx - 1]
            ctx_durations[0] = first_num_frames
            ctx_durations[-1] = last_num_frames
        else:
            ctx_durations[0] = ctx_n_frame

        gain = np.random.uniform(-1, -6) if self.train else -3
        y_seg, _ = torchaudio.sox_effects.apply_effects_tensor(
            y_seg, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        y_ctx, _ = torchaudio.sox_effects.apply_effects_tensor(
            y_ctx, self.sampling_rate, [["norm", f"{gain:.2f}"]])
        assert torch.sum(seg_durations) == num_frames, (
            f"{k}, sum durations {torch.sum(seg_durations)}, num frames: {num_frames}, "
            f"start_phone_idx: {start_phone_idx}, start_frame: {start_frame}, durations: {seg_durations}")
        assert torch.sum(ctx_durations) == ctx_n_frame, (
            f"{k}, ctx sum durations {torch.sum(ctx_durations)}, ctx num_frames: {ctx_n_frame}")
        return y_seg[0], (seg_token_ids, seg_durations, start_phone_idx, start_frame,
                          y_ctx[0], ctx_n_frame, ctx_token_ids, ctx_durations)


class E2ETTSCtxDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        assert cfg.task == "e2e"
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
            if self._cache is not None:
                self._cache[k] = (y, token_ids)
        else:
            y, token_ids = self._cache[k]

        num_frames = self.num_samples // self.hop_length + (1 if self.padding == "center" else 0)
        y = y.detach().clone()[:, :y.size(1) // self.hop_length * self.hop_length]
        total_frames = y.size(1) // self.hop_length + (1 if self.padding == "center" else 0)
        token_ids = token_ids.detach().clone()

        if y.size(-1) < self.num_samples + self.hop_length + self.min_context * self.hop_length:
            half_frames = total_frames // 2
            lo_y = y[:, :half_frames * self.hop_length]
            hi_y = y[:, half_frames * self.hop_length:]
            repeats = np.ceil(
                (self.num_samples + self.hop_length) / min(lo_y.size(-1), hi_y.size(-1))).astype(np.int64)
            lo_y = lo_y.repeat(1, repeats)
            hi_y = hi_y.repeat(1, repeats)
            y = torch.cat([lo_y, hi_y], dim=-1)
            if np.random.uniform() < 0.5:
                start = 0
                start_frame = 0
                end_frame = num_frames
                y_seg = y[:, :self.num_samples]
                ctx_start_frame = half_frames
            else:
                start_frame = half_frames
                start = start_frame * self.hop_length
                end_frame = num_frames + start_frame
                y_seg = y[:, start: start + self.num_samples]
                ctx_start_frame = 0
            ctx_n_frame = self.min_context
        else:
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

        # get context
        ctx_start = ctx_start_frame * self.hop_length
        ctx_end = (ctx_start_frame + ctx_n_frame) * self.hop_length
        y_ctx = y[:, ctx_start: ctx_end]
        ctx_n_frame = ctx_n_frame + 1 if self.padding == 'center' else ctx_n_frame
        return y_seg[0], (token_ids, start_frame, y_ctx[0],
                          torch.tensor([len(token_ids), ctx_n_frame], dtype=torch.long))


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


def tts_collate_segment(data):
    y = torch.stack([d[0] for d in data])
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    max_num = max(num_phones)
    token_ids = torch.zeros([len(data), max_num], dtype=torch.long)
    durations = torch.zeros([len(data), max_num], dtype=torch.long)
    for i, (ti, d, _, _) in enumerate(token_info):
        token_ids[i, :ti.size(0)] = ti
        durations[i, :ti.size(0)] = d
    start_phone_idx = torch.tensor([ti[2] for ti in token_info])
    start_frame = torch.tensor([ti[3] for ti in token_info])
    return y, (token_ids, durations, start_phone_idx, start_frame)


def tts_collate(data):
    y_lens = [d[0].size(0) for d in data]
    max_y_len = max(y_lens)
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    num_frames = [torch.sum(ti[1]) for ti in token_info]
    max_num_phone = max(num_phones) + 1
    max_num_frame = max(num_frames)
    y = torch.zeros([len(data), max_y_len], dtype=torch.float)
    token_ids = torch.zeros([len(data), max_num_phone], dtype=torch.long)
    durations = torch.zeros([len(data), max_num_phone], dtype=torch.long)
    for i, (ti, d, _, _) in enumerate(token_info):
        y[i, :y_lens[i]] = data[i][0]
        token_ids[i, :ti.size(0)] = ti
        durations[i, :ti.size(0)] = d
        token_ids[i, -1] = 0
        durations[i, -1] = max_num_frame - torch.sum(d)
    start_phone_idx = torch.tensor([ti[2] for ti in token_info])
    start_frame = torch.tensor([ti[3] for ti in token_info])
    return y, (token_ids, durations, start_phone_idx, start_frame)


def tts_ctx_collate_segment(data):
    y = torch.stack([d[0] for d in data])
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    max_num = max(num_phones)
    y_ctx = [ti[4] for ti in token_info]
    max_ctx_len = max([y.size(0) for y in y_ctx])
    num_ctx_phones = [ti[6].size(0) for ti in token_info]
    max_ctx_num = max(num_ctx_phones)
    token_ids = torch.zeros([len(data), max_num], dtype=torch.long)
    durations = torch.zeros([len(data), max_num], dtype=torch.long)
    y_ctx_pad = torch.zeros([len(data), max_ctx_len], dtype=torch.float)
    ctx_token_ids = torch.zeros([len(data), max_ctx_num], dtype=torch.long)
    ctx_durations = torch.zeros([len(data), max_ctx_num], dtype=torch.long)
    for i, (ti, d, _, _, ctx, _, cti, cd) in enumerate(token_info):
        token_ids[i, :ti.size(0)] = ti
        durations[i, :ti.size(0)] = d
        y_ctx_pad[i, :ctx.size(0)] = ctx
        ctx_token_ids[i, :cti.size(0)] = cti
        ctx_durations[i, :cd.size(0)] = cd
    start_phone_idx = torch.tensor([ti[2] for ti in token_info])
    start_frame = torch.tensor([ti[3] for ti in token_info])
    ctx_n_frame = torch.tensor([ti[5] for ti in token_info])
    return y, (token_ids, durations, start_phone_idx, start_frame,
               y_ctx_pad, ctx_n_frame, ctx_token_ids, ctx_durations)


def e2e_tts_ctx_collate_segment(data):
    y = torch.stack([d[0] for d in data])
    token_info = [d[1] for d in data]
    num_phones = [ti[0].size(0) for ti in token_info]
    max_num = max(num_phones)
    y_ctx = [ti[2] for ti in token_info]
    max_ctx_len = max([y.size(0) for y in y_ctx])
    token_ids = torch.zeros([len(data), max_num], dtype=torch.long)
    y_ctx_pad = torch.zeros([len(data), max_ctx_len], dtype=torch.float)
    for i, (ti, _, ctx, _) in enumerate(token_info):
        token_ids[i, :ti.size(0)] = ti
        y_ctx_pad[i, :ctx.size(0)] = ctx
    start_frame = torch.tensor([ti[1] for ti in token_info])
    ctx_n_frame = torch.stack([ti[3] for ti in token_info])
    return y, (token_ids, start_frame, y_ctx_pad, ctx_n_frame)


def dur_collate(data):
    num_phones = [len(d[0]) for d in data]
    max_num_phone = max(num_phones)
    token_ids = torch.zeros([len(data), max_num_phone], dtype=torch.long)
    durations = torch.zeros([len(data), 1, max_num_phone], dtype=torch.float)
    for i, (ti, d) in enumerate(data):
        token_ids[i, :ti.size(0)] = ti
        durations[i, 0, :ti.size(0)] = d
    return token_ids, durations


if __name__ == "__main__":
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
    tts_ctx_collate_segment(data)
    for d in dataset:
        print(d[0].shape)

    # dur_cfg = DataConfig(
    #     filelist_path="/data/corpus/LJSpeech-1.1/synta_filelist.valid",
    #     batch_size=8,
    #     num_workers=0,
    #     cache=True,
    #     task="dur",
    #     phoneset="/data/corpus/LJSpeech-1.1/synta_phoneset.th",
    # )
    # dur_dataset = DurDataset(dur_cfg, train=False)
    # print(dur_dataset[0])
    # data = [dur_dataset[0], dur_dataset[1]]
    # dur_collate(data)
