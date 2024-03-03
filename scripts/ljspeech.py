import torch
import torchaudio
import tqdm

from argparse import ArgumentParser
from pathlib import Path
from seamless_communication.models.aligner.alignment_extractor import AlignmentExtractor

extractor = AlignmentExtractor(
    aligner_model_name_or_card="nar_t2u_aligner",
    unit_extractor_model_name_or_card="xlsr2_1b_v2",
    unit_extractor_output_layer=35,
    unit_extractor_kmeans_model_uri="https://dl.fbaipublicfiles.com/seamlessM4T/models/unit_extraction/kmeans_10k.npy",
    device=torch.device('cuda')
)


def load_meta(meta_csv, use_normalized):
    meta_lines = [l.strip() for l in Path(meta_csv).open()]
    meta_dict = {}
    for line in meta_lines:
        k, unt, nt = line.split("|")
        if use_normalized:
            meta_dict[k] = nt.lower()
        else:
            meta_dict[k] = unt.lower()
    return meta_dict


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--use_normalized", action="store_true")

    args = parser.parse_args()
    wav_dir = Path(args.data_dir) / "wavs"
    meta_csv = Path(args.data_dir) / "metadata.csv"
    save_dir = Path(args.data_dir) / "seamless_alignments"
    save_dir.mkdir(exist_ok=True)
    phoneset_fp = Path(args.data_dir) / "phoneset.th"

    meta = load_meta(meta_csv, args.use_normalized)
    phoneset = set()
    for k, trans in tqdm.tqdm(list(meta.items())):
        wav_path = str(wav_dir / f"{k}.wav")
        save_path = save_dir / f"{k}.th"
        res = extractor.extract_alignment(
            wav_path, trans, plot=False, add_trailing_silence=True)
        if len(res) == 4:
            alignment_durations, _, tokenized_text_tokens, probs = res
            tokenized_text_tokens = [str(t) for t in tokenized_text_tokens]
            torch.save({"durations": alignment_durations[0], "tokens": tokenized_text_tokens,
                        "probs": probs}, save_path)
        else:
            alignment_durations, _, tokenized_text_tokens = res
            tokenized_text_tokens = [str(t) for t in tokenized_text_tokens]
            torch.save({"durations": alignment_durations[0], "tokens": tokenized_text_tokens}, save_path)
        phoneset.update(tokenized_text_tokens)
    phoneset = sorted(list(phoneset))
    torch.save(phoneset, phoneset_fp)
