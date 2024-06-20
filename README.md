# RFWave: Multi-band Rectified Flow for Audio Waveform Reconstruction.
[Audio samples](https://rfwave-demo.github.io/rfwave/) | [Paper](https://arxiv.org/abs/2403.05010)

### TL;DR
RFWave utilizes multi-band Rectified Flow for high-fidelity audio waveform reconstruction from either Mel-spectrograms or discrete tokens, and boasts a generation speed 97 times faster than real-time on a GPU.

### Abstract
Recent advancements in generative modeling have significantly enhanced the reconstruction of audio waveforms from various representations. While diffusion models are adept at this task, they are hindered by latency issues due to their operation at the individual sample point level and the need for numerous sampling steps. In this study, we introduce RFWave, a cutting-edge multi-band Rectified Flow approach designed to reconstruct high-fidelity audio waveforms from Mel-spectrograms or discrete tokens. RFWave uniquely generates complex spectrograms and operates at the frame level, processing all subbands simultaneously to boost efficiency. Leveraging Rectified Flow, which targets a flat transport trajectory, RFWave achieves reconstruction with just 10 sampling steps. Our empirical evaluations show that RFWave not only provides outstanding reconstruction quality but also offers vastly superior computational efficiency, enabling audio generation at speeds up to 97 times faster than real-time on a GPU. An online demonstration is available at: https://rfwave-demo.github.io/rfwave/.

<p align="middle">
    <br>
    <img src="assets/rfwave.jpeg" height="300" width="700"/>
    <img src="assets/spec.jpeg" height="300" width="700"/>
    <br>
</p>


## Usage

### Setup
1. Install the requirements.
```
sudo apt-get update
sudo apt-get install sox libsox-fmt-all libsox-dev
conda create -n rfwave python=3.10
conda activate rfwave
pip install -r requirements.txt
```
2. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/)
3. Update the wav paths in filelists `sed -i -- 's,LJSPEECH_PATH,ljs_dataset_folder,g' LJSpeech/*.filelist`
4. Update the `filelist_path` in configs/*.yaml.

### Vocoder
1. Train a vocoder `python3 train.py -c configs/rfwave.yaml`
2. Test a trained vocoder with `inference_voc.py`
### Encodec Decoder
1. Train an Encodec Decoder `python3 train.py -c configs/rfwave-encodec.yaml`
### Text to Speech
1. Download the [alignment](https://drive.google.com/file/d/1WfErAxKqMluQU3vupWS6VB6NdehXwCKM/view) from the [SyntaSpeech repo](https://github.com/yerfor/SyntaSpeech)
2. Convert the alignments and build a phoneset with `scripts/ljspeech_synta.py`
3. Modify the `filelist_path` and `phoneset` path in `configs/rfwave-dur.yaml` and `configs/rfwave-tts-ctx.yaml`
4. Train a duration model `python3 train.py -c configs/rfwave-dur.yaml`
5. Train an acoustic model `python3 train.py -c configs/rfwave-tts-ctx.yaml`
6. Test the trained model with `inference_tts.py`

## Pre-trained models

`python3 inference_voc.py --model_dir MODEL_DIR --wav_dir WAV_DIR --save_dir SAVE_DIR`

[rfwave-libritts-24k](https://drive.google.com/file/d/1IQNXAAVRTtr9P8Gc-CoPeRIJ_l_O4y38/view?usp=sharing)

[rfwave-jamendo-44.1k](https://drive.google.com/file/d/1yM0BWFrXvuwb2SZvyPOBr6Wgto0sjdqh/view?usp=sharing)

[rfwave-encodec-24k](https://drive.google.com/file/d/1gUpkpJPIs-9wKoLIhX8ZHfl5PhMNJdDb/view?usp=sharing)

## Test set
The test set for reconstructing waveform form EnCodec tokens.

[audio_reconstruct_universal_testset](https://drive.google.com/file/d/1WjRRfD1yJSjEA3xfC8-635ugpLvnRK0f/view?usp=sharing)

## Thanks

This repository uses code from [Vocos](https://github.com/gemelo-ai/vocos), [audiocraft](https://github.com/facebookresearch/audiocraft) 
