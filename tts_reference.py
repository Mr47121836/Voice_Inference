import re
import sys
sys.path.append("MoeGoe")
from scipy.io.wavfile import write
from MoeGoe import utils
from MoeGoe import commons
from MoeGoe.text import text_to_sequence
from MoeGoe import models
from torch import no_grad, LongTensor

def ex_print(text, escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)

def get_label(text, label):
    if f'[{label}]' in text:
        return True, text.replace(f'[{label}]', '')
    else:
        return False, text

def get_label_value(text, label, default, warning_name='value'):
    value = re.search(rf'\[{label}=(.+?)\]', text)
    if value:
        try:
            text = re.sub(rf'\[{label}=(.+?)\]', '', text, 1)
            value = float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value = default
    return value, text


def get_text(text, hps, cleaned=False):

    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def print_speakers(speakers, ):
    if len(speakers) > 100:
        return
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name)


def TTS(model_path,json_path,text):

    model = model_path
    config = json_path

    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']

    net_g_ms = models.SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval()

    utils.load_checkpoint(model, net_g_ms)

    print("text:"+text)

    length_scale, text = get_label_value(
        text, 'LENGTH', 1, 'length scale')
    noise_scale, text = get_label_value(
        text, 'NOISE', 0.667, 'noise scale')
    noise_scale_w, text = get_label_value(
        text, 'NOISEW', 0.8, 'deviation of noise')
    cleaned, text = get_label(text, 'CLEANED')

    stn_tst = get_text(text, hps_ms, cleaned=cleaned)

    print_speakers(speakers)

    speaker_id = 0

    out_path = "output.wav"

    with no_grad():
        x_tst = stn_tst.unsqueeze(0)
        x_tst_lengths = LongTensor([stn_tst.size(0)])
        sid = LongTensor([speaker_id])
        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale,
                               noise_scale_w=noise_scale_w, length_scale=length_scale)[0][
            0, 0].data.cpu().float().numpy()
    #write(out_path, hps_ms.data.sampling_rate, audio)
    print('Successfully saved!')
    return hps_ms.data.sampling_rate,audio
