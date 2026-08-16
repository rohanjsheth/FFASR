# FFASR simulation

This repository renders reverberant speech mixed with one or more spatially
rendered noise sources. The scene renderer operates on mono, floating-point
waveforms. All source waveforms and room impulse responses (RIRs) must already
be resampled to the same sample rate before they are passed to `make_scene`.

## Project layout

- `audio_utils/audio_mixing.py` contains the DSP primitives.
- `audio_utils/make_scene.py` composes one simulated acoustic scene.
- `audio_utils/audio_types.py` defines the scene input types.
- `data_utils.py` handles dataset indexing, recipe sampling, audio loading, and
  deterministic rendering from a recipe.
- `SceneDataset.py` selects clean or simulated examples for PyTorch.
- `data_collator.py` converts rendered examples into padded Qwen3-ASR inputs.
- `scripts/render_samples.py` is a manual inspection tool that writes example
  waveforms and spectrograms.
- `scripts/smoke_test_dataloader.py` streams two examples and validates the
  dataset-to-processor path.
- `scripts/evaluate_snr_wer.py` measures paired clean and simulated WER across
  high-, mid-, and low-SNR bands.

## Setup

The project targets Python 3.11 or newer and uses `uv` for environment and
lockfile management:

```bash
uv sync
```

To inspect the sample-rendering command without downloading data:

```bash
uv run ffasr-render-samples --help
```

To exercise one clean and one simulated example through the real Qwen3-ASR
processor without downloading the complete source datasets:

```bash
uv run ffasr-smoke-test
```

Pass `--with-model` to add a full-model forward-loss check, or use
`--train-steps 1` to verify frozen-module gradients and one optimizer update on
a CUDA machine. See the [Colab GPU runbook](docs/colab.md) for the complete
hosted workflow.

On a CUDA machine, run a small paired WER evaluation before scaling to the
default 50 utterances per band:

```bash
uv run ffasr-evaluate-snr \
  --cache-dir /content/hf-cache \
  --samples-per-band 5 \
  --batch-size 2
```

Each selected utterance is evaluated clean and in all three simulated bands.
The simulated versions share their room, receiver, RIRs, noise recordings,
offsets, and random-noise seed; only SNR changes. Bands use achieved
`final_snr_db`: high is above 14 dB, mid is 8--12 dB, and low is below 6 dB.
The command writes auditable per-utterance predictions and a corpus-level WER
summary under `results/snr_wer/`.

See [Evaluation baselines](docs/evaluation.md) for the recorded pretrained
Qwen3-ASR results, the FFASR comparison, and the next validation gate.

## Simulation process

### Inputs and notation

For one scene, the renderer receives:

| Input | Notation | Meaning |
| --- | --- | --- |
| `speech` | $s[n]$ | Dry speech waveform |
| `speech_rir` | $h_s[n]$ | RIR from the speech position to the receiver |
| `speech_distance` | $d_s$ | Speech-to-receiver direct-path length in metres |
| `noise_stems[i]` | $x_i[n]$ | Dry waveform for noise source $i$ |
| `noise_rirs[i]` | $h_i[n]$ | RIR from noise source $i$ to the receiver |
| `noise_distances[i]` | $d_i$ | Noise-source-to-receiver direct-path length in metres |
| `noise_offsets_ms[i]` | $o_i$ | Start of the noise source's pre-roll segment, in milliseconds |
| `sr` | $f_s$ | Sample rate in samples per second |
| `target_snr_db` | $T$ | Desired SNR of speech relative to the aggregate noise field |
| `pink_db` | $B$ | Pink-noise level in dB below active-speech RMS |
| `rng_seed` |  | Seed used to reproduce the pink-noise waveform |

There must be at least one noise source. The noise waveform, RIR, distance, and
offset lists must all have the same length.

The high-level signal flow is:

```text
speech:       dry source -> align RIR -> full convolution -> room speech
each noise:   dry source -> RMS reference -> offset/loop -> pre-roll convolution -> room noise
noise field:  room noises -> sum -> aggregate SNR gain -> scaled noise
pink bed:     seeded pink noise -> speech-relative gain -> scaled pink noise
output:       room speech + scaled noise + pink bed -> shared peak guard -> final mixture
```

### 0. Prepare inputs at a common sample rate

File loading and resampling happen outside `make_scene`. Audio is decoded as a
floating-point waveform. When an input sample rate $f_{\mathrm{in}}$ differs
from the scene sample rate $f_s$, the current loading script uses polyphase
resampling. Writing the reduced rational resampling ratio as

$$
\frac{f_s}{f_{\mathrm{in}}} = \frac{U}{D},
$$

polyphase resampling conceptually upsamples by $U$, applies an anti-aliasing
low-pass filter, and downsamples by $D$. Speech, noise, and RIRs are all
converted to the same $f_s$ before scene rendering. The current example uses
16 kHz, while the core simulation functions operate on the supplied sample
rate.

The caller is also responsible for selecting the speech utterance, room and
source RIRs, target SNR, pink-noise level, random seed, and per-noise pre-roll
offsets. These realized values become fixed inputs to the transformations
below.

### 1. Locate and trim each RIR

The renderer first estimates the direct-path arrival in each RIR. Given a
source-to-receiver distance $d$, the expected direct arrival is

$$
n_{\mathrm{expected}} = \operatorname{round}\left(\frac{d}{343} f_s\right),
$$

where 343 m/s is the assumed speed of sound. A local search window of 1 ms on
either side of this expected position is used:

$$
W = \operatorname{round}(0.001 f_s).
$$

Within the valid portion of
$[n_{\mathrm{expected}}-W,\ n_{\mathrm{expected}}+W]$, the sample with the
largest absolute amplitude is selected as the direct-path index
$n_{\mathrm{direct}}$.

The RIR is then trimmed so that it begins up to 1 ms before that direct-path
sample:

$$
n_{\mathrm{trim}} = \max\left(0,
n_{\mathrm{direct}}-\operatorname{round}(0.001 f_s)\right),
$$

$$
h'[n] = h[n+n_{\mathrm{trim}}].
$$

Both speech and noise convolution use the corresponding trimmed RIR. The
distance argument is used for direct-arrival estimation and RIR trimming; the
renderer does not separately multiply a signal by $1/d$. Trimming also removes
most of the absolute propagation delay, so the rendered sources do not preserve
their original relative time-of-flight delays.

### 2. Render reverberant speech

The dry speech is convolved with its trimmed RIR using full FFT convolution:

$$
s_r[n] = (s * h'_s)[n]
       = \sum_k s[k]h'_s[n-k].
$$

If the dry speech contains $N_s$ samples and the trimmed speech RIR contains
$M_s$ samples, the reverberant speech length is

$$
L = N_s + M_s - 1.
$$

The speech convolution is not cropped, so $s_r[n]$ includes the complete RIR
tail. This length $L$ becomes the required length of every rendered noise
source and of the final mixture. Speech is not RMS-normalized inside
`make_scene`.

### 3. Put each dry noise source on a common RMS reference

Each complete dry noise clip is independently normalized before it is placed in
the room. For noise source $i$ with $N_i$ samples,

$$
r_i = \sqrt{\frac{1}{N_i}\sum_{n=0}^{N_i-1}x_i[n]^2},
$$

$$
\hat{x}_i[n] = \frac{x_i[n]}{r_i + \epsilon},
\qquad \epsilon = 10^{-12}.
$$

Thus each dry clip has approximately unit RMS before convolution. The
normalization is measured over the complete dry clip, not only over the segment
used in the scene. There is currently no additional class-specific or
source-specific level offset; after dry normalization, relative received levels
are determined by the RIR waveforms.

### 4. Select, loop, and pre-roll each noise source

The supplied offset is converted from milliseconds to samples:

$$
q_i = \operatorname{round}\left(\frac{o_i f_s}{1000}\right).
$$

The offset denotes the beginning of the source segment used as convolution
pre-roll. Offset selection happens outside `make_scene`; the renderer consumes
and records the realized offsets. The RNG passed to `make_scene` is used only
to generate the pink-noise waveform.

For a raw noise RIR containing $M_i$ samples, the renderer reserves

$$
P_i = M_i - 1
$$

samples of warm-up history. It therefore requests a dry noise segment of length

$$
Q_i = P_i + L.
$$

To obtain this segment, the normalized noise clip is tiled. The number of
copies is

$$
R_i = \left\lceil\frac{q_i + Q_i}{N_i}\right\rceil,
$$

and the requested segment is

$$
\tilde{x}_i =
\operatorname{tile}(\hat{x}_i, R_i)[q_i:q_i+Q_i].
$$

This produces exactly $Q_i$ samples while allowing the segment to cross the end
of the original clip. A short clip can repeat multiple times. No cross-fade is
currently applied at loop boundaries.

The segment is then convolved with the trimmed noise RIR:

$$
v_i[n] = (\tilde{x}_i * h'_i)[n].
$$

The first $P_i$ output samples are discarded, and the following $L$ samples are
kept:

$$
n_i[n] = v_i[n + P_i], \qquad 0 \leq n < L.
$$

An RIR of length $M$ needs $M-1$ preceding input samples for full convolution
overlap. The implementation uses the original, untrimmed RIR length for
$P_i$, which is at least as long as the required warm-up for the trimmed RIR.
Consequently, the first retained noise sample is rendered after the finite RIR
has reached full overlap rather than representing a noise source that switched
on at scene time zero.

### 5. Form the aggregate room-noise field

All $K$ rendered noise waveforms have length $L$ and are summed sample by
sample:

$$
n_r[n] = \sum_{i=1}^{K} n_i[n].
$$

No source is independently adjusted to a target SNR. Any relative level
differences created by the RIRs remain present in this sum. One scene-level gain
is applied to the complete noise field later.

### 6. Build the active-speech mask

SNR is measured only during estimated speech activity. The reverberant speech
$s_r[n]$ is split into non-overlapping 25 ms frames. The frame length is

$$
F = \operatorname{int}\left(0.025 f_s\right).
$$

For frame $j$, the frame energy is its mean-square amplitude:

$$
E_j = \frac{1}{|\mathcal{F}_j|}
      \sum_{n \in \mathcal{F}_j}s_r[n]^2.
$$

The activity cutoff is 1% of the peak frame energy:

$$
E_{\mathrm{cutoff}} = 0.01\left(\max_j E_j + \epsilon\right).
$$

Every sample in frame $j$ is marked active when
$E_j \geq E_{\mathrm{cutoff}}$. The resulting Boolean sample mask is denoted
$m[n]$. If the mask contains no active samples, power measurement raises an
error.

### 7. Scale the aggregate noise to the target SNR

Speech and aggregate-noise powers are measured over the same active-speech
samples:

$$
P_s = \operatorname{mean}_{n:m[n]} s_r[n]^2,
$$

$$
P_n = \operatorname{mean}_{n:m[n]} n_r[n]^2 + \epsilon.
$$

The initial SNR is

$$
S_0 = 10\log_{10}\left(\frac{P_s}{P_n}\right).
$$

The renderer computes one linear amplitude gain for the aggregate noise:

$$
g = 10^{(S_0-T)/20}.
$$

For nondegenerate signals, ignoring only the numerical $\epsilon$ floor, this
follows from the fact that multiplying noise amplitude by $g$ multiplies its
power by $g^2$:

$$
10\log_{10}\left(\frac{P_s}{g^2P_n}\right)
= S_0 - 20\log_{10}(g) = T.
$$

The scaled noise is

$$
n_g[n] = g n_r[n].
$$

Because $g$ is applied after the sources are summed, every noise source receives
the same final gain and their relative contributions are preserved.

### 8. Add the pink-noise bed and create the mixture

The renderer generates a unit-standard-deviation pink-noise waveform $z_p[n]$
from the supplied RNG. Given the configured pink-noise level $B$, its amplitude
gain relative to active-speech RMS is

$$
g_p = \sqrt{P_s}10^{-B/20}.
$$

The complete noise component is

$$
n_c[n] = n_g[n] + g_p z_p[n],
$$

and the mixture is

$$
y[n] = s_r[n] + n_c[n].
$$

At this point, `mixture`, `speech`, and `noise` refer respectively to $y[n]$,
$s_r[n]$, and $n_c[n]$. All three have length $L$. The configured target SNR
is applied to the spatially rendered aggregate noise $n_g[n]$ before pink noise
is added. Consequently, `final_snr_db`, which includes both noise components,
can be lower than `target_snr_db`.

### 9. Apply the shared peak guard

The mixture peak is

$$
p = \max_n |y[n]|.
$$

With the default maximum peak $p_{\max}=0.99$, the shared clipping-prevention
scale is

$$
a =
\begin{cases}
0.99/p, & p > 0.99,\\
1, & p \leq 0.99.
\end{cases}
$$

The same scale is applied to the mixture and both component signals:

$$
y_f[n] = a y[n], \qquad
s_f[n] = a s_r[n], \qquad
n_f[n] = a n_c[n].
$$

Using one shared scale preserves the component sum
$y_f[n]=s_f[n]+n_f[n]$ and preserves the underlying SNR because both component
powers are multiplied by $a^2$ (apart from the numerical $\epsilon$ floor used
when measuring SNR). The `clipped` metadata field records whether $a<1$, and
`clipping_scale` records $a$. This is a peak guard only; the current pipeline
does not unconditionally normalize the final mixture to a target RMS.

### 10. Measure the final SNR and return the scene

The final SNR is measured from $s_f[n]$ and $n_f[n]$ using the original
active-speech mask:

$$
S_{\mathrm{final}} = 10\log_{10}\left(
\frac{\operatorname{mean}_{n:m[n]}s_f[n]^2}
     {\operatorname{mean}_{n:m[n]}n_f[n]^2 + \epsilon}
\right).
$$

`make_scene` returns the final mixture $y_f[n]$ and the following metadata:

| Field | Meaning |
| --- | --- |
| `sr` | Scene sample rate |
| `target_snr_db` | Requested active-speech SNR for the spatial aggregate noise $n_g[n]$ |
| `pink_db` | Pink-noise level $B$ below active-speech RMS |
| `rng_seed` | Seed used to reproduce the pink-noise waveform |
| `final_snr_db` | Measured SNR against spatial plus pink noise after the shared peak guard |
| `speech_drr_db` | DRR of the speech source path, in dB |
| `noises` | Ordered list of per-noise objects containing `drr_db` and `offset_ms` |
| `clipping_scale` | Shared scale $a$; `1.0` means no attenuation |
| `clipped` | Whether the mixture peak exceeded `0.99` before scaling |

The returned waveform has the same length as the full reverberant speech,
including its RIR tail.

### DRR calculation

`audio_utils.audio_mixing.drr_db` calculates the direct-to-reverberant ratio
(DRR) for an RIR. `make_scene` records this as `speech_drr_db` for the speech
path and as `noises[i].drr_db` for each noise path. The direct-path index is
estimated as described above. A window with a half-width of 1.25 ms is placed
around that index, giving an approximately 2.5 ms total direct-path window.

If $\mathcal{D}$ is that window, then

$$
E_{\mathrm{direct}} = \sum_{n\in\mathcal{D}}h[n]^2,
$$

$$
E_{\mathrm{total}} = \sum_n h[n]^2,
$$

$$
E_{\mathrm{reverberant}} =
\max(E_{\mathrm{total}}-E_{\mathrm{direct}}, 0),
$$

and

$$
\operatorname{DRR}_{\mathrm{dB}} = 10\log_{10}\left(
\frac{E_{\mathrm{direct}}+\epsilon}
     {E_{\mathrm{reverberant}}+\epsilon}
\right).
$$
