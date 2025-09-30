# Copyright (c) aiOla.ai
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Union

import torch
import torchaudio
from flow_matching.path.scheduler import ConvexScheduler, PolynomialConvexScheduler
from transformers import WhisperProcessor

from drax.flow.solver import DraxMixtureDiscreteEulerSolver
from drax.flow.source_dist import SourceDistribution, UniformSourceDistribution
from drax.model.drax_model import Drax


@dataclass(frozen=True)
class TranscriptionSample:
    transcript: str
    audio_path: str
    language: str


class TranscribeResult(Sequence[TranscriptionSample]):
    def __init__(self, transcript: list[str], audio_path: list[str], language: list[str]) -> None:
        assert len(transcript) == len(audio_path) == len(language), "transcript, audio_path, and language must have the same length"
        self.transcript = transcript
        self.audio_path = audio_path
        self.language = language

    def __len__(self) -> int:
        return len(self.transcript)

    def __getitem__(self, idx) -> Union[TranscriptionSample, "TranscribeResult"]:
        if isinstance(idx, slice):
            return TranscribeResult(
                transcript=self.transcript[idx],
                audio_path=self.audio_path[idx],
                language=self.language[idx],
            )
        return TranscriptionSample(
            transcript=self.transcript[idx],
            audio_path=self.audio_path[idx],
            language=self.language[idx],
        )

    def __iter__(self) -> Iterator[TranscriptionSample]:
        for transcript_text, path_str, language_code in zip(self.transcript, self.audio_path, self.language, strict=False):
            yield TranscriptionSample(transcript=transcript_text, audio_path=path_str, language=language_code)


class Transcriber:
    """Transcriber class for transcribing audio files with Drax."""

    def __init__(
        self,
        model_path: str,
        source_distribution: SourceDistribution = None,
        scheduler: ConvexScheduler = None,
        device: str = None,
    ):
        """Transcriber class for transcribing audio files with Drax.

        Args:
            model_path: path to the model checkpoint
            source_distribution: source distribution for the decoder
            scheduler: scheduler for the decoder
            device: device to use for the model

        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = Drax.from_pretrained(model_path, device=device)
        self.model.eval()
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
        self.target_sr = 16000
        self.support_language_codes = self.model.decoder.config.support_language_codes

        if source_distribution is None:
            source_distribution = UniformSourceDistribution(vocab_size=self.model.decoder.vocab_size)
        assert isinstance(source_distribution, SourceDistribution), "source_distribution must be a SourceDistribution"

        if scheduler is None:
            scheduler = PolynomialConvexScheduler(n=1.0)
        assert isinstance(scheduler, ConvexScheduler), "scheduler must be a ConvexScheduler"

        self.source_distribution = source_distribution
        self.solver = DraxMixtureDiscreteEulerSolver(
            model=self.model,
            scheduler=scheduler,
            vocabulary_size=self.model.decoder.vocab_size,
        )
        self.startoftranscript_id = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")

    def _preprocess_audio(self, audio_path: str):
        # todo: support cases where audio is already provided as features
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)
            sr = self.target_sr
        audio = audio.squeeze(0)
        audio = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        audio = audio.input_features
        audio = audio.to(self.device)
        audio_embeddings = self.model.encode_audio(audio)
        audio_cache = self.model.build_audio_cache(audio_embeddings)
        return audio_cache

    def _get_prompt(self, language: str):
        task = "transcribe"

        forced = self.processor.tokenizer.get_decoder_prompt_ids(
            language=language,
            task=task,
            no_timestamps=True,  # set False if you want timestamp tokens enabled
        )
        forced_ids = torch.tensor([self.startoftranscript_id] + [tok_id for _, tok_id in forced], device=self.device)
        return forced_ids

    def _get_input_ids(self, language: str, seq_length: int):
        seq_length = seq_length or self.model.decoder.config.length
        x_init = self.source_distribution.sample((1, seq_length), device=self.device)
        forced_ids = self._get_prompt(language)
        x_init[:, : len(forced_ids)] = forced_ids
        preserve_mask = torch.zeros((1, seq_length), device=self.device, dtype=torch.bool)
        preserve_mask[:, : len(forced_ids)] = True
        return x_init, preserve_mask

    def _build_batch(self, audio_path: list[str], language: list[str], seq_length: int):
        """Build a batch of input ids, preserve masks, and audio caches."""
        x_init = []
        preserve_mask = []
        audio_caches = []
        for path, lang in zip(audio_path, language, strict=False):
            audio_cache = self._preprocess_audio(path)
            x_init_i, preserve_mask_i = self._get_input_ids(lang, seq_length)
            x_init.append(x_init_i)
            preserve_mask.append(preserve_mask_i)
            audio_caches.append(audio_cache)

        # stack
        x_init = torch.cat(x_init, dim=0)
        preserve_mask = torch.cat(preserve_mask, dim=0)
        audio_caches = {k: torch.cat([audio_cache[k] for audio_cache in audio_caches], dim=0) for k in audio_caches[0].keys()}
        return x_init, preserve_mask, audio_caches

    def _validate_input(self, audio_path: str | list[str], language: str | list[str]):
        assert len(audio_path) == len(language), "audio_path and language must have the same length"
        for lang in language:
            assert lang in self.support_language_codes, f"Language {lang} is not supported, supported languages are {self.support_language_codes}."
        return audio_path, language

    def transcribe(
        self,
        audio_path: str | list[str],
        language: str | list[str],
        temperature: float = 1e-1,
        sampling_steps: int = 16,
        seq_length: int = None,
        verbose: bool = False,
    ) -> TranscribeResult:
        """Transcribe an audio file or a list of audio files.

        Args:
            audio_path: path to the audio file or a list of audio files
            language: language of the audio file or a list of audio files
            temperature: temperature for the sampling
            sampling_steps: number of sampling steps
            seq_length: sequence length for the sampling
            verbose: whether to print the progress

        Returns:
            TranscribeResult: a list of transcription samples

        """
        if isinstance(audio_path, str):
            audio_path = [audio_path]
        if isinstance(language, str):
            language = [language]

        self._validate_input(audio_path, language)

        x_init, preserve_mask, audio_cache = self._build_batch(audio_path, language, seq_length)
        samples = self.solver.sample(
            x_init=x_init,
            step_size=1 / sampling_steps if sampling_steps > 1 else 1 - 1e-8,
            **audio_cache,
            preserve_mask=preserve_mask,
            temperature=temperature,
            verbose=verbose,
        )

        decoded_text = self.processor.batch_decode(samples, skip_special_tokens=True)
        return TranscribeResult(transcript=decoded_text, audio_path=audio_path, language=language)
