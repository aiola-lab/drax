import argparse

from drax import Transcriber


def main(args):
    """Generate transcription for an audio file."""
    transcriber = Transcriber(args.model_path)

    result = transcriber.transcribe(
        audio_path=args.audio_path,
        language=args.language,
        temperature=args.temperature,
        sampling_steps=args.sampling_steps,
        seq_length=args.seq_length,
    )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to the audio file")
    parser.add_argument("--language", type=str, required=True, help="Language of the audio file")
    parser.add_argument("--temperature", type=float, default=1e-1, help="Temperature for the sampling")
    parser.add_argument("--sampling_steps", type=int, default=16, help="Number of sampling steps (NFE steps)")
    parser.add_argument("--seq_length", type=int, default=128, help="Sequence length for the decoder")
    args = parser.parse_args()
    result = main(args)
    print(result[0])
