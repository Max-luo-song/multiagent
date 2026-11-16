"""Convert an existing Nexus prediction to a validated 3DGS trajectory bundle."""
import argparse
import hashlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nexus_3dgs.bundle import export_arrays, read_run, save_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--prediction', type=Path)
    parser.add_argument('--task', choices=('sampled_tensor', 'behavior_pred', 'intent_conditioned_behavior_pred'), default='sampled_tensor')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--current-index', type=int, default=4)
    parser.add_argument('--dt', type=float, default=.5)
    parser.add_argument('--batch-index', type=int, default=0)
    args = parser.parse_args()
    import torch
    context = read_run(args.run_dir, args.prediction)
    context['prediction_sha256'] = hashlib.sha256(Path(context['prediction']).read_bytes()).hexdigest()
    payload = torch.load(context['prediction'], map_location='cpu', weights_only=False)
    arrays, metadata = export_arrays(payload, context, args.task, args.current_index, args.dt, args.batch_index)
    save_bundle(args.output, arrays, metadata)
    print(f'NEXUS_3DGS_EXPORT_OK agents={len(arrays["track_tokens"])} frames={len(arrays["timestamps_us"])} '
          f'max_reference_error_m={metadata["max_reference_position_error_m"]:.6f} output={args.output}', flush=True)


if __name__ == '__main__':
    main()
