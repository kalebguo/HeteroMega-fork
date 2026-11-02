"""Hardware validation of the C++ executor used by the model.

Checks alternating/zero inputs, two layers, two HIP streams, serial/concurrent
branches, completion counts and numerical metrics. Does not download weights.
This is a deployment check, not a whole-model benchmark or an accuracy claim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--iterations', type=int, default=3)
    ap.add_argument('--compare-python', action='store_true', help='after C++ teardown, require bit-exact NPU parity with the original Python host')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if args.iterations < 1:
        ap.error('--iterations must be positive')
    report = dict(status='starting', scope=__doc__, config=str(args.config.resolve()),
                  checks=[], samples=[], python_parity=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError('requires ROCm Torch and an available AMD GPU')
        import heteromega_ffn_deploy as deployment
        from experiments.ffn_partition.contract import (OutputIntegrity, accuracy_metrics,
            accuracy_summary, bf16, reference_partitioned, random_weights)
        from experiments.ffn_partition.run_upstream_partition import prepare_original_weights

        settings = json.loads(args.config.read_text())
        directory = (args.config.resolve().parent / settings['artifact_dir']).resolve()
        manifest = json.loads((directory / 'final.xclbin.json').read_text())
        m, h, inter, p = (manifest[key] for key in ('m', 'hidden', 'weight_n', 'channels'))
        report.update(shape=dict(m=m, hidden=h, intermediate=inter, npu_channels=p),
                      manifest=manifest, torch=str(torch.__version__), hip=torch.version.hip)
        executor = deployment.MergedFfn(str(args.config.resolve()))
        if not executor.supports(m, h, inter) or executor.supports(1, h, inter) or executor.supports(m + 1, h, inter):
            raise RuntimeError('shape/decode selection contract failed')
        canonical = [random_weights(h, inter, 12832 + layer) for layer in range(2)]
        originals = [prepare_original_weights(w) for w in canonical]
        # copy(): canonical buffers are deliberately read-only; Torch owns its separate GPU copy.
        packed = [[torch.from_numpy(w.data.copy()).to('cuda') for w in weights] for weights in originals]
        for weights in packed:
            executor.add_layer(*weights)
        rng = np.random.default_rng(12833)
        inputs = [bf16(rng.normal(size=(m, h)).astype(np.float32)) for _ in range(2)]
        inputs.append(np.zeros_like(inputs[0]))
        gpu_inputs = [torch.from_numpy(x).to(device='cuda', dtype=torch.bfloat16) for x in inputs]
        references = [[reference_partitioned(x, w, p) for x in inputs] for w in canonical]
        report['packed_weight_sha256'] = [[hashlib.sha256(w.data).hexdigest() for w in ws] for ws in originals]
        report['input_sha256'] = [hashlib.sha256(x.tobytes()).hexdigest() for x in inputs]
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        torch.cuda.synchronize()
        integrity = {name: OutputIntegrity() for name in ('output', 'gpu', 'npu')}
        cpp_npu = {}
        calls = 0

        def invoke(layer, index, slot, serial):
            with torch.cuda.stream(streams[slot]):
                # Guard the caller-provided output, including across repeated buffer use.
                guarded = torch.full((m * h + 256,), 123, dtype=torch.bfloat16, device='cuda')
                output = guarded[128:-128].view(m, h)
                streams[slot].synchronize()
                start = time.perf_counter_ns()
                result = executor.forward_out(layer, gpu_inputs[index], output, serial)
                streams[slot].synchronize()
                elapsed = (time.perf_counter_ns() - start) / 1000
                if not bool(torch.all(guarded[:128] == 123).item() and torch.all(guarded[-128:] == 123).item()):
                    raise RuntimeError('caller output guard corrupted')
                values = [tensor.float().cpu().numpy().copy() for tensor in result]
                return values, elapsed

        def record(layer, index, slot, serial, result, warmup=False):
            nonlocal calls
            calls += 1
            values, elapsed = result
            reference, partials = references[layer][index]
            metrics = {}
            for name, actual, expected in (('output', values[0], reference),
                                           ('gpu', values[1], partials['gpu']),
                                           ('npu', values[2], partials['npu'])):
                metrics[name] = accuracy_metrics(actual, expected,
                    'strict' if name == 'gpu' else 'report', integrity[name], (layer, index))
                if not metrics[name]['accepted']:
                    raise RuntimeError(f'{name} integration/control check failed: {metrics[name]}')
            if not np.array_equal(values[0], bf16(values[1] + values[2])):
                raise RuntimeError('C++ boundary output is not the BF16 sum of both partials')
            cpp_npu[(layer, index)] = values[2].copy()
            report['checks'].append(metrics)
            report['samples'].append(dict(layer=layer, input=index, slot=slot, serial=serial,
                warmup=warmup, complete_host_wall_us=elapsed))

        # Warm both layers and both slots, then alternate random/zero/repeated inputs.
        for layer in range(2):
            for slot in range(2):
                record(layer, 0, slot, False, invoke(layer, 0, slot, False), warmup=True)
        for iteration in range(args.iterations):
            for index in (0, 1, 2, 0):
                for serial in (True, False):
                    record(iteration % 2, index, iteration % 2, serial,
                           invoke(iteration % 2, index, iteration % 2, serial))
        # Simultaneous chunk-like calls: independent slots, one shared NPU execution owner.
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(invoke, layer, layer, layer, False) for layer in range(2)]
            for layer, job in enumerate(jobs):
                record(layer, layer, layer, False, job.result())
        if executor.completed_dispatches() != calls:
            raise RuntimeError('successful XRT completion count does not match FFN calls')
        before = executor.completed_dispatches()
        for bad in (gpu_inputs[0][:1], gpu_inputs[0].float()):
            try:
                executor.forward(0, bad)
            except RuntimeError:
                pass
            else:
                raise RuntimeError('invalid input accepted')
        if executor.completed_dispatches() != before:
            raise RuntimeError('invalid inputs dispatched NPU work')
        # Positive control of HETEROMEGA_FFN_FINITE_CHECK: a whole NaN row is what the next layer sees after an NPU
        # nonfinite element, and both modes must stop on it before dispatching.
        nan_row = gpu_inputs[0].clone()
        nan_row[7, :] = float('nan')
        try:
            executor.forward(0, nan_row)
        except RuntimeError as exc:
            if 'nonfinite input' not in str(exc):
                raise
        else:
            raise RuntimeError('NaN input row accepted')
        if executor.completed_dispatches() != before:
            raise RuntimeError('NaN input row dispatched NPU work')
        report['finite_check'] = dict(mode=os.environ.get('HETEROMEGA_FFN_FINITE_CHECK') or 'full', nan_row_rejected=True)
        report['completed_dispatches'] = before
        del executor
        gc.collect()
        if args.compare_python:
            # No build, no simultaneous Python/C++ contexts. Match current host-contract reference.
            from experiments.ffn_partition.npu_backend import PersistentXrtRunner
            specs = [((m * h,), np.uint16), *[((w.nbytes,), np.uint8) for w in originals[0]],
                     ((m * p,), np.uint16), ((m * h,), np.uint16)]
            runner = PersistentXrtRunner(directory / 'final.xclbin', directory / 'insts.bin', specs, output_count=2)
            for i, w in enumerate(originals[0], 1):
                runner.write_input(i, w.data)
            for index in (0, 1, 2, 0):
                bits = (inputs[index].view(np.uint32) >> 16).astype(np.uint16)
                runner.write_input(0, bits.reshape(-1))
                runner.dispatch_wait()
                got = runner.output_bits(index=1).reshape(m, h)
                expected_bits = (cpp_npu[(0, index)].view(np.uint32) >> 16).astype(np.uint16)
                equal = np.array_equal(got, expected_bits)
                report['python_parity'].append(dict(input=index, bit_exact=bool(equal)))
                if not equal:
                    raise RuntimeError('C++/Python NPU host parity failed')
        metrics = [metric for group in report['checks'] for metric in group.values()]
        report['accuracy'] = accuracy_summary(metrics)
        report['status'] = 'passed' if report['accuracy']['passed'] else 'completed_with_accuracy_deviations'
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps({k: report[k] for k in ('status', 'error', 'completed_dispatches', 'accuracy') if k in report}, indent=2))


if __name__ == '__main__':
    main()
