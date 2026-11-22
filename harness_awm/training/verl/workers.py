"""Recipe worker initialization and compact parameter-update evidence."""
import hashlib
import os

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers import ActorRolloutRefWorker


class AWMActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, *args, **kwargs):
        if os.environ.get('AWM_VERL_RUNTIME_PROFILE') in {'local-sdpa', 'local-sdpa-cu128'}:
            from .compat import install_local_sdpa
            install_local_sdpa()
        super().__init__(*args, **kwargs)
        if os.environ.get('AWM_PILOT_ALLOCATOR_GB'):
            import torch
            limit = float(os.environ['AWM_PILOT_ALLOCATOR_GB']) * 10**9
            torch.cuda.set_per_process_memory_fraction(min(1.0, limit / torch.cuda.get_device_properties(0).total_memory))
        self._parameter_sample = None

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def parameter_probe(self):
        import torch
        pieces = []
        finite = True
        for parameter in self.actor.engine.module.parameters():
            value = parameter.detach()
            if hasattr(value, 'to_local'):
                value = value.to_local()
            finite = finite and bool(torch.isfinite(value).all())
            flat = value.reshape(-1)
            stride = max(1, flat.numel() // 1024)
            pieces.append(flat[::stride][:1024].float().cpu())
        sample = torch.cat(pieces)
        previous = self._parameter_sample
        self._parameter_sample = sample
        result = {'finite': finite, 'sample_count': sample.numel(),
                  'sample_sha256': hashlib.sha256(sample.numpy().tobytes()).hexdigest()}
        if previous is not None:
            delta = (sample - previous).abs()
            result.update(changed_samples=int((delta > 0).sum()), max_abs_change=float(delta.max()))
        steps = []
        for state in self.actor.engine.optimizer.state.values():
            if 'step' in state:
                steps.append(int(state['step']))
        result['optimizer_state_steps'] = sorted(set(steps))
        result['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['cuda_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        return result
