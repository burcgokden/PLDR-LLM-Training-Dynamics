# Runtime compatibility and bounded maintenance

The numerical programs retain their recorded dependency behavior. Current CPU
checks exercise the installed Python, PyTorch and Transformers versions; this
is separate from reproducing an acquisition environment.

Both bundled native architecture families document `cache_position` in the
shared forward-argument documentation. It gives the absolute indices of the
current input tokens. The decoder derives consecutive indices from the cached
length when it is omitted, uses it for default `position_ids`, and forwards it
to causal masking and key/value cache updates. It does not select the G cache
or remove the proper-prefix requirement of query-Gram measurements. The change
only edits this documentation constant; all other parsed program nodes are
unchanged.

Three compatibility boundaries remain visible:

- Some dependency import routes invoke TorchScript. PyTorch warns that
  `torch.jit.script` is unsupported on Python 3.14 and later. Replacing it with
  compilation or export would change an execution interface and needs separate
  numerical qualification. No such replacement or warning filter is installed.
- The native configuration retains its legacy rotary-position configuration and
  calls `rope_config_validation`. Recent Transformers versions deprecate this
  call in favor of standardized rotary parameters and `validate_rope`. The
  constructor and serialized configuration fields remain those of the pinned
  model family. A configuration migration is outside this documentation change.
- The resource executor's injected CPU-test GPU-probe callback uses an explicit
  `fork` context so local callbacks and lambdas can be isolated and killed at a
  deadline. Python warns if the parent is multithreaded. Changing to `spawn`
  requires a picklable callback interface and new admission/teardown checks.
  Production resource probes use bounded subprocesses; this injected callback
  route is for CPU validation. The deadline and cleanup regressions remain part
  of the scientific suite.

SentencePiece's SWIG bindings may also emit Python deprecation diagnostics at
import or shutdown. Successful checks do not establish support for every future
library combination. See `VALIDATION.md` for executed routes and fixture skips.
