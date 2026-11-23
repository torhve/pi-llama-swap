# Changelog

Pi shows this package's changelog on the launch after an update. It reads `CHANGELOG.md` from the
package root and prints the version sections newer than the version it last recorded, under
"Extension Updates". It matches `## ` followed by three dot-separated numbers, so keep the heading
form `## X.Y.Z — YYYY-MM-DD` and put the newest section first. `## Unreleased` is not an entry: it
is staging for the next release. See `AGENTS.md` for the rules.

## Unreleased

- Name the engine behind a model. The extension reads each running model's `/props` and caches
  `engine` in `modelCapabilities` as `basalt`, `strata`, or `llamacpp`, with the engine's own build
  string in `engineBuild`. Other extensions read both fields from `pi-llama-swap.json` instead of
  probing the engine themselves; `pi-answer-now` uses them to pick its interrupt route and to arm a
  llama.cpp request. An absent `engine` means no engine signal was read, which is not the same as a
  model being known to run something else. A cached name is the last one detected: swapping an
  alias to a backend that sends no engine signal leaves the old name in place.
- The extension sends no sampling overrides to any model. A reasoning budget is set in the engine's
  own config, such as `basalt_config.json`, or in pi's `modelOverrides` under `samplingParams`. Keep
  a budget below pi's `max_tokens` when the engine caps the whole reply at it.
- Reads a running model's `/props` through llama-swap's `/upstream/<id>/props` when the engine's proxy
  address and llama-swap's `/props?model=<id>` both give no answer. llama-swap forwards that query to
  the engine unchanged, so an engine that serves `/props` only under its own model name rejects it —
  Strata answers `model not found` for a llama-swap id — and the model was left with no thinking
  support in pi. Neither probe loads a model: the `?model=` probe carries `autoload=false`, and the
  `/upstream/…` probe runs only after that one answers 404, and only for a model `/running` reported
  ready. A reverse proxy in front of llama-swap must forward that path.

