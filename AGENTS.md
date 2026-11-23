# pi-llama-swap — Agent Context

## Structure

```
index.ts          # pi hooks, slash commands, provider registration
lib/provider.ts   # model list, refresh, mapping to pi model configs
lib/context.ts    # /v1/models, /running and /props probes; capability detection
lib/config.ts     # <agent-dir>/pi-llama-swap.json: load, normalize, save
lib/abort.ts      # one helper that combines caller abort and timeout
lib/client.ts     # HTTP with the shared abort/timeout helper
lib/url.ts        # URL building, defaults, argument parsing
lib/types.ts      # shared types, including the engine names
tests/            # vitest unit tests, one file per module
```

No build step. Pi loads `.ts` directly through jiti.

## Commands

```bash
npm run check      # biome check, vitest, tsc --noEmit
npm run test       # vitest only
npm run lint:fix   # biome autofix
```

Run `npm run check` yourself before you report work as done. Do not wait for CI. Fix every failure;
do not delete or weaken a test to pass one.

## modelCapabilities cache

`modelCapabilities` in `pi-llama-swap.json` is the best known fact per model, not a snapshot of the
last process that ran. Live detection writes what it saw. A field that detection did not read is
left alone, so a cached `true` or a cached `engine` survives a pass where the model was not running.
The two exceptions are deliberate: `imageInput` and `reasoning` are never demoted from `true` by a
silent live probe, while `engine` and `answerNow` are written only from a `/props` answer that
actually named something.

Do not add a sampling override for any engine — no `reasoning_budget_tokens`, no other
`samplingParams`. The operator sets the reasoning budget in the engine's own config, such as
`basalt_config.json`, or in pi's `modelOverrides`. Detect the engine and report it; do not tune it.
`pi-answer-now` follows the same rule.

## Changelog

Pi shows this package's changelog on the launch after an update. It reads `CHANGELOG.md` from the
package root and prints the version sections newer than the version it last recorded, under
"Extension Updates". `pi.changelogPath` points at another file, and `""` turns the changelog off.

Keep the file current:

- Land a user-facing change and its entry in the same commit. An internal change that nothing in a
  session can see gets no entry, or one that starts with `Internal:`.
- Write the entry under `## Unreleased`. At a release, turn that section into a version heading and
  move the entries under it.
- Use the heading form `## X.Y.Z — YYYY-MM-DD`. Pi recognises only `## ` followed by three
  dot-separated numbers, so `## Unreleased` and `### A sub-heading` are not entries. Never put a
  `## ` line inside a version section: it ends the entry.
- Put the newest version first. Pi prints entries in file order.
- Bump `version` in `package.json` in the same commit as the new version heading. Pi records the
  `package.json` version, not the heading. A heading above the current version prints again on every
  launch, and a heading below it never prints.
- Keep `CHANGELOG.md` in the `files` list of `package.json`, so an npm install carries it. Pi reads
  the changelog from the installed package root.
- Write what a user would act on: a new setting, a changed default, a renamed key, a new command, or
  a behaviour that changes what a session sees. Name the key and the value it replaces.
- Do not use `npm version` for the bump. It creates a tag and can publish. Edit the field, and let
  the maintainer cut the tag.

## Engine names

`lib/types.ts` holds `KNOWN_ENGINES`, and pi-answer-now's `EngineName` uses the same spellings:
`basalt`, `llamacpp`, `strata`. Add a name together with the `/props` signal that proves it, and
only when a consumer needs it. Do not invent a name for an engine you cannot detect.

## Push

Never run `git push` — not for branches, not for tags, not `--force-with-lease` — unless the user
approves that push in the current turn. The same applies to `npm publish`, `npm version`, and to
rewriting commits that already exist on the remote.
