/**
 * Pi extension entry: llama-swap providers with dynamic model discovery.
 * Supports one or more llama-swap instances (see config.ts).
 *
 * Usage: pi -e /path/to/pi-llama-swap
 *
 * Config (optional): ~/.pi/agent/pi-llama-swap.json overrides defaults.
 */

import type { ExtensionAPI, ExtensionUIContext } from "@earendil-works/pi-coding-agent";

import { configPath, loadConfig, saveContextOverride } from "./lib/config.js";
import { refreshProvider } from "./lib/provider.js";
import type { LlamaSwapConfig } from "./lib/types.js";

/** Warns about an unloadable config file at most once per session. */
let configWarned = false;

/**
 * Loads config, reporting failures to the user (once) instead of rejecting.
 * @param ui - Optional UI context for a user-visible warning.
 * @returns Effective config, or undefined when the config file cannot be loaded.
 */
async function loadConfigSafe(
	ui: Pick<ExtensionUIContext, "notify"> | undefined,
): Promise<LlamaSwapConfig | undefined> {
	try {
		return await loadConfig();
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		console.warn(`[llama-swap] failed to load config: ${message}`);
		if (!configWarned) {
			configWarned = true;
			ui?.notify(`[llama-swap] config at ${configPath()} could not be loaded: ${message}`, "warning");
		}
		return undefined;
	}
}

/**
 * Pi extension factory (async for model discovery before startup).
 * @param pi - Extension API instance.
 */
export default async function llamaSwapExtension(pi: ExtensionAPI): Promise<void> {
	const initialConfig = await loadConfig();
	const result = await refreshProvider(pi, initialConfig, { isInitial: true });
	/** Provider id → model id whose status tag has settled on a stable state. */
	let settledModelByProvider = new Map<string, string>();

	if (result.error) {
		console.warn(`[llama-swap] ${result.error}`);
	}

	// A llama-swap upstream is started by the first provider request. Refresh
	// once its response headers arrive so `/running` exposes the proxy and we
	// can read the actual llama-server `/props` n_ctx for subsequent requests.
	// llama-swap may flip `/running` to "ready" slightly after the first
	// response arrives, so a refresh that still sees "starting" re-checks on
	// the next response until the tag catches up (see settled latch below).
	pi.on("after_provider_response", async (_event, ctx) => {
		const model = ctx.model;
		if (!model) {
			return;
		}
		const config = await loadConfigSafe(ctx.ui);
		if (!config) {
			return;
		}
		const instance = config.instances.find((inst) => inst.id === model.provider);
		if (!instance || model.id === settledModelByProvider.get(model.provider)) {
			return;
		}

		const refresh = await refreshProvider(pi, config);
		if (refresh.error) {
			console.warn(`[llama-swap] post-response refresh failed (will retry next response): ${refresh.error}`);
			return;
		}
		// Settle only once the model's process state is stable. A transient
		// state (starting/stopping) keeps re-checking on the next response so
		// the `[starting]` tag does not stick once the upstream is ready.
		const state = refresh.runningStates?.[`${model.provider}:${model.id}`];
		if (state !== "starting" && state !== "stopping") {
			settledModelByProvider.set(model.provider, model.id);
		}
	});

	// Pre-warm on model switch: re-probe running models so capability flags
	// (reasoning, vision, context) are fresh for models with cached capabilities
	// from a prior session. A never-run model's upstream isn't started yet, so
	// the post-response refresh still does the real work for that case.
	// Same-provider switches and the cooldown keep cycling from hammering the proxy.
	const PRE_WARM_COOLDOWN_MS = 30_000;
	let lastPreWarmAt = 0;
	pi.on("model_select", (event, ctx) => {
		const model = event.model;
		if (model.provider === event.previousModel?.provider) {
			return;
		}
		const now = Date.now();
		if (now - lastPreWarmAt < PRE_WARM_COOLDOWN_MS) {
			return;
		}
		lastPreWarmAt = now;
		void (async () => {
			const config = await loadConfigSafe(ctx.ui);
			if (!config || !config.instances.some((inst) => inst.id === model.provider)) {
				return;
			}
			const refresh = await refreshProvider(pi, config);
			if (refresh.error) {
				console.warn(`[llama-swap] model_select refresh failed: ${refresh.error}`);
			}
		})().catch((err) => {
			console.warn(`[llama-swap] model_select refresh failed: ${err instanceof Error ? err.message : String(err)}`);
		});
	});

	pi.registerCommand("llama-swap-refresh", {
		description: "Re-probe running llama-swap models and re-register capability flags (reasoning/vision/context)",
		handler: async (_args, ctx) => {
			try {
				const config = await loadConfigSafe(ctx.ui);
				if (!config) {
					return;
				}
				const refresh = await refreshProvider(pi, config);
				ctx.ui.notify(
					refresh.error
						? `[llama-swap] refresh finished with errors: ${refresh.error}`
						: `[llama-swap] refreshed ${refresh.modelCount} models`,
					refresh.error ? "warning" : "info",
				);
			} catch (err) {
				ctx.ui.notify(
					`[llama-swap] refresh failed: ${err instanceof Error ? err.message : String(err)}`,
					"error",
				);
			}
		},
	});

	pi.registerCommand("llama-swap-set-context-length", {
		description: "Set or clear context window override for the current model",
		handler: async (args, ctx) => {
			const model = ctx.model;
			if (!model) {
				ctx.ui.notify("No model selected. Use /model first.", "warning");
				return;
			}
			const config = await loadConfigSafe(ctx.ui);
			if (!config) {
				return;
			}
			const instance = config.instances.find((inst) => inst.id === model.provider);
			if (!instance) {
				ctx.ui.notify("No llama-swap model selected. Use /model first.", "warning");
				return;
			}

			const trimmed = args.trim();

			if (trimmed === "auto") {
				const ok = await ctx.ui.confirm(
					"Clear context override",
					`Clear context override for "${model.id}" (${instance.name})?`,
				);
				if (!ok) return;

				await saveContextOverride(instance.id, model.id, undefined);
				// ponyail: reload config + refresh to pick up removed override
				const config2 = await loadConfigSafe(ctx.ui);
				if (config2) {
					await refreshProvider(pi, config2);
				}
				ctx.ui.notify(`Context override removed for ${model.id}. Now auto-detected.`);
				return;
			}

			const ctxSize = Number(trimmed);
			if (!Number.isInteger(ctxSize) || ctxSize < 1) {
				ctx.ui.notify(
					"Invalid context size. Use a positive integer or \"auto\".\nExample: /llama-swap-set-context-length 32768",
					"error",
				);
				return;
			}

			const ok = await ctx.ui.confirm(
				"Set context override",
				`Set context window to ${ctxSize} for "${model.id}" (${instance.name})?`,
			);
			if (!ok) return;

			await saveContextOverride(instance.id, model.id, ctxSize);
			// ponyail: reload config + refresh to pick up new override
			const config2 = await loadConfigSafe(ctx.ui);
			if (config2) {
				await refreshProvider(pi, config2);
			}
			ctx.ui.notify(`Context window for ${model.id} set to ${ctxSize}.`);
		},
	});
}
