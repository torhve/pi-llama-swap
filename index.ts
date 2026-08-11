/**
 * Pi extension entry: llama-swap providers with dynamic model discovery.
 * Supports one or more llama-swap instances (see config.ts).
 *
 * Usage: pi -e /path/to/pi-llama-swap
 *
 * Config (optional): ~/.pi/agent/pi-llama-swap.json overrides defaults.
 */

import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

import { loadConfig, saveContextOverride } from "./lib/config.js";
import { refreshProvider } from "./lib/provider.js";

/**
 * Pi extension factory (async for model discovery before startup).
 * @param pi - Extension API instance.
 */
export default async function llamaSwapExtension(pi: ExtensionAPI): Promise<void> {
	const initialConfig = await loadConfig();
	const result = await refreshProvider(pi, initialConfig, { isInitial: true });
	/** Provider id → model id already refreshed with context limits. */
	let contextRefreshedForModel = new Map<string, string>();

	if (result.error) {
		console.warn(`[llama-swap] ${result.error}`);
	}

	// A llama-swap upstream is started by the first provider request. Refresh
	// once its response headers arrive so `/running` exposes the proxy and we
	// can read the actual llama-server `/props` n_ctx for subsequent requests.
	pi.on("after_provider_response", async (_event, ctx) => {
		const model = ctx.model;
		if (!model) {
			return;
		}
		const config = await loadConfig();
		const instance = config.instances.find((inst) => inst.id === model.provider);
		if (!instance || model.id === contextRefreshedForModel.get(model.provider)) {
			return;
		}

		const refresh = await refreshProvider(pi, config);
		if (!refresh.error) {
			contextRefreshedForModel.set(model.provider, model.id);
		}
	});

	pi.registerCommand("llama-swap-set-context-length", {
		description: "Set or clear context window override for the current model",
		handler: async (args, ctx) => {
			const model = ctx.model;
			if (!model) {
				ctx.ui.notify("No model selected. Use /model first.", "warning");
				return;
			}
			const config = await loadConfig();
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
				const config2 = await loadConfig();
				await refreshProvider(pi, config2);
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
			const config2 = await loadConfig();
			await refreshProvider(pi, config2);
			ctx.ui.notify(`Context window for ${model.id} set to ${ctxSize}.`);
		},
	});
}
