/**
 * Pi provider registration and model list refresh for llama-swap.
 * Supports one or more llama-swap instances, each registered as its own
 * pi provider id.
 */

import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

import { buildModelLimits, resolveContextWindow, resolveMaxTokens } from "./context.js";
import { fetchModels, LlamaSwapClientError } from "./client.js";
import { buildBaseUrl } from "./url.js";
import { DEFAULT_INSTANCE_ID } from "./config.js";
import type { LlamaSwapConfig, LlamaSwapInstance, OpenAIModelEntry, RefreshResult } from "./types.js";

/** Provider id of the first (default) instance. */
export const PROVIDER_ID = DEFAULT_INSTANCE_ID;

/**
 * Placeholder apiKey so pi lists models when llama-swap has no apiKeys.
 * OpenAI client may send `Authorization: Bearer <this>`; most open local proxies ignore it.
 */
export const NO_AUTH_API_KEY_PLACEHOLDER = "local-no-auth";

/** Provider ids successfully registered this session. */
const registeredIds = new Set<string>();

/**
 * Maps OpenAI model entries to pi provider model definitions.
 * @param entries - Models from GET /v1/models.
 * @param contextByModel - Resolved context window per model id.
 * @param maxTokensByModel - Resolved max output tokens per model id.
 * @param imageInputByModel - Image-input support reported by GET /props per model id.
 * @param reasoningByModel - Reasoning support reported by GET /props per model id.
 * @returns Pi-compatible model configs.
 */
export function mapOpenAIModelsToPi(
	entries: OpenAIModelEntry[],
	contextByModel: Map<string, number>,
	maxTokensByModel: Map<string, number>,
	imageInputByModel: Map<string, boolean>,
	reasoningByModel: Map<string, boolean>,
): ProviderModelConfig[] {
	return entries.map((model) => {
		const contextWindow = resolveContextWindow(model.id, contextByModel);
		const maxTokens = resolveMaxTokens(model.id, maxTokensByModel, contextWindow);
		const name = typeof model.name === "string" && model.name.length > 0 ? model.name : model.id;

		return {
			id: model.id,
			name,
			reasoning: reasoningByModel.has(model.id),
			input: (imageInputByModel.has(model.id) ? ["text", "image"] : ["text"]) as ("text" | "image")[],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow,
			maxTokens,
		};
	});
}

/**
 * Registers one llama-swap provider with the given models.
 * @param pi - Pi extension API.
 * @param instance - Connection settings for this instance.
 * @param models - Model list (may be empty).
 */
export function registerLlamaSwapProvider(
	pi: ExtensionAPI,
	instance: LlamaSwapInstance,
	models: ProviderModelConfig[],
): void {
	const baseUrl = buildBaseUrl(instance);
	const hasKey = Boolean(instance.apiKey?.trim());

	const providerConfig = {
		name: instance.name,
		baseUrl,
		api: "openai-completions" as const,
		models,
		...(hasKey
			? { apiKey: instance.apiKey, authHeader: true }
			: models.length > 0
				? { apiKey: NO_AUTH_API_KEY_PLACEHOLDER }
				: {}),
	};

	pi.registerProvider(instance.id, providerConfig);
	registeredIds.add(instance.id);
}

/**
 * Refreshes a single llama-swap instance's provider registration.
 * @param pi - Pi extension API.
 * @param instance - Instance connection settings.
 * @param options - `isInitial`: first load; may register empty provider on failure.
 * @returns Refresh outcome for this instance.
 */
async function refreshInstance(
	pi: ExtensionAPI,
	instance: LlamaSwapInstance,
	options?: { isInitial?: boolean },
): Promise<RefreshResult> {
	const baseUrl = buildBaseUrl(instance);

	try {
		const modelsController = new AbortController();
		const modelsTimeout = setTimeout(() => modelsController.abort(), 3000);
		let entries: OpenAIModelEntry[];
		try {
			entries = await fetchModels(baseUrl, instance.apiKey, modelsController.signal);
		} finally {
			clearTimeout(modelsTimeout);
		}
		// Initial load probes only models already running (no model swaps), so
		// reasoning/vision/context flags are correct before the first request.
		const { contextByModel, maxTokensByModel, imageInputByModel, reasoningByModel } = await buildModelLimits(
			entries,
			instance,
			instance.contextOverrides,
		);
		const models = mapOpenAIModelsToPi(entries, contextByModel, maxTokensByModel, imageInputByModel, reasoningByModel);

		if (registeredIds.has(instance.id)) {
			pi.unregisterProvider(instance.id);
		}
		registerLlamaSwapProvider(pi, instance, models);

		return { baseUrl, modelCount: models.length };
	} catch (err) {
		const message = err instanceof LlamaSwapClientError ? err.message : err instanceof Error ? err.message : String(err);

		if (options?.isInitial) {
			registerLlamaSwapProvider(pi, instance, []);
			return { baseUrl, modelCount: 0, error: message };
		}

		return { baseUrl, modelCount: 0, error: message };
	}
}

/**
 * Refreshes all configured llama-swap providers.
 * On failure of one instance, others still refresh; the returned result
 * reports per-instance errors and the total model count.
 * @param pi - Pi extension API.
 * @param config - Effective connection settings (one or more instances).
 * @param options - `isInitial`: first load; may register empty providers on failure.
 * @returns Aggregate refresh outcome.
 */
export async function refreshProvider(
	pi: ExtensionAPI,
	config: LlamaSwapConfig,
	options?: { isInitial?: boolean },
): Promise<RefreshResult> {
	const results = await Promise.all(config.instances.map((instance) => refreshInstance(pi, instance, options)));

	const errors = results.filter((r) => r.error);
	const error =
		errors.length > 0
			? errors.map((r) => `${r.baseUrl}: ${r.error}`).join("; ")
			: undefined;

	return {
		baseUrl: results.map((r) => r.baseUrl).join(", "),
		modelCount: results.reduce((sum, r) => sum + r.modelCount, 0),
		error,
	};
}
