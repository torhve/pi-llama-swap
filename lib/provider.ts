/**
 * Pi provider registration and model list refresh for llama-swap.
 * Supports one or more llama-swap instances, each registered as its own
 * pi provider id.
 */

import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";

import { buildModelLimits, resolveContextWindow, resolveMaxTokens } from "./context.js";
import { fetchModels, LlamaSwapClientError } from "./client.js";
import { buildBaseUrl } from "./url.js";
import { DEFAULT_INSTANCE_ID, saveModelCapabilities } from "./config.js";
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
 * Last successfully-registered models per instance id. On a later refresh
 * failure these are re-registered with a ` [⛔ down]` tag so the picker
 * keeps showing the instance's models instead of losing them.
 */
const lastKnownModelsByInstance = new Map<string, ProviderModelConfig[]>();

/**
 * Instance ids currently registered with a ` [⛔ down]` tag. While down,
 * repeated failed refreshes skip re-registering the (stable) tagged list to
 * avoid unregister/register churn on every re-probe.
 */
const downTaggedIds = new Set<string>();

/**
 * Maps OpenAI model entries to pi provider model definitions.
 * @param entries - Models from GET /v1/models.
 * @param contextByModel - Resolved context window per model id.
 * @param maxTokensByModel - Resolved max output tokens per model id.
 * @param imageInputByModel - Image input per model id (GET /v1/models entry flags,
 *   live GET /props of a running model, or the capabilities cache).
 * @param reasoningByModel - Reasoning support reported by GET /props per model id.
 * @param runningStateByModel - llama-swap process state per model id (GET /running).
 * @returns Pi-compatible model configs.
 */
export function mapOpenAIModelsToPi(
	entries: OpenAIModelEntry[],
	contextByModel: Map<string, number>,
	maxTokensByModel: Map<string, number>,
	imageInputByModel: Map<string, boolean>,
	reasoningByModel: Map<string, boolean>,
	runningStateByModel?: Map<string, string>,
): ProviderModelConfig[] {
	return entries.map((model) => {
		const contextWindow = resolveContextWindow(model.id, contextByModel);
		const maxTokens = resolveMaxTokens(model.id, maxTokensByModel, contextWindow);
		const baseName = typeof model.name === "string" && model.name.length > 0 ? model.name : model.id;
		const runState = runningStateByModel?.get(model.id);
		// pi renders the model name in the /model picker footer and search text;
		// tagging running models so their upstream state is visible there.
		const name = runState !== undefined ? `${baseName} [${runStateTag(runState)}]` : baseName;
		const supportsReasoning = reasoningByModel.has(model.id);

		return {
			id: model.id,
			name,
			reasoning: supportsReasoning,
			input: (imageInputByModel.has(model.id) ? ["text", "image"] : ["text"]) as ("text" | "image")[],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow,
			maxTokens,
			...(supportsReasoning ? {
				// Map pi thinking levels to the chat template's reasoning_effort
				// vocabulary. llama-server renders these into the template, which
				// picks the effort tier. "off" is omitted (see thinking.enabled).
				thinkingLevelMap: {
					minimal: "low",
					low: "low",
					medium: "medium",
					high: "xhigh",
				},
				compat: {
					thinkingFormat: "chat-template" as const,
					chatTemplateKwargs: {
						// Toggle the template's enable_thinking (defaults true, so
						// "off" must explicitly send false to stop thinking).
						enable_thinking: { $var: "thinking.enabled" as const },
						// Effort tier; resolved through thinkingLevelMap, omitted at off.
						reasoning_effort: { $var: "thinking.effort" as const },
					},
				},
			} : {}),
		};
	});
}

/**
 * Returns the bracketed status tag (icon + text) for a llama-swap process state
 * shown in model names.
 * @param state - Process state from GET /running ("ready", "starting", ...).
 * @returns Tag content, e.g. "🟢 running".
 */
function runStateTag(state: string): string {
	switch (state) {
		case "ready":
		case "running":
			return "🟢 running";
		case "starting":
			return "🟡 starting";
		case "stopping":
			return "🟠 stopping";
		default:
			return `⚪ ${state}`;
	}
}

/**
 * Returns copies of cached model configs with any existing trailing bracket
 * tag (e.g. ` [🟢 running]`) replaced by the down tag.
 * @param models - Last successfully-registered models for a downed instance.
 * @returns Models with names ending in ` [⛔ down]`.
 */
function downTaggedModels(models: ProviderModelConfig[]): ProviderModelConfig[] {
	return models.map((m) => ({
		...m,
		name: `${m.name.replace(/\s*\[[^\]]*\]$/, "")} [⛔ down]`,
	}));
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
		// Initial load probes only models already running (no model swaps). Vision
		// comes from the /v1/models entry flags instead, so an unloaded vision
		// model is registered image-capable before the first request.
		const { contextByModel, maxTokensByModel, imageInputByModel, reasoningByModel, detectedByModel, runningStateByModel } = await buildModelLimits(
			entries,
			instance,
			instance.contextOverrides,
		);
		const models = mapOpenAIModelsToPi(entries, contextByModel, maxTokensByModel, imageInputByModel, reasoningByModel, runningStateByModel);
		lastKnownModelsByInstance.set(instance.id, models);

		if (registeredIds.has(instance.id)) {
			pi.unregisterProvider(instance.id);
		}
		registerLlamaSwapProvider(pi, instance, models);
		downTaggedIds.delete(instance.id);

		// ponyail: persist discovered capabilities so non-running models keep
		// them on the next run (e.g. thinking support before the first request).
		if (detectedByModel.size > 0) {
			try {
				await saveModelCapabilities(instance.id, Object.fromEntries(detectedByModel));
			} catch (err) {
				const message = err instanceof Error ? err.message : String(err);
				console.warn(`[llama-swap] failed to cache model capabilities: ${message}`);
			}
		}

		const runningStates: Record<string, string> = {};
		for (const [modelId, state] of runningStateByModel) {
			runningStates[`${instance.id}:${modelId}`] = state;
		}
		return { baseUrl, modelCount: models.length, instanceId: instance.id, runningStates };
	} catch (err) {
		const message = err instanceof LlamaSwapClientError ? err.message : err instanceof Error ? err.message : String(err);
		const lastKnown = lastKnownModelsByInstance.get(instance.id);

		if (lastKnown) {
			// Instance is down: keep its last good model list visible, tagged.
			// The tagged list is stable, so re-register only on the healthy→down
			// transition; repeated failures skip the unregister/register churn.
			if (!downTaggedIds.has(instance.id)) {
				if (registeredIds.has(instance.id)) {
					pi.unregisterProvider(instance.id);
				}
				registerLlamaSwapProvider(pi, instance, downTaggedModels(lastKnown));
				downTaggedIds.add(instance.id);
			}
			return { baseUrl, modelCount: lastKnown.length, instanceId: instance.id, error: message };
		}

		if (options?.isInitial) {
			registerLlamaSwapProvider(pi, instance, []);
			return { baseUrl, modelCount: 0, instanceId: instance.id, error: message };
		}

		return { baseUrl, modelCount: 0, instanceId: instance.id, error: message };
	}
}

/** In-flight refresh so concurrent callers share one unregister/register pass. */
let inflightRefresh: Promise<RefreshResult> | undefined;

/**
 * Refreshes all configured llama-swap providers.
 * On failure of one instance, others still refresh; the returned result
 * reports per-instance errors and the total model count.
 * Concurrent calls share a single in-flight refresh so interleaved
 * unregister/register sequences cannot race.
 * @param pi - Pi extension API.
 * @param config - Effective connection settings (one or more instances).
 * @param options - `isInitial`: first load; may register empty providers on failure.
 * @returns Aggregate refresh outcome.
 */
export function refreshProvider(
	pi: ExtensionAPI,
	config: LlamaSwapConfig,
	options?: { isInitial?: boolean },
): Promise<RefreshResult> {
	inflightRefresh ??= doRefreshProvider(pi, config, options).finally(() => {
		inflightRefresh = undefined;
	});
	return inflightRefresh;
}

async function doRefreshProvider(
	pi: ExtensionAPI,
	config: LlamaSwapConfig,
	options?: { isInitial?: boolean },
): Promise<RefreshResult> {
	const results = await Promise.all(config.instances.map((instance) => refreshInstance(pi, instance, options)));

	const runningStates: Record<string, string> = {};
	const errorsByInstance: Record<string, string> = {};
	for (const r of results) {
		if (r.error) {
			errorsByInstance[r.instanceId ?? ""] = r.error;
		}
		Object.assign(runningStates, r.runningStates);
	}
	const error = Object.keys(errorsByInstance).length > 0
		? results.filter((r) => r.error).map((r) => `${r.baseUrl}: ${r.error}`).join("; ")
		: undefined;

	return {
		baseUrl: results.map((r) => r.baseUrl).join(", "),
		modelCount: results.reduce((sum, r) => sum + r.modelCount, 0),
		error,
		...(error ? { errorsByInstance } : {}),
		runningStates,
	};
}
