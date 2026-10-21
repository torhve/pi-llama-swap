/**
 * Pi provider registration and model list refresh for llama-swap.
 * Supports one or more llama-swap instances, each registered as its own
 * pi provider id.
 */

import type { ExtensionAPI, ProviderConfig, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import { withAbortTimeout } from "./abort.js";
import { fetchModels, LlamaSwapClientError } from "./client.js";
import { DEFAULT_INSTANCE_ID, saveModelCapabilities } from "./config.js";
import { buildModelLimits, resolveContextWindow, resolveMaxTokens } from "./context.js";
import type { LlamaSwapConfig, LlamaSwapInstance, OpenAIModelEntry, RefreshResult } from "./types.js";
import { buildBaseUrl } from "./url.js";

/**
 * pi's refresh context, derived from the registration contract so the package
 * needs no direct dependency on pi-ai.
 */
export type RefreshModelsContext = Parameters<NonNullable<ProviderConfig["refreshModels"]>>[0];

/** /v1/models fetch cap; pi's model picker allows 15s for a whole refresh. */
const MODELS_FETCH_TIMEOUT_MS = 3_000;

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
 * Outcome of the last probe per instance, reused by a catalog refresh that
 * follows closely: pi refreshes catalogs right after startup and twice per
 * picker open, and re-probing then would cost a second full round trip (or a
 * second 3s timeout when the instance is down).
 */
const lastProbeByInstance = new Map<string, { at: number; models: ProviderModelConfig[] }>();

/** Window in which a catalog refresh reuses the last probe. `force` bypasses it. */
const REFRESH_REUSE_WINDOW_MS = 1_000;

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
			...(supportsReasoning
				? {
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
					}
				: {}),
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
 * Registers one llama-swap provider with the given models, plus a
 * `refreshModels` hook pi calls when it refreshes catalogs.
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
		// pi calls this when it refreshes catalogs: opening the /model picker,
		// startup, `/models`, and `pi update --models`. pi publishes whatever we
		// return, so this only fetches — registering here would rebuild providers
		// mid-refresh. A failed probe returns the last known list tagged down, so
		// the picker keeps showing this instance's models instead of losing them.
		// context.allowNetwork is deliberately ignored: llama-swap is a local
		// service, and pi only passes allowNetwork:false under PI_OFFLINE — pi's own
		// llama.cpp extension likewise keeps local model discovery live offline.
		refreshModels: async (context: RefreshModelsContext): Promise<ProviderModelConfig[]> => {
			// pi refreshes catalogs at startup and every time the picker opens, so
			// reuse a probe that just ran rather than round-tripping again.
			const recent = lastProbeByInstance.get(instance.id);
			if (!context.force && recent && Date.now() - recent.at < REFRESH_REUSE_WINDOW_MS) {
				return recent.models;
			}
			try {
				const { models: refreshed } = await fetchInstanceModels(instance, { signal: context.signal });
				downTaggedIds.delete(instance.id);
				return refreshed;
			} catch (err) {
				// Closing the picker aborts the refresh. That is not evidence the
				// instance is down, so report the cancellation unchanged and cache
				// nothing: a later refresh must not inherit stale [⛔ down] tags.
				if (context.signal.aborted) {
					throw err;
				}
				const lastKnown = lastKnownModelsByInstance.get(instance.id);
				if (!lastKnown) {
					// Nothing cached to fall back on: let pi report the failure.
					throw err;
				}
				const tagged = downTaggedModels(lastKnown);
				lastProbeByInstance.set(instance.id, { at: Date.now(), models: tagged });
				return tagged;
			}
		},
		...(hasKey
			? { apiKey: instance.apiKey, authHeader: true }
			: models.length > 0
				? { apiKey: NO_AUTH_API_KEY_PLACEHOLDER }
				: {}),
	};

	pi.registerProvider(instance.id, providerConfig);
	registeredIds.add(instance.id);
}

/** Live model list of one instance, with the process states observed for it. */
export interface InstanceModelFetch {
	/** Pi-compatible model configs, status tags included. */
	models: ProviderModelConfig[];
	/** llama-swap process state per model id (from GET /running). */
	runningStateByModel: Map<string, string>;
}

/**
 * Fetches and maps one instance's live model list without touching pi's
 * provider registration. Used by `refreshInstance`, which registers the
 * result, and by `ProviderConfig.refreshModels`, where pi publishes the
 * returned list itself — registering from there would rebuild providers in
 * the middle of a refresh. Caches the list as this instance's last known one
 * and persists newly discovered capabilities.
 * @param instance - Instance connection settings.
 * @param options - `signal`: caller cancellation (pi aborts a picker refresh).
 * @returns Model configs plus the observed per-model process states.
 * @throws When the `/v1/models` probe fails.
 */
export async function fetchInstanceModels(
	instance: LlamaSwapInstance,
	options?: { signal?: AbortSignal },
): Promise<InstanceModelFetch> {
	const baseUrl = buildBaseUrl(instance);
	const { signal: fetchSignal, cleanup } = withAbortTimeout(options?.signal, MODELS_FETCH_TIMEOUT_MS);
	let entries: OpenAIModelEntry[];
	try {
		entries = await fetchModels(baseUrl, instance.apiKey, fetchSignal);
	} finally {
		cleanup();
	}

	// Vision comes from the /v1/models entry flags, so an unloaded vision model
	// is already image-capable here; /running + /props only add live context,
	// reasoning support, and process state for models llama-swap has loaded.
	const {
		contextByModel,
		maxTokensByModel,
		imageInputByModel,
		reasoningByModel,
		detectedByModel,
		runningStateByModel,
	} = await buildModelLimits(entries, instance, instance.contextOverrides, options?.signal);
	const models = mapOpenAIModelsToPi(
		entries,
		contextByModel,
		maxTokensByModel,
		imageInputByModel,
		reasoningByModel,
		runningStateByModel,
	);
	lastKnownModelsByInstance.set(instance.id, models);
	lastProbeByInstance.set(instance.id, { at: Date.now(), models });

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

	return { models, runningStateByModel };
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
		const { models, runningStateByModel } = await fetchInstanceModels(instance);

		if (registeredIds.has(instance.id)) {
			pi.unregisterProvider(instance.id);
		}
		registerLlamaSwapProvider(pi, instance, models);
		downTaggedIds.delete(instance.id);

		const runningStates: Record<string, string> = {};
		for (const [modelId, state] of runningStateByModel) {
			runningStates[`${instance.id}:${modelId}`] = state;
		}
		return { baseUrl, modelCount: models.length, instanceId: instance.id, runningStates };
	} catch (err) {
		const message =
			err instanceof LlamaSwapClientError ? err.message : err instanceof Error ? err.message : String(err);
		const lastKnown = lastKnownModelsByInstance.get(instance.id);

		if (lastKnown) {
			// Instance is down: keep its last good model list visible, tagged.
			// The tagged list is stable, so re-register only on the healthy→down
			// transition; repeated failures skip the unregister/register churn.
			const tagged = downTaggedModels(lastKnown);
			lastProbeByInstance.set(instance.id, { at: Date.now(), models: tagged });
			if (!downTaggedIds.has(instance.id)) {
				if (registeredIds.has(instance.id)) {
					pi.unregisterProvider(instance.id);
				}
				registerLlamaSwapProvider(pi, instance, tagged);
				downTaggedIds.add(instance.id);
			}
			return { baseUrl, modelCount: lastKnown.length, instanceId: instance.id, error: message };
		}

		// Nothing was ever discovered: record the empty outcome so pi's catalog
		// refresh right after a failed startup reuses it instead of paying for a
		// second fetch timeout on a dead instance.
		lastProbeByInstance.set(instance.id, { at: Date.now(), models: [] });

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
	const error =
		Object.keys(errorsByInstance).length > 0
			? results
					.filter((r) => r.error)
					.map((r) => `${r.baseUrl}: ${r.error}`)
					.join("; ")
			: undefined;

	return {
		baseUrl: results.map((r) => r.baseUrl).join(", "),
		modelCount: results.reduce((sum, r) => sum + r.modelCount, 0),
		error,
		...(error ? { errorsByInstance } : {}),
		runningStates,
	};
}
