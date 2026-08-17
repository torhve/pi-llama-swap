/**
 * Resolve per-model context windows from llama-swap HTTP APIs only.
 */

import type { LlamaSwapInstance, ModelCapabilities, OpenAIModelEntry } from "./types.js";
import { buildServerOrigin } from "./url.js";

/** Default context when llama-swap APIs do not report one (256K). */
export const DEFAULT_CONTEXT_WINDOW = 262_144;

const REQUEST_TIMEOUT_MS = 5_000;

/** Running process entry from GET /running. */
interface RunningProcess {
	model: string;
	cmd?: string;
	proxy?: string;
	state?: string;
}

/** Response shape from GET /running. */
interface RunningResponse {
	running?: RunningProcess[];
}

/** Chat template capabilities from /props. */
interface ChatTemplateCaps {
	supports_preserve_reasoning?: boolean;
	[key: string]: unknown;
}

/** llama-server /props response (subset). */
interface LlamaServerProps {
	default_generation_settings?: {
		n_ctx?: number;
	};
	chat_template_caps?: ChatTemplateCaps;
	capabilities?: unknown;
	modalities?: unknown;
	input?: unknown;
	vision?: unknown;
	supports_vision?: unknown;
	supports_images?: unknown;
}

/**
 * Coerces a value to a positive integer context size, or undefined if invalid.
 * @param value - Raw value from JSON.
 * @returns Valid context window or undefined.
 */
function toPositiveInt(value: unknown): number | undefined {
	if (typeof value === "number" && Number.isInteger(value) && value > 0) {
		return value;
	}
	if (typeof value === "string" && /^\d+$/.test(value)) {
		const n = Number(value);
		return n > 0 ? n : undefined;
	}
	return undefined;
}

/**
 * Returns whether a model's chat template supports reasoning (thinking).
 * Checks `chat_template_caps.supports_preserve_reasoning` from /props.
 */
function supportsReasoning(props: LlamaServerProps): boolean {
	return props.chat_template_caps?.supports_preserve_reasoning === true;
}

/**
 * Returns whether a model property advertises image/vision input.
 * It checks a boolean `vision` and lists such as `capabilities: ["vision"]` or
 * `modalities: ["text", "image"]`.
 */
function supportsImageInput(props: LlamaServerProps): boolean {
	if (props.vision === true || props.supports_vision === true || props.supports_images === true) {
		return true;
	}

	const hasImageCapability = (value: unknown): boolean => {
		if (Array.isArray(value)) {
			return value.some((item) => typeof item === "string" && /^(?:image|images|vision|multimodal)$/i.test(item));
		}
		if (value && typeof value === "object") {
			return Object.entries(value).some(
				([key, supported]) => /^(?:image|images|vision|multimodal)$/i.test(key) && supported === true,
			);
		}
		return false;
	};

	return hasImageCapability(props.capabilities) || hasImageCapability(props.modalities) || hasImageCapability(props.input);
}

/**
 * Extracts context length from a /v1/models entry (top-level or nested metadata).
 * @param entry - Model object from llama-swap.
 * @returns Context window in tokens, or undefined.
 */
export function extractContextFromModelEntry(entry: OpenAIModelEntry): number | undefined {
	const topLevel =
		toPositiveInt(entry.context_length) ??
		toPositiveInt(entry.max_context_length) ??
		toPositiveInt(entry.context_window);

	if (topLevel) {
		return topLevel;
	}

	const meta = entry.meta;
	if (meta && typeof meta === "object") {
		const llamaswap = (meta as Record<string, unknown>).llamaswap;
		if (llamaswap && typeof llamaswap === "object") {
			const ls = llamaswap as Record<string, unknown>;
			const fromLs =
				toPositiveInt(ls.context_length) ??
				toPositiveInt(ls.context) ??
				toPositiveInt(ls.max_context) ??
				toPositiveInt(ls.max_context_length);
			if (fromLs) {
				return fromLs;
			}
		}
		const fromMeta = toPositiveInt((meta as Record<string, unknown>).n_ctx);
		if (fromMeta) {
			return fromMeta;
		}
	}

	const metadata = entry.metadata;
	if (metadata && typeof metadata === "object") {
		const md = metadata as Record<string, unknown>;
		const fromMd = toPositiveInt(md.context_length) ?? toPositiveInt(md.context);
		if (fromMd) {
			return fromMd;
		}
	}

	return undefined;
}

/**
 * Extracts max output tokens from a /v1/models entry when present.
 * @param entry - Model object from llama-swap.
 * @returns Max output tokens or undefined.
 */
export function extractMaxTokensFromModelEntry(entry: OpenAIModelEntry): number | undefined {
	const top = toPositiveInt(entry.output_length) ?? toPositiveInt(entry.max_tokens);
	if (top) {
		return top;
	}

	const meta = entry.meta;
	if (meta && typeof meta === "object") {
		const llamaswap = (meta as Record<string, unknown>).llamaswap;
		if (llamaswap && typeof llamaswap === "object") {
			const fromLs =
				toPositiveInt((llamaswap as Record<string, unknown>).output_length) ??
				toPositiveInt((llamaswap as Record<string, unknown>).max_tokens);
			if (fromLs) {
				return fromLs;
			}
		}
	}

	return undefined;
}

/**
 * Parses `-c` or `--ctx-size` from a llama-server command string.
 * @param cmd - Shell command (may span multiple lines).
 * @returns Context size in tokens, or undefined.
 */
export function parseContextFromCmd(cmd: string): number | undefined {
	const ctxSizeMatch = cmd.match(/(?:^|\s)--ctx-size(?:=|\s+)(\d+)/);
	if (ctxSizeMatch) {
		return Number(ctxSizeMatch[1]);
	}

	const cMatch = cmd.match(/(?:^|\s)-c(?:=|\s+)(\d+)/);
	if (cMatch) {
		return Number(cMatch[1]);
	}

	return undefined;
}

/**
 * Fetches and parses a `/props` response.
 * @param url - Absolute `/props` URL.
 * @param headers - Optional request headers.
 * @returns Props or undefined when the endpoint is unavailable or invalid.
 */
async function fetchProps(url: URL | string, headers: Record<string, string> = { Accept: "application/json" }): Promise<LlamaServerProps | undefined> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
	try {
		const response = await fetch(url, { method: "GET", headers, signal: controller.signal });
		if (!response.ok) {
			return undefined;
		}
		return (await response.json()) as LlamaServerProps;
	} catch {
		return undefined;
	} finally {
		clearTimeout(timeout);
	}
}

/**
 * Info discovered for one running model.
 */
interface RunningModelInfo {
	/** n_ctx in tokens (upstream /props, else `-c`/`--ctx-size` in cmd). */
	ctx?: number;
	/** Parsed /props for capability detection (vision, reasoning). */
	props?: LlamaServerProps;
	/** llama-swap process state ("ready", "starting", "stopping", ...). */
	state?: string;
}

/**
 * Fetches GET /running and discovers per-model info in a single pass.
 * For each running model, /props is fetched from the upstream proxy, with
 * llama-swap's own /props?model=… as fallback (remote instance: the proxy
 * is localhost on the swap host, unreachable from Pi's machine). Only
 * already-running models are queried, so no model swaps are started that
 * could block Pi startup.
 * @param serverOrigin - llama-swap root URL (no `/v1`).
 * @param apiKey - Optional Bearer token.
 * @returns Map of model id → discovered info.
 */
async function loadRunningModelInfo(serverOrigin: string, apiKey?: string): Promise<Map<string, RunningModelInfo>> {
	const result = new Map<string, RunningModelInfo>();
	const url = `${serverOrigin.replace(/\/$/, "")}/running`;
	const headers: Record<string, string> = { Accept: "application/json" };
	if (apiKey) {
		headers.Authorization = `Bearer ${apiKey}`;
	}

	let response: Response;
	try {
		const controller = new AbortController();
		const timeout = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
		try {
			response = await fetch(url, { method: "GET", headers, signal: controller.signal });
		} finally {
			clearTimeout(timeout);
		}
	} catch {
		return result;
	}
	if (!response.ok) {
		return result;
	}

	let payload: RunningResponse;
	try {
		payload = (await response.json()) as RunningResponse;
	} catch {
		return result;
	}

	const origin = serverOrigin.replace(/\/$/, "");
	await Promise.all(
		(payload.running ?? []).map(async (proc) => {
			if (!proc.model) {
				return;
			}

			let props: LlamaServerProps | undefined;
			if (proc.proxy) {
				props = await fetchProps(`${proc.proxy.replace(/\/$/, "")}/props`, headers);
			}
			if (!props) {
				props = await fetchProps(`${origin}/props?model=${encodeURIComponent(proc.model)}`, headers);
			}

			let ctx: number | undefined = toPositiveInt(props?.default_generation_settings?.n_ctx);
			if (!ctx && proc.cmd) {
				ctx = parseContextFromCmd(proc.cmd);
			}

			result.set(proc.model, { ctx, props, state: proc.state });
		}),
	);

	return result;
}

/**
 * Builds per-model context and max-token maps from llama-swap APIs.
 * Merges the cached capabilities from the config file for models not
 * discovered live this pass (precedence: /v1/models entry < live /running +
 * /props < cache < user overrides). Also reports only the values actually
 * discovered this pass (`detectedByModel`) so callers can persist them.
 * @param entries - Models from GET /v1/models.
 * @param config - Instance connection settings.
 * @param overrides - Per-model context overrides (highest precedence).
 * @returns Context/max-token/capability maps, discovered-only caps, and running state per model id.
 */
export async function buildModelLimits(
	entries: OpenAIModelEntry[],
	config: LlamaSwapInstance,
	overrides?: Record<string, number>,
): Promise<{
	contextByModel: Map<string, number>;
	maxTokensByModel: Map<string, number>;
	imageInputByModel: Map<string, boolean>;
	reasoningByModel: Map<string, boolean>;
	detectedByModel: Map<string, ModelCapabilities>;
	/** llama-swap process state per model id (from GET /running). */
	runningStateByModel: Map<string, string>;
}> {
	const contextByModel = new Map<string, number>();
	const maxTokensByModel = new Map<string, number>();
	const detectedByModel = new Map<string, ModelCapabilities>();
	const runningStateByModel = new Map<string, string>();

	const recordDetected = (id: string, caps: ModelCapabilities): void => {
		if (Object.keys(caps).length === 0) {
			return;
		}
		detectedByModel.set(id, { ...detectedByModel.get(id), ...caps });
	};

	for (const entry of entries) {
		const fromEntry = extractContextFromModelEntry(entry);
		if (fromEntry) {
			contextByModel.set(entry.id, fromEntry);
			recordDetected(entry.id, { contextWindow: fromEntry });
		}
		const maxOut = extractMaxTokensFromModelEntry(entry);
		if (maxOut) {
			maxTokensByModel.set(entry.id, maxOut);
			recordDetected(entry.id, { maxTokens: maxOut });
		}
	}

	const serverOrigin = buildServerOrigin(config);
	const skipModels = overrides ? new Set(Object.keys(overrides)) : undefined;
	const running = await loadRunningModelInfo(serverOrigin, config.apiKey);
	for (const [id, info] of running) {
		// ponyail: models with context overrides keep their user-chosen size
		if (!skipModels?.has(id) && info.ctx) {
			contextByModel.set(id, info.ctx);
		}
		if (info.ctx) {
			recordDetected(id, { contextWindow: info.ctx });
		}
		if (info.props) {
			recordDetected(id, { reasoning: supportsReasoning(info.props), imageInput: supportsImageInput(info.props) });
		}
		// Models listed by /running are live upstreams; tag them with their state.
		runningStateByModel.set(id, info.state ?? "running");
	}

	// ponyail: cache fills gaps for models not discovered live this pass
	const cached = config.modelCapabilities;
	if (cached) {
		for (const [id, caps] of Object.entries(cached)) {
			if (caps.contextWindow !== undefined && !contextByModel.has(id)) {
				contextByModel.set(id, caps.contextWindow);
			}
			if (caps.maxTokens !== undefined && !maxTokensByModel.has(id)) {
				maxTokensByModel.set(id, caps.maxTokens);
			}
		}
	}

	// ponyail: user overrides beat all discovered values
	if (overrides) {
		for (const [id, ctx] of Object.entries(overrides)) {
			contextByModel.set(id, ctx);
		}
	}

	const reasoningByModel = new Map(
		[...running]
			.filter(([, info]) => info.props && supportsReasoning(info.props))
			.map(([id]) => [id, true] as const),
	);
	const imageInputByModel = new Map(
		[...running]
			.filter(([, info]) => info.props && supportsImageInput(info.props))
			.map(([id]) => [id, true] as const),
	);
	// ponyail: cached capabilities apply to models not running this pass
	if (cached) {
		for (const [id, caps] of Object.entries(cached)) {
			if (caps.reasoning === true && !reasoningByModel.has(id)) {
				reasoningByModel.set(id, true);
			}
			if (caps.imageInput === true && !imageInputByModel.has(id)) {
				imageInputByModel.set(id, true);
			}
		}
	}

	return { contextByModel, maxTokensByModel, imageInputByModel, reasoningByModel, detectedByModel, runningStateByModel };
}

/**
 * Resolves context window for a model id from llama-swap APIs, else default 256K.
 * @param modelId - Model identifier.
 * @param contextByModel - Resolved context map.
 * @returns Context window in tokens.
 */
export function resolveContextWindow(modelId: string, contextByModel: Map<string, number>): number {
	return contextByModel.get(modelId) ?? DEFAULT_CONTEXT_WINDOW;
}

/**
 * Resolves max output tokens for a model id.
 * Defaults to half the context window: llama-swap servers typically run
 * llama.cpp with `n_predict -1` (unlimited output), so the real limit is
 * context space. A fixed small cap would also suppress pi's compact-and-retry
 * overflow recovery for long truncated responses.
 * @param modelId - Model identifier.
 * @param maxTokensByModel - Resolved max-token map.
 * @param contextWindow - Model context window.
 * @returns Max output tokens.
 */
export function resolveMaxTokens(
	modelId: string,
	maxTokensByModel: Map<string, number>,
	contextWindow: number,
): number {
	if (maxTokensByModel.has(modelId)) {
		return maxTokensByModel.get(modelId)!;
	}
	return Math.floor(contextWindow / 2);
}
