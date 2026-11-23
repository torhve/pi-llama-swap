/**
 * Shared types for the pi llama-swap extension.
 */

/** Persisted connection settings for one llama-swap endpoint. */
export interface LlamaSwapInstance {
	/** Provider id registered in pi (unique, e.g. `llama-swap`, `llama-swap-2`). */
	id: string;
	/** Human-readable provider name shown in pi (e.g. `/model` list). */
	name: string;
	/** Origin without path, e.g. `http://127.0.0.1`. */
	origin: string;
	/** TCP port (1–65535). */
	port: number;
	/** OpenAI API path prefix (default `/v1`). */
	basePath?: string;
	/** Optional API key sent as Bearer when set. */
	apiKey?: string;
	/** Per-model context window overrides (model id -> tokens). */
	contextOverrides?: Record<string, number>;
	/** Cached per-model capabilities discovered from llama-swap (model id -> caps). */
	modelCapabilities?: Record<string, ModelCapabilities>;
}

/** Extension config: one or more llama-swap instances. */
export interface LlamaSwapConfig {
	instances: LlamaSwapInstance[];
}

/**
 * Engines pi-llama-swap can name from an observable `/props` signal. The
 * spellings match the engine names in pi-answer-now, so a value written here is
 * comparable there. Add a name and its detection signal together, once a
 * consumer needs it.
 */
export const KNOWN_ENGINES = ["basalt", "llamacpp", "strata"] as const;

/** One of `KNOWN_ENGINES`. */
export type PiLlamaSwapEngine = (typeof KNOWN_ENGINES)[number];

/** An engine name and its raw build string, as one `/props` response gave them. */
export interface DetectedEngine {
	id: PiLlamaSwapEngine;
	build?: string;
}

/**
 * Discovered capabilities for one model, cached in the config file between runs.
 * All fields optional: only values actually discovered are stored.
 */
export interface ModelCapabilities {
	/** Chat template supports reasoning (thinking). */
	reasoning?: boolean;
	/** Model accepts image input. */
	imageInput?: boolean;
	/** Discovered context window in tokens. */
	contextWindow?: number;
	/** Discovered max output tokens. */
	maxTokens?: number;
	/**
	 * Engine accepts `POST /v1/answer_now` to end a running reply's thinking.
	 * Basalt reports it in `/props` under `basalt.capabilities`. Absent means
	 * never discovered, unlike `false`, which means the running engine's props
	 * were read and did not list it.
	 */
	answerNow?: boolean;
	/**
	 * Engine behind this model, named by the signal its `/props` showed. Absent
	 * means no engine signal was read, which a consumer must not read as "is not
	 * that engine": a backend swapped under the same model id can go quiet and
	 * leave this value stale. Spellings match pi-answer-now's engine names.
	 */
	engine?: PiLlamaSwapEngine;
	/**
	 * The engine's own build string, as `/props` wrote `build_info`, such as
	 * `b9468-354ebac8`. Stored only for a named engine. Nothing here parses it,
	 * so a consumer can gate on its build number itself.
	 */
	engineBuild?: string;
}

/** OpenAI-compatible model entry from GET /v1/models. */
export interface OpenAIModelEntry {
	id: string;
	name?: string;
	[key: string]: unknown;
}

/** OpenAI-compatible models list response. */
export interface OpenAIModelsListResponse {
	object?: string;
	data: OpenAIModelEntry[];
}

/** Result of a provider refresh attempt. */
export interface RefreshResult {
	baseUrl: string;
	modelCount: number;
	/** Set on per-instance results: the instance this result belongs to. */
	instanceId?: string;
	/** Set on aggregate results: per-instance errors, keyed by instance id. */
	errorsByInstance?: Record<string, string>;
	error?: string;
	/** Observed llama-swap process state per model, keyed `${instanceId}:${modelId}`. */
	runningStates?: Record<string, string>;
}
