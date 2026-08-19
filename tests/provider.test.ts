import { afterEach, describe, expect, it, vi } from "vitest";

// Mock the external boundaries of provider.ts:
//  - client.fetchModels  (we drive the model list per test)
//  - config.saveModelCapabilities (we assert it persists detected caps)
// Everything else (context.buildModelLimits, url.buildBaseUrl) stays real;
// buildModelLimits' /running probe goes through the stubbed global fetch.
vi.mock("../lib/client.js", async (importOriginal) => {
	const original = await importOriginal<typeof import("../lib/client.js")>();
	return { ...original, fetchModels: vi.fn() };
});

vi.mock("../lib/config.js", async (importOriginal) => {
	const original = await importOriginal<typeof import("../lib/config.js")>();
	return { ...original, saveModelCapabilities: vi.fn() };
});

import {
	mapOpenAIModelsToPi,
	NO_AUTH_API_KEY_PLACEHOLDER,
	PROVIDER_ID,
	registerLlamaSwapProvider,
} from "../lib/provider.js";
import { createEntry, createInstance, createMockPi, jsonResponse, mockFetch } from "./mocks.js";
import type { ExtensionAPI, ProviderModelConfig } from "@earendil-works/pi-coding-agent";
import type { LlamaSwapInstance } from "../lib/types.js";

afterEach(() => {
	vi.unstubAllGlobals();
	vi.resetModules();
});

/**
 * Re-imports provider/client/config after a module reset.
 * Imports are sequential so provider.js fully evaluates (pulling in its
 * client.js/config.js dependencies) before we grab those modules — a parallel
 * Promise.all race can bind provider to a different client.js instance than we
 * return here. client.js/config.js are not re-evaluated by resetModules, so we
 * reset their mock state explicitly to avoid calls accumulating across tests.
 * The mocked fetchModels/saveModelCapabilities are re-typed so their mock
 * helpers type-check (TS otherwise sees the real module signatures).
 */
async function loadFresh() {
	const provider = await import("../lib/provider.js");
	const client = (await import("../lib/client.js")) as unknown as { fetchModels: ReturnType<typeof vi.fn> };
	const config = (await import("../lib/config.js")) as unknown as { saveModelCapabilities: ReturnType<typeof vi.fn> };
	client.fetchModels.mockReset();
	config.saveModelCapabilities.mockReset();
	return { provider, client, config };
}

/** Stubs global fetch so buildModelLimits sees an empty /running. */
function stubEmptyRunning() {
	return mockFetch((url) => {
		if (String(url).endsWith("/running")) return jsonResponse({ running: [] });
		return jsonResponse({});
	});
}

const oneModel = (): ProviderModelConfig[] => [
	{ id: "m1", name: "m1", reasoning: false, input: ["text"], cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 }, contextWindow: 4096, maxTokens: 2048 },
];

describe("PROVIDER_ID / NO_AUTH_API_KEY_PLACEHOLDER", () => {
	it("exposes the default provider id and the no-auth placeholder", () => {
		expect(PROVIDER_ID).toBe("llama-swap");
		expect(NO_AUTH_API_KEY_PLACEHOLDER).toBe("local-no-auth");
	});
});

describe("mapOpenAIModelsToPi", () => {
	const emptyMaps = () => ({
		ctx: new Map<string, number>(),
		max: new Map<string, number>(),
		img: new Map<string, boolean>(),
		reason: new Map<string, boolean>(),
	});

	it("maps id and name, falling back to id when name is missing or empty", () => {
		const { ctx, max, img, reason } = emptyMaps();
		const [withName, missing, empty] = mapOpenAIModelsToPi(
			[
				{ id: "m1", name: "My Model" },
				{ id: "m2" },
				{ id: "m3", name: "" },
			],
			ctx,
			max,
			img,
			reason,
		);
		expect(withName.name).toBe("My Model");
		expect(missing.name).toBe("m2");
		expect(empty.name).toBe("m3");
	});

	it("sets all cost fields to zero", () => {
		const { ctx, max, img, reason } = emptyMaps();
		const [m] = mapOpenAIModelsToPi([{ id: "m1" }], ctx, max, img, reason);
		expect(m.cost).toEqual({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
	});

	it("reads contextWindow from the map, defaulting to 256K when absent", () => {
		const { max, img, reason } = emptyMaps();
		const ctx = new Map([["m1", 4096]]);
		const [mapped, missing] = mapOpenAIModelsToPi([{ id: "m1" }, { id: "m2" }], ctx, max, img, reason);
		expect(mapped.contextWindow).toBe(4096);
		expect(missing.contextWindow).toBe(262_144);
	});

	it("reads maxTokens from the map, else floor(contextWindow/2)", () => {
		const { img, reason } = emptyMaps();
		const ctx = new Map([["m1", 4096]]);
		const max = new Map([["m1", 1024]]);
		const [mapped, missing] = mapOpenAIModelsToPi([{ id: "m1" }, { id: "m2" }], ctx, max, img, reason);
		expect(mapped.maxTokens).toBe(1024);
		// m2 has no maxTokens and no contextWindow → default 262144 → half.
		expect(missing.maxTokens).toBe(131_072);
	});

	it("marks image input when the model is in imageInputByModel", () => {
		const { ctx, max, reason } = emptyMaps();
		const img = new Map([["m1", true]]);
		const [mapped, missing] = mapOpenAIModelsToPi([{ id: "m1" }, { id: "m2" }], ctx, max, img, reason);
		expect(mapped.input).toEqual(["text", "image"]);
		expect(missing.input).toEqual(["text"]);
	});

	it("adds reasoning, thinkingLevelMap, and the compat block only when reasoning is supported", () => {
		const { ctx, max, img } = emptyMaps();
		const reason = new Map([["m1", true]]);
		const [mapped, missing] = mapOpenAIModelsToPi([{ id: "m1" }, { id: "m2" }], ctx, max, img, reason);

		expect(mapped.reasoning).toBe(true);
		expect(mapped.thinkingLevelMap).toEqual({ minimal: "low", low: "low", medium: "medium", high: "xhigh" });
		expect(mapped.compat).toEqual({
			thinkingFormat: "chat-template",
			chatTemplateKwargs: {
				enable_thinking: { $var: "thinking.enabled" },
				reasoning_effort: { $var: "thinking.effort" },
			},
		});

		expect(missing.reasoning).toBe(false);
		expect(missing.thinkingLevelMap).toBeUndefined();
		expect(missing.compat).toBeUndefined();
	});

	it("tags the model name with the running state", () => {
		const { ctx, max, img, reason } = emptyMaps();
		const states = new Map([
			["ready", "ready"],
			["running", "running"],
			["starting", "starting"],
			["stopping", "stopping"],
			["weird", "weird"],
		]);
		const entries = [...states.keys()].map((id) => ({ id, name: "My Model" }));
		const mapped = mapOpenAIModelsToPi(entries, ctx, max, img, reason, states);
		const byId = Object.fromEntries(mapped.map((m) => [m.id, m.name]));
		expect(byId.ready).toBe("My Model [🟢 running]");
		expect(byId.running).toBe("My Model [🟢 running]");
		expect(byId.starting).toBe("My Model [🟡 starting]");
		expect(byId.stopping).toBe("My Model [🟠 stopping]");
		expect(byId.weird).toBe("My Model [⚪ weird]");
	});

	it("leaves the name plain when there is no running state", () => {
		const { ctx, max, img, reason } = emptyMaps();
		const [m] = mapOpenAIModelsToPi([{ id: "m1", name: "My Model" }], ctx, max, img, reason);
		expect(m.name).toBe("My Model");
	});
});

describe("registerLlamaSwapProvider", () => {
	it("sets apiKey and authHeader when the instance has an apiKey", () => {
		const pi = createMockPi();
		registerLlamaSwapProvider(pi as unknown as ExtensionAPI, createInstance({ apiKey: "secret" }), oneModel());
		const config = pi.providers.get("llama-swap") as Record<string, unknown>;
		expect(config.apiKey).toBe("secret");
		expect(config.authHeader).toBe(true);
	});

	it("uses the no-auth placeholder when there is no key and models are present", () => {
		const pi = createMockPi();
		registerLlamaSwapProvider(pi as unknown as ExtensionAPI, createInstance(), oneModel());
		const config = pi.providers.get("llama-swap") as Record<string, unknown>;
		expect(config.apiKey).toBe("local-no-auth");
		expect(config.authHeader).toBeUndefined();
	});

	it("omits apiKey entirely when there is no key and no models", () => {
		const pi = createMockPi();
		registerLlamaSwapProvider(pi as unknown as ExtensionAPI, createInstance(), []);
		const config = pi.providers.get("llama-swap") as Record<string, unknown>;
		expect(config).not.toHaveProperty("apiKey");
		expect(config).not.toHaveProperty("authHeader");
	});

	it("registers under the instance id with name, baseUrl, api, and models", () => {
		const pi = createMockPi();
		const models = oneModel();
		registerLlamaSwapProvider(pi as unknown as ExtensionAPI, createInstance(), models);

		expect(pi.registerProvider).toHaveBeenCalledWith(
			"llama-swap",
			expect.objectContaining({
				name: "Llama Swap",
				baseUrl: "http://127.0.0.1:8080/v1",
				api: "openai-completions",
				models,
			}),
		);
	});
});

describe("refreshProvider", () => {
	it("registers the provider with mapped models and persists detected caps", async () => {
		const { provider, client, config } = await loadFresh();
		client.fetchModels.mockResolvedValue([createEntry({ id: "model-1", context_length: 4096 })]);
		stubEmptyRunning();
		const pi = createMockPi();

		const result = await provider.refreshProvider(pi as unknown as ExtensionAPI, { instances: [createInstance()] });

		expect(result.baseUrl).toBe("http://127.0.0.1:8080/v1");
		expect(result.modelCount).toBe(1);
		expect(result.error).toBeUndefined();
		expect(pi.registerProvider).toHaveBeenCalledWith(
			"llama-swap",
			expect.objectContaining({
				models: expect.arrayContaining([expect.objectContaining({ id: "model-1", contextWindow: 4096 })]),
			}),
		);
		expect(config.saveModelCapabilities).toHaveBeenCalledWith("llama-swap", {
			"model-1": { contextWindow: 4096 },
		});
	});

	it("registers an empty provider and reports the error on an initial failure", async () => {
		const { provider, client } = await loadFresh();
		client.fetchModels.mockRejectedValue(new Error("boom"));
		stubEmptyRunning();
		const pi = createMockPi();

		const result = await provider.refreshProvider(pi as unknown as ExtensionAPI, { instances: [createInstance()] }, { isInitial: true });

		expect(result.modelCount).toBe(0);
		expect(result.error).toBe("http://127.0.0.1:8080/v1: boom");
		expect(pi.registerProvider).toHaveBeenCalledWith("llama-swap", expect.objectContaining({ models: [] }));
	});

	it("does not register the provider on a non-initial failure", async () => {
		const { provider, client } = await loadFresh();
		client.fetchModels.mockRejectedValue(new Error("boom"));
		stubEmptyRunning();
		const pi = createMockPi();

		const result = await provider.refreshProvider(pi as unknown as ExtensionAPI, { instances: [createInstance()] });

		expect(result.modelCount).toBe(0);
		expect(result.error).toBe("http://127.0.0.1:8080/v1: boom");
		expect(pi.registerProvider).not.toHaveBeenCalled();
	});

	it("aggregates per-instance results across multiple instances", async () => {
		const { provider, client } = await loadFresh();
		client.fetchModels.mockImplementation((baseUrl: string) => {
			if (baseUrl.includes(":8080")) {
				return Promise.resolve([createEntry({ id: "model-1", context_length: 4096 })]);
			}
			return Promise.reject(new Error("second instance down"));
		});
		stubEmptyRunning();
		const pi = createMockPi();
		const config: { instances: LlamaSwapInstance[] } = {
			instances: [createInstance({ id: "llama-swap", port: 8080 }), createInstance({ id: "llama-swap-2", port: 9000 })],
		};

		const result = await provider.refreshProvider(pi as unknown as ExtensionAPI, config);

		expect(result.modelCount).toBe(1);
		expect(result.baseUrl).toBe("http://127.0.0.1:8080/v1, http://127.0.0.1:9000/v1");
		expect(result.error).toBe("http://127.0.0.1:9000/v1: second instance down");
	});

	it("reports observed running states keyed by instance id", async () => {
		const { provider, client } = await loadFresh();
		client.fetchModels.mockResolvedValue([createEntry({ id: "model-1" })]);
		mockFetch((url) => {
			if (String(url).endsWith("/running")) {
				return jsonResponse({ running: [{ model: "model-1", state: "starting" }] });
			}
			return jsonResponse({});
		});
		const pi = createMockPi();

		const result = await provider.refreshProvider(pi as unknown as ExtensionAPI, { instances: [createInstance()] });

		expect(result.runningStates).toEqual({ "llama-swap:model-1": "starting" });
	});

	it("aggregates running states across instances with prefixed keys", async () => {
		const { provider, client } = await loadFresh();
		client.fetchModels.mockImplementation((baseUrl: string) => {
			return Promise.resolve([createEntry({ id: "model-1" })]);
		});
		mockFetch((url) => {
			const target = String(url);
			if (target.endsWith("/running")) {
				const state = target.includes(":8080") ? "ready" : "starting";
				return jsonResponse({ running: [{ model: "model-1", state }] });
			}
			return jsonResponse({});
		});
		const pi = createMockPi();
		const config: { instances: LlamaSwapInstance[] } = {
			instances: [createInstance({ id: "llama-swap", port: 8080 }), createInstance({ id: "llama-swap-2", port: 9000 })],
		};

		const result = await provider.refreshProvider(pi as unknown as ExtensionAPI, config);

		expect(result.runningStates).toEqual({
			"llama-swap:model-1": "ready",
			"llama-swap-2:model-1": "starting",
		});
	});

	it("shares a single in-flight refresh across concurrent calls", async () => {
		const { provider, client } = await loadFresh();
		let resolveFetch: (v: unknown) => void = () => {};
		client.fetchModels.mockImplementation(() => new Promise((res) => (resolveFetch = res)));
		stubEmptyRunning();
		const pi = createMockPi();
		const config = { instances: [createInstance()] };

		const p1 = provider.refreshProvider(pi as unknown as ExtensionAPI, config);
		const p2 = provider.refreshProvider(pi as unknown as ExtensionAPI, config);
		await new Promise((r) => setTimeout(r, 0));

		// Both callers share one refresh: fetchModels ran once per instance.
		expect(client.fetchModels).toHaveBeenCalledTimes(1);
		resolveFetch([createEntry({ id: "model-1" })]);
		const [r1, r2] = await Promise.all([p1, p2]);
		expect(r1).toBe(r2);
	});

	it("unregisters before re-registering on a refresh after the first has settled", async () => {
		const { provider, client } = await loadFresh();
		client.fetchModels.mockResolvedValue([createEntry({ id: "model-1" })]);
		stubEmptyRunning();
		const pi = createMockPi();
		const config = { instances: [createInstance()] };

		await provider.refreshProvider(pi as unknown as ExtensionAPI, config);
		await provider.refreshProvider(pi as unknown as ExtensionAPI, config);

		expect(pi.registerProvider).toHaveBeenCalledTimes(2);
		expect(pi.unregisterProvider).toHaveBeenCalledTimes(1);
		expect(pi.unregisterProvider).toHaveBeenCalledWith("llama-swap");
		// The unregister must occur after the first register and before the second.
		const firstRegister = pi.registerProvider.mock.invocationCallOrder[0];
		const secondRegister = pi.registerProvider.mock.invocationCallOrder[1];
		const unregister = pi.unregisterProvider.mock.invocationCallOrder[0];
		expect(unregister).toBeGreaterThan(firstRegister);
		expect(unregister).toBeLessThan(secondRegister);
	});
});
