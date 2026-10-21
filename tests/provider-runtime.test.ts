import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, describe, expect, it, vi } from "vitest";
import { registerLlamaSwapProvider } from "../lib/provider.js";
import { createInstance, jsonResponse, mockFetch } from "./mocks.js";

/**
 * Integration check against pi's real model runtime: pi calls a provider's
 * `refreshModels` when it refreshes catalogs (opening the `/model` picker does
 * that), so models llama-swap gained after startup must reach
 * `getAvailableSnapshot()`, which is what the picker renders.
 *
 * This is the only test that touches pi's runtime API rather than the hook
 * contract. If a pi bump breaks it, `provider.test.ts` still covers the hook
 * itself — update the `ModelRuntime.create`/`refresh` options here, don't drop
 * the check.
 */
describe("provider refresh through pi's ModelRuntime", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	/** Serves /v1/models from `ids` and an empty /running. */
	function stubLlamaSwap(ids: string[]) {
		return mockFetch((url) => {
			const u = String(url);
			if (u.endsWith("/models")) return jsonResponse({ object: "list", data: ids.map((id) => ({ id })) });
			if (u.endsWith("/running")) return jsonResponse({ running: [] });
			return jsonResponse({});
		});
	}

	it("publishes models llama-swap gained after the first registration", async () => {
		const { mkdtempSync } = await import("node:fs");
		const { join } = await import("node:path");
		const { tmpdir } = await import("node:os");
		const runtime = await ModelRuntime.create({
			modelsPath: null,
			authPath: join(mkdtempSync(tmpdir()), "auth.json"),
			refreshOnCreate: true,
		});

		stubLlamaSwap(["model-1"]);
		registerLlamaSwapProvider(
			{
				registerProvider: (id: string, config: unknown) => runtime.registerProvider(id, config as never),
				unregisterProvider: (id: string) => runtime.unregisterProvider(id),
			} as never,
			createInstance({ apiKey: "test-key" }),
			[],
		);

		await runtime.refresh({ providers: ["llama-swap"], signal: AbortSignal.timeout(5_000) });
		expect(runtime.getAvailableSnapshot().map((m) => m.id)).toEqual(["model-1"]);

		// llama-swap gained a model. pi passes force+allowNetwork for an explicit
		// catalog refresh (`pi update --models`); the picker relies on real elapsed
		// time instead, and the hook's reuse window is covered in provider.test.ts.
		stubLlamaSwap(["model-1", "model-2"]);
		await runtime.refresh({
			providers: ["llama-swap"],
			allowNetwork: true,
			force: true,
			signal: AbortSignal.timeout(5_000),
		});
		expect(runtime.getAvailableSnapshot().map((m) => m.id)).toEqual(["model-1", "model-2"]);
	});
});
