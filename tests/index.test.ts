import { afterEach, describe, expect, it, vi } from "vitest";

const { loadConfigMock, saveContextOverrideMock, configPathMock } = vi.hoisted(() => ({
	loadConfigMock: vi.fn(),
	saveContextOverrideMock: vi.fn().mockResolvedValue(undefined),
	configPathMock: vi.fn(() => "/fake/home/.pi/agent/pi-llama-swap.json"),
}));

const { refreshProviderMock } = vi.hoisted(() => ({
	refreshProviderMock: vi.fn(),
}));

vi.mock("../lib/config.js", () => ({
	loadConfig: loadConfigMock,
	saveContextOverride: saveContextOverrideMock,
	configPath: configPathMock,
}));

vi.mock("../lib/provider.js", () => ({
	refreshProvider: refreshProviderMock,
}));

import llamaSwapExtension from "../index.js";
import { createInstance, createMockCtx, createMockPi, createMockUi, emit, invokeCommand } from "./mocks.js";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const OK_REFRESH = { baseUrl: "http://127.0.0.1:8080/v1", modelCount: 1 };

/** Boots the extension against the mocks; returns the mock pi. */
async function boot() {
	loadConfigMock.mockResolvedValue({ instances: [createInstance()] });
	refreshProviderMock.mockResolvedValue(OK_REFRESH);
	const pi = createMockPi();
	await llamaSwapExtension(pi as unknown as ExtensionAPI);
	return pi;
}

afterEach(() => {
	vi.clearAllMocks();
});

describe("extension factory", () => {
	it("runs the initial refresh with isInitial and registers both handlers and commands", async () => {
		const pi = await boot();

		expect(refreshProviderMock).toHaveBeenCalledWith(pi, expect.anything(), { isInitial: true });
		expect(pi.handlers.has("after_provider_response")).toBe(true);
		expect(pi.handlers.has("model_select")).toBe(true);
		expect(pi.commands.has("llama-swap-refresh")).toBe(true);
		expect(pi.commands.has("llama-swap-set-context-length")).toBe(true);
	});
});

describe("after_provider_response", () => {
	it("does nothing when there is no model", async () => {
		const pi = await boot();
		const base = refreshProviderMock.mock.calls.length;
		await emit(pi, "after_provider_response", { type: "after_provider_response" }, createMockCtx());
		expect(refreshProviderMock.mock.calls.length).toBe(base);
	});

	it("does nothing when the model's provider has no matching instance", async () => {
		const pi = await boot();
		const base = refreshProviderMock.mock.calls.length;
		const ctx = createMockCtx(undefined, { provider: "other-provider", id: "model-1" });
		await emit(pi, "after_provider_response", { type: "after_provider_response" }, ctx);
		expect(refreshProviderMock.mock.calls.length).toBe(base);
	});

	it("refreshes a matching instance once per (provider, model)", async () => {
		const pi = await boot();
		const base = refreshProviderMock.mock.calls.length;
		const ctx = createMockCtx(undefined, { provider: "llama-swap", id: "model-1" });

		await emit(pi, "after_provider_response", { type: "after_provider_response" }, ctx);
		expect(refreshProviderMock.mock.calls.length).toBe(base + 1);

		// Same (provider, model) again → deduplicated, no new refresh.
		await emit(pi, "after_provider_response", { type: "after_provider_response" }, ctx);
		expect(refreshProviderMock.mock.calls.length).toBe(base + 1);
	});

	it("retries on the next response when the refresh reports an error", async () => {
		loadConfigMock.mockResolvedValue({ instances: [createInstance()] });
		refreshProviderMock.mockResolvedValue({ ...OK_REFRESH, error: "boom" });
		const pi = createMockPi();
		await llamaSwapExtension(pi as unknown as ExtensionAPI);
		const base = refreshProviderMock.mock.calls.length;
		const ctx = createMockCtx(undefined, { provider: "llama-swap", id: "model-1" });

		// Failed refresh is not recorded, so the next response refreshes again.
		await emit(pi, "after_provider_response", { type: "after_provider_response" }, ctx);
		expect(refreshProviderMock.mock.calls.length).toBe(base + 1);
		await emit(pi, "after_provider_response", { type: "after_provider_response" }, ctx);
		expect(refreshProviderMock.mock.calls.length).toBe(base + 2);
	});
});

describe("model_select", () => {
	it("pre-warms on a provider switch (fire-and-forget)", async () => {
		const pi = await boot();
		const base = refreshProviderMock.mock.calls.length;
		const event = {
			type: "model_select",
			model: { provider: "llama-swap", id: "m1" },
			previousModel: { provider: "some-other" },
		};
		emit(pi, "model_select", event, createMockCtx());
		await vi.waitFor(() => expect(refreshProviderMock.mock.calls.length).toBe(base + 1));
	});

	it("does nothing for a same-provider switch", async () => {
		const pi = await boot();
		const base = refreshProviderMock.mock.calls.length;
		const event = {
			type: "model_select",
			model: { provider: "llama-swap", id: "m1" },
			previousModel: { provider: "llama-swap" },
		};
		emit(pi, "model_select", event, createMockCtx());
		await new Promise((r) => setTimeout(r, 20));
		expect(refreshProviderMock.mock.calls.length).toBe(base);
	});

	it("ignores a second switch within the pre-warm cooldown", async () => {
		const pi = await boot();
		const base = refreshProviderMock.mock.calls.length;
		const ctx = createMockCtx();

		emit(
			pi,
			"model_select",
			{ type: "model_select", model: { provider: "llama-swap", id: "m1" }, previousModel: { provider: "a" } },
			ctx,
		);
		await vi.waitFor(() => expect(refreshProviderMock.mock.calls.length).toBe(base + 1));

		// Second switch to a different provider, well inside the 30s cooldown.
		emit(
			pi,
			"model_select",
			{ type: "model_select", model: { provider: "b", id: "m2" }, previousModel: { provider: "llama-swap" } },
			ctx,
		);
		await new Promise((r) => setTimeout(r, 20));
		expect(refreshProviderMock.mock.calls.length).toBe(base + 1);
	});
});

describe("/llama-swap-refresh", () => {
	it("notifies info on success", async () => {
		const pi = await boot();
		refreshProviderMock.mockResolvedValue({ baseUrl: "x", modelCount: 3 });
		const ui = createMockUi();
		await invokeCommand(pi, "llama-swap-refresh", "", createMockCtx(ui));
		expect(ui.notify).toHaveBeenCalledWith("[llama-swap] refreshed 3 models", "info");
	});

	it("notifies warning when the refresh reports an error", async () => {
		const pi = await boot();
		refreshProviderMock.mockResolvedValue({ baseUrl: "x", modelCount: 0, error: "boom" });
		const ui = createMockUi();
		await invokeCommand(pi, "llama-swap-refresh", "", createMockCtx(ui));
		expect(ui.notify).toHaveBeenCalledWith("[llama-swap] refresh finished with errors: boom", "warning");
	});

	it("notifies error when the refresh throws", async () => {
		const pi = await boot();
		refreshProviderMock.mockRejectedValue(new Error("exploded"));
		const ui = createMockUi();
		await invokeCommand(pi, "llama-swap-refresh", "", createMockCtx(ui));
		expect(ui.notify).toHaveBeenCalledWith("[llama-swap] refresh failed: exploded", "error");
	});
});

describe("/llama-swap-set-context-length", () => {
	it("warns when no model is selected", async () => {
		const pi = await boot();
		const ui = createMockUi();
		await invokeCommand(pi, "llama-swap-set-context-length", "4096", createMockCtx(ui));
		expect(ui.notify).toHaveBeenCalledWith("No model selected. Use /model first.", "warning");
	});

	it("warns when the selected model is not a llama-swap instance", async () => {
		const pi = await boot();
		const ui = createMockUi();
		const ctx = createMockCtx(ui, { provider: "other-provider", id: "model-1" });
		await invokeCommand(pi, "llama-swap-set-context-length", "4096", ctx);
		expect(ui.notify).toHaveBeenCalledWith("No llama-swap model selected. Use /model first.", "warning");
	});

	it("clears the override on 'auto' after confirmation", async () => {
		const pi = await boot();
		const ui = createMockUi(); // confirm resolves true by default
		const ctx = createMockCtx(ui, { provider: "llama-swap", id: "model-1" });
		await invokeCommand(pi, "llama-swap-set-context-length", "auto", ctx);

		expect(saveContextOverrideMock).toHaveBeenCalledWith("llama-swap", "model-1", undefined);
		expect(ui.notify).toHaveBeenCalledWith("Context override removed for model-1. Now auto-detected.");
	});

	it("does not save when the 'auto' confirmation is declined", async () => {
		const pi = await boot();
		const ui = createMockUi();
		ui.confirm.mockResolvedValue(false);
		const ctx = createMockCtx(ui, { provider: "llama-swap", id: "model-1" });
		await invokeCommand(pi, "llama-swap-set-context-length", "auto", ctx);
		expect(saveContextOverrideMock).not.toHaveBeenCalled();
	});

	it("errors on an invalid context size", async () => {
		const pi = await boot();
		const ui = createMockUi();
		const ctx = createMockCtx(ui, { provider: "llama-swap", id: "model-1" });
		await invokeCommand(pi, "llama-swap-set-context-length", "not-a-number", ctx);
		expect(ui.notify).toHaveBeenCalledWith(
			'Invalid context size. Use a positive integer or "auto".\nExample: /llama-swap-set-context-length 32768',
			"error",
		);
		expect(saveContextOverrideMock).not.toHaveBeenCalled();
	});

	it("saves a positive integer override after confirmation", async () => {
		const pi = await boot();
		const ui = createMockUi(); // confirm resolves true by default
		const ctx = createMockCtx(ui, { provider: "llama-swap", id: "model-1" });
		await invokeCommand(pi, "llama-swap-set-context-length", " 32768 ", ctx);

		expect(saveContextOverrideMock).toHaveBeenCalledWith("llama-swap", "model-1", 32768);
		expect(ui.notify).toHaveBeenCalledWith("Context window for model-1 set to 32768.");
	});

	it("does not save when the set confirmation is declined", async () => {
		const pi = await boot();
		const ui = createMockUi();
		ui.confirm.mockResolvedValue(false);
		const ctx = createMockCtx(ui, { provider: "llama-swap", id: "model-1" });
		await invokeCommand(pi, "llama-swap-set-context-length", "32768", ctx);
		expect(saveContextOverrideMock).not.toHaveBeenCalled();
	});
});
