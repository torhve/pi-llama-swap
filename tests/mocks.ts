/**
 * Shared mock factories for the pi-llama-swap test suite.
 *
 * These mirror the testing strategy of the reference repo: a small set of
 * `vi.fn()`-based factories that each test configures as needed. No real
 * network, filesystem, or pi runtime is touched — every boundary goes through
 * a mock.
 */
import { vi } from "vitest";

import type { LlamaSwapInstance, OpenAIModelEntry } from "../lib/types.js";

/** Minimal model shape the extension actually reads (`provider`, `id`). */
export interface MockModel {
	provider: string;
	id: string;
	name?: string;
}

/** An event/command handler with loose args (test-side storage only). */
export type AnyHandler = (event: any, ctx: any) => any;

export interface MockCommand {
	description?: string;
	handler: (args: string, ctx: any) => Promise<void>;
}

/** A pi extension API stub recording registrations and event handlers. */
export interface MockPi {
	/** Registered event handlers by event name (in registration order). */
	handlers: Map<string, AnyHandler[]>;
	/** Last registered provider config per provider id. */
	providers: Map<string, unknown>;
	/** Registered commands by name. */
	commands: Map<string, MockCommand>;
	on: (event: string, handler: AnyHandler) => void;
	registerProvider: ReturnType<typeof vi.fn>;
	unregisterProvider: ReturnType<typeof vi.fn>;
	registerCommand: ReturnType<typeof vi.fn>;
}

/**
 * Creates a mock pi extension API.
 * `on` stores handlers so `emit` can dispatch; `registerProvider` /
 * `unregisterProvider` are spies that also record into `providers`;
 * `registerCommand` stores the command definition for `invokeCommand`.
 */
export function createMockPi(): MockPi {
	const handlers = new Map<string, AnyHandler[]>();
	const providers = new Map<string, unknown>();
	const commands = new Map<string, MockCommand>();
	return {
		handlers,
		providers,
		commands,
		on: (event: string, handler: AnyHandler) => {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
		},
		registerProvider: vi.fn((id: string, config: unknown) => {
			providers.set(id, config);
		}),
		unregisterProvider: vi.fn((id: string) => {
			providers.delete(id);
		}),
		registerCommand: vi.fn((name: string, def: MockCommand) => {
			commands.set(name, def);
		}),
	};
}

/**
 * Dispatches `payload` to every handler registered for `event`, awaiting each.
 * Used for `after_provider_response` (handler is awaited) — for fire-and-forget
 * events like `model_select`, assert with `vi.waitFor` instead.
 */
export async function emit(pi: MockPi, event: string, payload: unknown, ctx: unknown): Promise<void> {
	const list = pi.handlers.get(event) ?? [];
	for (const handler of list) {
		await handler(payload, ctx);
	}
}

/** Invokes a registered command handler with `args` and `ctx`. */
export async function invokeCommand(pi: MockPi, name: string, args: string, ctx: unknown): Promise<void> {
	const cmd = pi.commands.get(name);
	if (!cmd) {
		throw new Error(`command not registered: ${name}`);
	}
	await cmd.handler(args, ctx);
}

/** Mock UI context: `confirm` resolves true by default. */
export function createMockUi() {
	return {
		notify: vi.fn(),
		confirm: vi.fn().mockResolvedValue(true),
		select: vi.fn(),
	};
}

/** Mock extension context with a ui, optional model, and a fixed cwd. */
export function createMockCtx(ui?: ReturnType<typeof createMockUi>, model?: MockModel) {
	return {
		ui: ui ?? createMockUi(),
		model,
		cwd: "/tmp/test",
	};
}

/** A llama-swap instance with the documented defaults, overridable per test. */
export function createInstance(overrides: Partial<LlamaSwapInstance> = {}): LlamaSwapInstance {
	return {
		id: "llama-swap",
		name: "Llama Swap",
		origin: "http://127.0.0.1",
		port: 8080,
		...overrides,
	};
}

/** An OpenAI /v1/models entry with a default id, overridable per test. */
export function createEntry(overrides: Partial<OpenAIModelEntry> = {}): OpenAIModelEntry {
	return {
		id: "model-1",
		...overrides,
	};
}

/**
 * Stubs the global `fetch` with a `vi.fn` wrapping `handler`.
 * Returns the spy so tests can assert on calls and set per-test behavior.
 * Pair with `vi.unstubAllGlobals()` in `afterEach`.
 */
export function mockFetch(handler: (url: string | URL | Request, init?: RequestInit) => Response | Promise<Response>) {
	const spy = vi.fn(handler);
	vi.stubGlobal("fetch", spy);
	return spy;
}

/** Builds a JSON `Response` with the given body and status (default 200). */
export function jsonResponse(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), {
		status,
		headers: { "Content-Type": "application/json" },
	});
}
