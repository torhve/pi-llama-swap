/**
 * Shared abort-signal helper for llama-swap probes.
 */

/** A combined abort signal plus the cleanup for its listener and timer. */
export interface AbortableTimeout {
	/** Signal that aborts when the caller aborts or the timeout elapses. */
	signal: AbortSignal;
	/** Removes the caller listener and clears the timer. Call once settled. */
	cleanup: () => void;
}

/**
 * Combines an optional caller signal with a timeout into one abort signal, so
 * a caller-initiated cancel (pi aborting a model-list refresh) and our own
 * timeout drive the same request.
 * @param signal - Caller signal that also aborts the request, or undefined.
 * @param timeoutMs - Abort after this many milliseconds.
 * @returns The combined signal and its cleanup.
 */
export function withAbortTimeout(signal: AbortSignal | undefined, timeoutMs: number): AbortableTimeout {
	const controller = new AbortController();
	const abort = () => controller.abort();
	if (signal?.aborted) {
		abort();
	} else {
		signal?.addEventListener("abort", abort, { once: true });
	}
	const timeout = setTimeout(abort, timeoutMs);
	return {
		signal: controller.signal,
		cleanup: () => {
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abort);
		},
	};
}
