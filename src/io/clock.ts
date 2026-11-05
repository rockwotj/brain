/** Milliseconds since epoch. Injected so tests can control time. */
export type Clock = () => number;

export const systemClock: Clock = () => Date.now();
