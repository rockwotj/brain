/** Milliseconds since epoch. Injected so tests and demos can fake time. */
export type Clock = () => number;

export const systemClock: Clock = () => Date.now();
