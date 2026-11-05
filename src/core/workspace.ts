import type { Bid, NewBid } from "./types";

/**
 * The global workspace: a tiny shared stage (≈7 slots). Anything that wins a slot is
 * broadcast to every loop that subscribes to it. Everything else stays local.
 *
 * Brain analogue: global workspace theory / conscious access, with lateral inhibition.
 */
export class Workspace {
  private items: Bid[] = [];
  private pending: Bid[] = [];
  private nextId = 1;

  constructor(
    private readonly capacity = 7,
    /** Global inhibition: one source can't flood the stage in a single tick. */
    private readonly maxBidsPerSource = 3,
    /** How much an old item's salience fades each tick. */
    private readonly decay = 0.7,
  ) {}

  /** Give a bid an id without entering it into competition (used for private feedback). */
  create(bid: NewBid): Bid {
    return { ttl: 3, ...bid, id: `b${this.nextId++}` };
  }

  /** Enter a bid into the next round of competition. */
  submit(bid: NewBid): Bid {
    const full = this.create(bid);
    this.pending.push(full);
    return full;
  }

  hasPending(): boolean {
    return this.pending.length > 0;
  }

  /** Current items plus bids waiting to compete. */
  candidates(): Bid[] {
    return [...this.items, ...this.pending];
  }

  contents(): readonly Bid[] {
    return this.items;
  }

  /**
   * One round of competition. Returns the bids that just entered (these get broadcast).
   *  1. old items fade (salience decays, ttl ticks down)
   *  2. new bids are capped per source (global inhibition)
   *  3. contradicting bids suppress each other (lateral inhibition)
   *  4. the executive's top-down bias scales every score
   *  5. the top `capacity` survive
   */
  compete(bias: (bid: Bid) => number): Bid[] {
    for (const item of this.items) {
      item.salience *= this.decay;
      item.ttl -= 1;
    }

    const fresh = this.capPerSource(this.pending);
    this.pending = [];
    const all = [...this.items.filter((i) => i.ttl > 0), ...fresh];

    this.inhibitContradictions(all);

    const score = (b: Bid) => b.salience * bias(b);
    const winners = all.sort((a, b) => score(b) - score(a)).slice(0, this.capacity);

    this.items = winners;
    return winners.filter((w) => fresh.includes(w));
  }

  private capPerSource(bids: Bid[]): Bid[] {
    const bySource = new Map<string, Bid[]>();
    for (const bid of bids) bySource.set(bid.source, [...(bySource.get(bid.source) ?? []), bid]);
    return [...bySource.values()].flatMap((group) =>
      group.sort((a, b) => b.salience - a.salience).slice(0, this.maxBidsPerSource),
    );
  }

  private inhibitContradictions(bids: Bid[]) {
    for (const bid of bids) {
      for (const targetId of bid.contradicts ?? []) {
        const target = bids.find((b) => b.id === targetId);
        if (!target) continue;
        const weaker = bid.salience < target.salience ? bid : target;
        weaker.salience *= 0.5;
      }
    }
  }
}
