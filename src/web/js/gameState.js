const listeners = new Set();

// The latest server snapshot (unchanged payload) and the one before it.
export const game = { state: null, previous: null };

// Stores a fresh server snapshot and notifies every subscriber.
export function applyState(s) {
  game.previous = game.state;
  game.state = s;
  for (const fn of listeners) fn(s, game.previous);
}

// Registers a callback run after every state change; returns an unsubscribe function.
export function subscribe(fn) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}
