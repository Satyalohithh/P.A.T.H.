import type { Alert, StreamStatus } from "../types";

export type AlertsState = {
  alerts: Alert[];
  setAlerts: (alerts: Alert[]) => void;
};

export type StreamsState = {
  streams: StreamStatus[];
  setStreams: (streams: StreamStatus[]) => void;
};

export function createAlertsStore(): AlertsState {
  // TODO(implementation): lightweight store (zustand-style) parity.
  return { alerts: [], setAlerts: () => {} };
}

export function createStreamsStore(): StreamsState {
  return { streams: [], setStreams: () => {} };
}