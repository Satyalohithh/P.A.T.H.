import type { Alert, StreamStatus } from "../types";

export type UseAlertsResult = {
  alerts: Alert[];
  loading: boolean;
  ack: (alertId: string) => Promise<void>;
};

export function useAlerts(): UseAlertsResult {
  // TODO(implementation): subscribe to /ws, query /api/v1/alerts.
  return { alerts: [], loading: true, ack: async () => {} };
}

export type UseStreamsResult = {
  streams: StreamStatus[];
  register: (uri: string) => Promise<void>;
};

export function useStreams(): UseStreamsResult {
  // TODO(implementation): query /api/v1/streams, POST /api/v1/streams.
  return { streams: [], register: async () => {} };
}