import type { Alert, StreamStatus } from "../types";

export const API_BASE = "/api/v1";
export const WS_URL = "/ws";

export async function listStreams(): Promise<StreamStatus[]> {
  // TODO(implementation): GET ${API_BASE}/streams
  throw new Error("TODO(implementation): listStreams");
}

export async function listAlerts(limit = 100): Promise<Alert[]> {
  // TODO(implementation): GET ${API_BASE}/alerts?limit=${limit}
  throw new Error("TODO(implementation): listAlerts");
}

export async function acknowledgeAlert(alertId: string): Promise<void> {
  // TODO(implementation): POST ${API_BASE}/alerts/${alertId}/ack
  throw new Error("TODO(implementation): acknowledgeAlert");
}

export function connectAlertSocket(onAlert: (alert: Alert) => void): () => void {
  // TODO(implementation): open WebSocket(WS_URL) and wire onAlert.
  return () => {};
}