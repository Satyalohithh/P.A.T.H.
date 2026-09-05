export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
export type BehaviorClass = "normal" | "playful" | "suspicious" | "aggressive";

export type Alert = {
  id: string;
  streamId: string;
  status: "active" | "acknowledged" | "resolved";
  message: string;
  risk: { score: number; level: RiskLevel; frameIndex: number; trackId: number };
  createdAt: number;
};

export type StreamTileProps = {
  streamId: string;
  source: string;
};

export type StreamStatus = {
  streamId: string;
  uri: string;
  state: "started" | "stopped" | "error";
  fps: number;
};