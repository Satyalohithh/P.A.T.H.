import type { ReactNode } from "react";

export type CardProps = {
  title: string;
  children: ReactNode;
};

export function Card(_props: CardProps) {
  return null;
}

export type PillProps = {
  status: string;
  label: string;
};

export function Pill(_props: PillProps) {
  return null;
}