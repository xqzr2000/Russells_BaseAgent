const paths = {
  brand: "M3 7.5 12 3l9 4.5-9 4.5L3 7.5ZM3 12l9 4.5 9-4.5M3 16.5l9 4.5 9-4.5",
  panel: "M4 4h16a1 1 0 0 1 1 1v14a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1ZM9 4v16",
  plus: "M12 5v14M5 12h14",
  close: "m6 6 12 12M18 6 6 18",
  chat: "M20 11.5a7.5 7.5 0 0 1-7.5 7.5H5l1.5-3A7.5 7.5 0 1 1 20 11.5Z",
  agents: "m12 3 2.3 6.7L21 12l-6.7 2.3L12 21l-2.3-6.7L3 12l6.7-2.3L12 3Z",
  data: "M4 4v16h16M9 15v-3m5 3V9m5 6V5",
  chess: "m7 13 3-4-2-2 2-4 2 2c4 0 5 4 5 7v4H7v-3Zm-1 3h12l1 4H5l1-4Z",
  sliders: "M4 7h7m4 0h5M4 17h3m4 0h9M11 4v6M7 14v6",
  attach: "m19 11-7.5 7.5a4 4 0 0 1-5.7-5.7l8-8a2.5 2.5 0 0 1 3.6 3.5l-8 8a1 1 0 0 1-1.4-1.4l7.2-7.2",
  send: "m4 9 16-5-5 16-3-8-8-3Zm8 3 8-8",
  stop: "M8 8h8v8H8z",
  trash: "M4 7h16M10 7V4h4v3m4 0-1 13H7L6 7m4 4v5m4-5v5",
} as const;

export type IconName = keyof typeof paths;

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
      strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d={paths[name]} />
    </svg>
  );
}
