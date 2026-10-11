import { useCallback, useEffect, useRef, useState } from "react";
import type { RefObject } from "react";

/*
 * The chat pane's background: Jack's intro video plays once per page load,
 * then the still meadow with one Jack cut-out that moves to a new random
 * spot every few seconds.
 *
 * Everything is drawn inside a 16:9 "scene" that covers the pane (like
 * object-fit: cover), so positions are stored as fractions of the scene and
 * stay on the same patch of grass when the window is resized.
 */

const BASE = "/meadow/";
const STILL = `${BASE}meadow.webp`;
/* MP4 (H.264) first; WebM (VP9) for Chromium builds without H.264. */
const VIDEOS = [
  { src: `${BASE}jack-intro.mp4`, type: "video/mp4" },
  { src: `${BASE}jack-intro.webm`, type: "video/webm" },
];

/* Ground model measured from the intro video: a dog whose feet are at scene
   height y is about K * (y - HORIZON) of the scene's height tall. */
const HORIZON = 0.454;
const K = 1.74;
const FEET_MIN = 0.5;
const FEET_MAX = 0.97;

/* Where video-Jack sits in the last frame, so the cut-out can take over. */
const SIT_SPOT = { pose: 4, cx: 0.508, feet: 0.948 };

const SWAP_EVERY_MS = 5000;
const FADE_MS = 450;
const VIDEO_FADE_MS = 900;
const VIDEO_START_TIMEOUT_MS = 6000;

type Pose = { src: string; aspect: number; srcHeight: number };

/* aspect = width / height of each trimmed cut-out */
const POSES: Pose[] = [
  { src: `${BASE}jack-run-far.webp`, aspect: 161 / 198, srcHeight: 198 },
  { src: `${BASE}jack-run-mid.webp`, aspect: 280 / 397, srcHeight: 397 },
  { src: `${BASE}jack-run-near.webp`, aspect: 391 / 639, srcHeight: 639 },
  { src: `${BASE}jack-sniff.webp`, aspect: 327 / 547, srcHeight: 547 },
  { src: `${BASE}jack-sit.webp`, aspect: 313 / 543, srcHeight: 543 },
];

/** A dog on the grass, in scene fractions: centre x, feet y, height. */
type Placement = { pose: number; cx: number; feet: number; h: number };

const heightAt = (feet: number) => K * (feet - HORIZON);

const reducedMotion = () =>
  typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

function overlap(a: DOMRect | Box, b: DOMRect) {
  const w = Math.min(a.right, b.right) - Math.max(a.left, b.left);
  const h = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
  return w > 0 && h > 0 ? w * h : 0;
}

type Box = { left: number; right: number; top: number; bottom: number };

/**
 * Pick a new spot: on the grass, sized for its distance, fully inside the
 * pane, and preferably not hidden behind the chat panel or the composer.
 */
function pickPlacement(scene: DOMRect, pane: DOMRect, avoid: DOMRect[], prev: Placement | null): Placement {
  const candidates: { p: Placement; visible: number }[] = [];
  const maxPx = Math.min(pane.height * 0.62, 560);
  for (let i = 0; i < 160; i++) {
    let pose = Math.floor(Math.random() * POSES.length);
    if (prev && pose === prev.pose) pose = (pose + 1 + Math.floor(Math.random() * (POSES.length - 1))) % POSES.length;
    const { aspect, srcHeight } = POSES[pose];
    const feet = FEET_MIN + Math.random() * (FEET_MAX - FEET_MIN);
    const h = heightAt(feet);
    const px = h * scene.height;
    // Not a speck, not taller than the pane can show, not blown up past ~1.4x the artwork.
    if (px < 56 || px > maxPx || px > srcHeight * 1.4) continue;
    const w = px * aspect;
    const cx = Math.random();
    const box: Box = {
      left: scene.left + cx * scene.width - w / 2,
      right: scene.left + cx * scene.width + w / 2,
      top: scene.top + feet * scene.height - px,
      bottom: scene.top + feet * scene.height,
    };
    if (box.left < pane.left + 6 || box.right > pane.right - 6 || box.top < pane.top || box.bottom > pane.bottom) continue;
    if (prev && Math.abs(cx - prev.cx) * scene.width < Math.min(pane.width * 0.12, 140) && Math.abs(feet - prev.feet) < 0.08) continue;
    const area = w * px;
    const hidden = avoid.reduce((sum, r) => sum + overlap(box, r), 0);
    candidates.push({ p: { pose, cx, feet, h }, visible: Math.max(0, 1 - hidden / area) });
  }
  if (!candidates.length) {
    // Tiny pane: fall back to a small dog in the middle distance.
    const feet = 0.6;
    return { pose: prev?.pose === 1 ? 0 : 1, cx: 0.5, feet, h: heightAt(feet) };
  }
  const good = candidates.filter((c) => c.visible >= 0.6);
  if (good.length) return good[Math.floor(Math.random() * good.length)].p;
  // No open grass (phones, narrow windows): keep the dog small so it reads as a
  // soft shape through the frosted panel rather than a blur behind the text.
  const small = candidates.filter((c) => c.p.h * scene.height <= pane.height * 0.26);
  const pool = (small.length ? small : candidates).sort((a, b) => b.visible - a.visible).slice(0, 6);
  return pool[Math.floor(Math.random() * pool.length)].p;
}

type Props = {
  /** Set to true to cut the intro short (the user started typing, dropped a file, ...). */
  skip: boolean;
  /** Called once when the intro is over, whether it finished or was skipped. */
  onIntroEnd: () => void;
  /** Elements the dog should try not to hide behind. */
  avoid: RefObject<HTMLElement | null>[];
};

export function Meadow({ skip, onIntroEnd, avoid }: Props) {
  const [still] = useState(reducedMotion);
  const [video, setVideo] = useState<"playing" | "fading" | "gone">(still ? "gone" : "playing");
  const [dog, setDog] = useState<Placement | null>(null);
  const [dogShown, setDogShown] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const introOver = useRef(false);
  const dogRef = useRef<Placement | null>(null);
  dogRef.current = dog;
  // Props live in refs so a parent re-render (e.g. streaming a reply) doesn't restart the timers.
  const avoidRef = useRef(avoid);
  avoidRef.current = avoid;
  const onIntroEndRef = useRef(onIntroEnd);
  onIntroEndRef.current = onIntroEnd;

  const nextPlacement = useCallback(() => {
    const scene = sceneRef.current?.getBoundingClientRect();
    const pane = rootRef.current?.getBoundingClientRect();
    if (!scene || !pane || !pane.width) return null;
    const rects = avoidRef.current.map((r) => r.current?.getBoundingClientRect()).filter((r): r is DOMRect => !!r && r.width > 0);
    return pickPlacement(scene, pane, rects, dogRef.current);
  }, []);

  const endIntro = useCallback(
    (finished: boolean) => {
      if (introOver.current) return;
      introOver.current = true;
      if (finished) {
        // Swap video-Jack for the sitting cut-out in the same spot, then fade the video away.
        setDog({ ...SIT_SPOT, h: heightAt(SIT_SPOT.feet) });
        setDogShown(true);
      } else {
        const p = nextPlacement();
        if (p) {
          setDog(p);
          requestAnimationFrame(() => setDogShown(true));
        }
      }
      setVideo((v) => (v === "gone" ? v : "fading"));
      window.setTimeout(() => setVideo("gone"), VIDEO_FADE_MS);
      onIntroEndRef.current();
    },
    [nextPlacement],
  );

  // Preload the cut-outs so the first swap doesn't flash.
  useEffect(() => {
    for (const p of POSES) new Image().src = p.src;
  }, []);

  // Reduced motion: no video, start straight on the meadow.
  useEffect(() => {
    if (still) endIntro(false);
  }, [still, endIntro]);

  // Start the video; if it can't autoplay or never starts, skip to the meadow.
  useEffect(() => {
    if (still) return;
    const el = videoRef.current;
    if (!el) return;
    let started = false;
    const onPlaying = () => (started = true);
    el.addEventListener("playing", onPlaying);
    el.play()?.catch(() => endIntro(false));
    const watchdog = window.setTimeout(() => {
      if (!started && !document.hidden) endIntro(false);
    }, VIDEO_START_TIMEOUT_MS);
    return () => {
      el.removeEventListener("playing", onPlaying);
      window.clearTimeout(watchdog);
    };
  }, [still, endIntro]);

  useEffect(() => {
    if (skip) endIntro(false);
  }, [skip, endIntro]);

  // Every few seconds: fade the dog out, move it, fade it back in. Paused while the tab is hidden.
  useEffect(() => {
    if (!dog) return;
    let timer = 0;
    let inner = 0;
    const swap = () => {
      const fade = reducedMotion() ? 0 : FADE_MS;
      setDogShown(false);
      inner = window.setTimeout(() => {
        const p = nextPlacement();
        if (p) setDog(p);
        requestAnimationFrame(() => requestAnimationFrame(() => setDogShown(true)));
      }, fade);
    };
    const schedule = () => {
      window.clearTimeout(timer);
      if (!document.hidden) timer = window.setTimeout(swap, SWAP_EVERY_MS);
    };
    schedule();
    document.addEventListener("visibilitychange", schedule);
    return () => {
      window.clearTimeout(timer);
      window.clearTimeout(inner);
      document.removeEventListener("visibilitychange", schedule);
    };
    // Re-arm after each move (dog changes), so the dog stays put for the full interval.
  }, [dog, nextPlacement]);

  const pose = dog ? POSES[dog.pose] : null;

  return (
    <div
      className={`meadow ${video !== "gone" ? "is-intro" : ""}`}
      ref={rootRef}
      onClick={() => !introOver.current && endIntro(false)}
    >
      <div className="meadow-scene" ref={sceneRef} aria-hidden>
        <img className="meadow-still" src={STILL} alt="" draggable={false} />
        {dog && pose && (
          <div
            className={`meadow-dog ${dogShown ? "is-shown" : ""}`}
            style={{ left: `${dog.cx * 100}%`, top: `${dog.feet * 100}%`, height: `${dog.h * 100}%` }}
          >
            <span className="meadow-dog-shadow" />
            <img src={pose.src} alt="" draggable={false} />
          </div>
        )}
        {video !== "gone" && (
          <video
            ref={videoRef}
            className={`meadow-video ${video === "fading" ? "is-fading" : ""}`}
            poster={STILL}
            muted
            playsInline
            autoPlay
            preload="auto"
            onEnded={() => endIntro(true)}
          >
            {VIDEOS.map((v, i) => (
              // An error on the last source means nothing could play: skip to the meadow.
              <source key={v.src} src={v.src} type={v.type} onError={i === VIDEOS.length - 1 ? () => endIntro(false) : undefined} />
            ))}
          </video>
        )}
      </div>
      {video === "playing" && (
        <button type="button" className="meadow-skip" onClick={() => endIntro(false)}>
          Skip intro
        </button>
      )}
    </div>
  );
}
