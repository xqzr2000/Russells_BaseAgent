import { useCallback, useEffect, useRef, useState } from "react";

/*
 * The full-window background: a still of the meadow with Jack in it.
 *
 * First impression: the very first time this browser opens the chat room,
 * if the window is no bigger than the video (so it never gets stretched),
 * Jack's video is armed. It plays once, when the user first starts typing,
 * then fades back to the still. A refresh, a later visit or a bigger window
 * only ever shows the still.
 *
 * Testing: add ?jack=1 to the URL to arm it again (the size rule still applies).
 */

const BASE = "/meadow/";
const STILL = `${BASE}meadow-jack.webp`;
/* MP4 (H.264) first; WebM (VP9) for Chromium builds without H.264. Both are silent. */
const VIDEOS = [
  { src: `${BASE}happy-jack.mp4`, type: "video/mp4" },
  { src: `${BASE}happy-jack.webm`, type: "video/webm" },
];
const VIDEO_W = 1344;
const VIDEO_H = 768;

const SEEN_KEY = "baseagent.meadow.firstImpression";
const START_WAIT_MS = 2000;
const FADE_OUT_MS = 900;

const reducedMotion = () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

/** True while the window is small enough to show the video at or below its native size. */
export const fitsVideo = () => window.innerWidth <= VIDEO_W && window.innerHeight <= VIDEO_H;

/** `?jack=1` re-arms the first impression for testing. */
export const jackForced = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("jack") === "1";

/**
 * Decided exactly once per page load, at module scope so React's StrictMode
 * double-run of effects can't decide it twice. The "seen" mark is written on
 * this first load whether or not the video gets to play.
 */
function decideFirstImpression(): boolean {
  if (typeof window === "undefined") return false;
  let seenBefore: boolean;
  try {
    seenBefore = window.localStorage.getItem(SEEN_KEY) !== null;
    window.localStorage.setItem(SEEN_KEY, new Date().toISOString());
  } catch {
    return false; // storage blocked: we couldn't promise "only once", so never play
  }
  if (seenBefore && !jackForced) return false;
  if (reducedMotion()) return false;
  return fitsVideo();
}

/** Whether this page load may show Jack's video (the app still checks there are no chats). */
export const firstImpression = decideFirstImpression();

type Phase = "idle" | "waiting" | "showing" | "fading" | "done";

type Props = {
  /** Load the video quietly so it can start the moment it's asked to. */
  armed: boolean;
  /** Start playing (set once, on the user's first keystroke). */
  play: boolean;
  /** Called once the video has finished (or was skipped) and the still is back. */
  onDone: () => void;
};

export function Meadow({ armed, play, onDone }: Props) {
  const [phase, setPhase] = useState<Phase>("idle");
  const videoRef = useRef<HTMLVideoElement>(null);
  const started = useRef(false);
  const finished = useRef(false);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  /** Fade back to the still (once), then report that the video is over. */
  const finish = useCallback(() => {
    if (finished.current) return;
    finished.current = true;
    setPhase("fading");
    window.setTimeout(
      () => {
        videoRef.current?.pause();
        setPhase("done");
        onDoneRef.current();
      },
      reducedMotion() ? 0 : FADE_OUT_MS,
    );
  }, []);

  // Start on request. play() is called straight away (rather than waiting for
  // the browser to preload), because iOS Safari doesn't buffer until then.
  useEffect(() => {
    if (!play || started.current) return;
    started.current = true;
    const el = videoRef.current;
    if (!el || !fitsVideo()) {
      finish();
      return;
    }
    setPhase("waiting");
    const giveUp = window.setTimeout(() => {
      if (el.paused || el.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) finish();
    }, START_WAIT_MS);
    const onPlaying = () => {
      window.clearTimeout(giveUp);
      if (!finished.current) setPhase("showing");
    };
    el.addEventListener("playing", onPlaying, { once: true });
    el.currentTime = 0;
    el.play()?.catch(() => finish());
    // No cleanup: this runs once, and the timer/listener guard themselves with `finished`.
  }, [play, finish]);

  // Never stretch the video: if the window grows past it while playing, go to the still.
  useEffect(() => {
    if (phase !== "waiting" && phase !== "showing") return;
    const onResize = () => !fitsVideo() && finish();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [phase, finish]);

  const showVideo = phase !== "done" && (armed || phase !== "idle");

  return (
    <div className="meadow" aria-hidden>
      <img className="meadow-still" src={STILL} alt="" draggable={false} />
      {showVideo && (
        <video
          ref={videoRef}
          className={`meadow-video is-${phase}`}
          muted
          playsInline
          preload="auto"
          disablePictureInPicture
          disableRemotePlayback
          onEnded={finish}
        >
          {VIDEOS.map((v, i) => (
            // An error on the last source means nothing could play: stay on the still.
            <source key={v.src} src={v.src} type={v.type} onError={i === VIDEOS.length - 1 ? finish : undefined} />
          ))}
        </video>
      )}
    </div>
  );
}
