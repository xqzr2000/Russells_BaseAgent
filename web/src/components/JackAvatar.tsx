import { useEffect, useState } from "react";
import portrait from "../assets/jack-portrait.webp";
import seated from "../assets/jack-seated.webp";
import sniffing from "../assets/jack-sniffing.webp";

export const JACK_AGENT = "coordinator";

export function JackAvatar({ working = false }: { working?: boolean }) {
  return (
    <span className={`jack-avatar ${working ? "jack-avatar-working" : ""}`} aria-hidden="true">
      {working ? (
        <>
          <img className="jack-seated" src={seated} alt="" />
          <img className="jack-sniffing" src={sniffing} alt="" />
        </>
      ) : (
        <img className="jack-portrait" src={portrait} alt="" />
      )}
    </span>
  );
}

export function JackResponseLabel({ waiting, title }: { waiting: boolean; title: string }) {
  const [waitElapsed, setWaitElapsed] = useState(false);

  useEffect(() => {
    if (!waiting) {
      setWaitElapsed(false);
      return;
    }
    const timer = window.setTimeout(() => setWaitElapsed(true), 1000);
    return () => window.clearTimeout(timer);
  }, [waiting]);

  const working = waiting && waitElapsed;

  return (
    <div className="response-label jack-response-label">
      <JackAvatar working={working} />
      <div>
        <span role="status" className="jack-status">{working && "Jack is working…"}</span>
        {!working && <span>{title}</span>}
      </div>
    </div>
  );
}
