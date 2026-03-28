import { useCallback, useRef, useState } from 'react';

export default function useWebcam() {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [enabled, setEnabled] = useState(false);

  const start = useCallback(async () => {
    const media = await navigator.mediaDevices.getUserMedia({ video: true });
    if (videoRef.current) {
      videoRef.current.srcObject = media;
      videoRef.current.play();
    }
    setStream(media);
    setEnabled(true);
  }, []);

  const stop = useCallback(() => {
    stream?.getTracks().forEach((track) => track.stop());
    setEnabled(false);
    setStream(null);
  }, [stream]);

  return {
    videoRef,
    enabled,
    start,
    stop,
  };
}
