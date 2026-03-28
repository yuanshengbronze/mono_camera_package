import React from 'react';
import useWebcam from '../hooks/useWebcam';

export default function Webcam({ onReady }) {
  const { videoRef, enabled, start, stop } = useWebcam();

  async function handleEnable() {
    await start();
    onReady?.();
  }

  return (
    <div className="space-y-3 rounded border p-4">
      <video ref={videoRef} className="h-56 w-full rounded bg-black object-cover" muted playsInline />
      <div className="flex gap-2">
        <button className="rounded bg-blue-600 px-3 py-2 text-white" onClick={handleEnable}>
          Enable webcam
        </button>
        <button className="rounded bg-gray-700 px-3 py-2 text-white" onClick={stop} disabled={!enabled}>
          Disable
        </button>
      </div>
    </div>
  );
}
