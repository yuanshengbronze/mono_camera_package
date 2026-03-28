import { useEffect, useMemo, useState } from 'react';

export default function useTimer(initialSeconds, isRunning, onExpire) {
  const [secondsLeft, setSecondsLeft] = useState(initialSeconds);

  useEffect(() => {
    setSecondsLeft(initialSeconds);
  }, [initialSeconds]);

  useEffect(() => {
    if (!isRunning || secondsLeft <= 0) return;

    const id = setInterval(() => {
      setSecondsLeft((current) => {
        if (current <= 1) {
          clearInterval(id);
          onExpire?.();
          return 0;
        }
        return current - 1;
      });
    }, 1000);

    return () => clearInterval(id);
  }, [isRunning, secondsLeft, onExpire]);

  const secondsElapsed = useMemo(() => initialSeconds - secondsLeft, [initialSeconds, secondsLeft]);

  return {
    secondsLeft,
    secondsElapsed,
    reset: () => setSecondsLeft(initialSeconds),
  };
}
