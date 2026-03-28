import React, { useMemo, useState } from 'react';
import useTimer from '../hooks/useTimer';
import { TEST_DURATION_SECONDS } from '../utils/constants';
import { calculateAccuracy, calculateConsistency, calculateWpm } from '../utils/helpers';

const SAMPLE_SIGNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'];

export default function TypingTest({ difficulty = 'normal', onComplete }) {
  const [targetIndex, setTargetIndex] = useState(0);
  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState([]);

  const { secondsLeft, secondsElapsed, reset } = useTimer(TEST_DURATION_SECONDS, isRunning, () => {
    setIsRunning(false);
    onComplete?.(summary);
  });

  const currentSign = SAMPLE_SIGNS[targetIndex % SAMPLE_SIGNS.length];
  const correct = results.filter(Boolean).length;
  const total = results.length;
  const accuracy = calculateAccuracy(correct, total);
  const wpm = calculateWpm(correct, Math.max(secondsElapsed, 1));
  const consistency = calculateConsistency(results.map((item) => (item ? 100 : 0)));

  const summary = useMemo(
    () => ({ correct, total, accuracy, wpm, consistency, difficulty }),
    [correct, total, accuracy, wpm, consistency, difficulty],
  );

  function startRound() {
    setResults([]);
    setTargetIndex(0);
    reset();
    setIsRunning(true);
  }

  function onGestureResult(isCorrect) {
    if (!isRunning) return;
    setResults((prev) => [...prev, isCorrect]);
    setTargetIndex((prev) => prev + 1);
  }

  return (
    <div className="space-y-4 rounded border p-4">
      <div className="text-sm uppercase tracking-wide text-gray-500">Difficulty: {difficulty}</div>
      <div className="rounded bg-black p-6 text-center text-6xl font-bold text-white">{currentSign}</div>
      <div className="flex flex-wrap gap-2">
        <button className="rounded bg-emerald-600 px-3 py-2 text-white" onClick={startRound}>
          Start 30s Test
        </button>
        <button className="rounded bg-green-700 px-3 py-2 text-white" onClick={() => onGestureResult(true)}>
          Mark Correct
        </button>
        <button className="rounded bg-red-700 px-3 py-2 text-white" onClick={() => onGestureResult(false)}>
          Mark Wrong
        </button>
      </div>
      <div className="text-sm text-gray-500">Time left: {secondsLeft}s</div>
    </div>
  );
}
