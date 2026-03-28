import React, { useState } from 'react';
import TypingTest from '../components/TypingTest';
import Webcam from '../components/Webcam';
import StatsPanel from '../components/StatsPanel';
import { DEFAULT_DIFFICULTY, TEST_DURATION_SECONDS } from '../utils/constants';

export default function HomePage() {
  const [difficulty] = useState(DEFAULT_DIFFICULTY);
  const [summary, setSummary] = useState({
    correct: 0,
    total: 0,
    accuracy: 0,
    wpm: 0,
    consistency: 100,
  });

  return (
    <main className="mx-auto max-w-5xl space-y-6 p-6">
      <h1 className="text-4xl font-bold">Sign Language Race</h1>
      <p className="text-gray-500">
        Complete as many correct signs as possible in {TEST_DURATION_SECONDS} seconds.
      </p>
      <StatsPanel
        secondsLeft={TEST_DURATION_SECONDS}
        wpm={summary.wpm}
        accuracy={summary.accuracy}
        consistency={summary.consistency}
        correct={summary.correct}
        total={summary.total}
      />
      <div className="grid gap-6 lg:grid-cols-2">
        <TypingTest difficulty={difficulty} onComplete={setSummary} />
        <Webcam />
      </div>
    </main>
  );
}
