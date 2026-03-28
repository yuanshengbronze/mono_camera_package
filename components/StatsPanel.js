import React from 'react';

export default function StatsPanel({ secondsLeft, wpm, accuracy, consistency, correct, total }) {
  const items = [
    { label: 'Time Left', value: `${secondsLeft}s` },
    { label: 'WPM', value: wpm },
    { label: 'Accuracy', value: `${accuracy}%` },
    { label: 'Consistency', value: `${consistency}%` },
    { label: 'Correct Gestures', value: `${correct}/${total}` },
  ];

  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {items.map((item) => (
        <div key={item.label} className="rounded border bg-white/5 p-3 text-center">
          <div className="text-xs uppercase text-gray-500">{item.label}</div>
          <div className="text-xl font-semibold">{item.value}</div>
        </div>
      ))}
    </div>
  );
}
