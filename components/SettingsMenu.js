import React from 'react';
import { DIFFICULTY_LEVELS } from '../utils/constants';

export default function SettingsMenu({ difficulty, onChange }) {
  return (
    <div className="space-y-2 rounded border p-4">
      <h2 className="text-xl font-semibold">Settings</h2>
      <p className="text-sm text-gray-500">Choose typing behavior profile.</p>
      <div className="flex gap-2">
        {DIFFICULTY_LEVELS.map((level) => (
          <button
            key={level}
            className={`rounded px-3 py-2 capitalize ${difficulty === level ? 'bg-blue-600 text-white' : 'border'}`}
            onClick={() => onChange(level)}
          >
            {level}
          </button>
        ))}
      </div>
    </div>
  );
}
