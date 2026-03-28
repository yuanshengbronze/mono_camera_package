import React, { useState } from 'react';
import SettingsMenu from '../components/SettingsMenu';
import { DEFAULT_DIFFICULTY } from '../utils/constants';

export default function SettingsPage() {
  const [difficulty, setDifficulty] = useState(DEFAULT_DIFFICULTY);

  return (
    <main className="mx-auto max-w-3xl space-y-6 p-6">
      <h1 className="text-3xl font-bold">Settings</h1>
      <SettingsMenu difficulty={difficulty} onChange={setDifficulty} />
      <p className="text-sm text-gray-500">Selected mode: {difficulty}</p>
    </main>
  );
}
