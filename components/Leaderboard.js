import React from 'react';
import useLeaderboard from '../hooks/useLeaderboard';
import { formatDate } from '../utils/helpers';

export default function Leaderboard() {
  const { rows, loading, error, refresh } = useLeaderboard();

  if (loading) return <div>Loading leaderboard...</div>;
  if (error) return <div className="text-red-500">Failed to load: {error}</div>;

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold">Leaderboard</h2>
        <button className="rounded border px-3 py-2" onClick={refresh}>
          Refresh
        </button>
      </div>
      <div className="overflow-auto rounded border">
        <table className="min-w-full text-left text-sm">
          <thead className="bg-gray-900 text-white">
            <tr>
              <th className="px-3 py-2">Player</th>
              <th className="px-3 py-2">WPM</th>
              <th className="px-3 py-2">Accuracy</th>
              <th className="px-3 py-2">Consistency</th>
              <th className="px-3 py-2">Date</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.id} className="border-t">
                <td className="px-3 py-2">{row.name}</td>
                <td className="px-3 py-2">{row.wpm}</td>
                <td className="px-3 py-2">{row.accuracy}%</td>
                <td className="px-3 py-2">{row.consistency}%</td>
                <td className="px-3 py-2">{formatDate(row.createdAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
