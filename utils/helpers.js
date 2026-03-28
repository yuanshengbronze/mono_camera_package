export function calculateWpm(correctGestures, secondsElapsed) {
  if (!secondsElapsed) return 0;
  const minutes = secondsElapsed / 60;
  return Math.round(correctGestures / minutes);
}

export function calculateAccuracy(correct, total) {
  if (!total) return 0;
  return Math.round((correct / total) * 100);
}

export function calculateConsistency(results = []) {
  if (!results.length) return 100;
  const mean = results.reduce((sum, value) => sum + value, 0) / results.length;
  const variance =
    results.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / results.length;
  const deviation = Math.sqrt(variance);

  return Math.max(0, Math.round(100 - deviation));
}

export function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '-';
  return date.toLocaleDateString();
}
