import { API_BASE_URL } from '../utils/constants';

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
    ...options,
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || 'API request failed');
  }

  return response.json();
}

export function inferGesture(imagePayload) {
  return request('/gesture/infer', {
    method: 'POST',
    body: JSON.stringify(imagePayload),
  });
}

export function getLeaderboard() {
  return request('/leaderboard');
}

export function submitRun(payload) {
  return request('/runs', {
    method: 'POST',
    body: JSON.stringify(payload),
  });
}
