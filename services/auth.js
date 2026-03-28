import { API_BASE_URL } from '../utils/constants';

async function authRequest(path, payload) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error('Authentication request failed');
  }

  return response.json();
}

export function loginWithEmail({ email, password }) {
  return authRequest('/auth/login', { email, password });
}

export function registerWithEmail({ name, email, password }) {
  return authRequest('/auth/register', { name, email, password });
}

export function loginWithOAuth(provider) {
  window.location.href = `${API_BASE_URL}/auth/oauth/${provider}`;
}

export function signOut() {
  return authRequest('/auth/logout', {});
}
