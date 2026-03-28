export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000/api';

export const DEFAULT_DIFFICULTY = 'normal';
export const DIFFICULTY_LEVELS = ['easy', 'normal', 'hard'];

export const TEST_DURATION_SECONDS = 30;
export const LEADERBOARD_LIMIT = 50;

export const AUTH_PROVIDERS = {
  GOOGLE: 'google',
  GITHUB: 'github',
};
