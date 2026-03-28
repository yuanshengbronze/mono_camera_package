export function isValidEmail(email = '') {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

export function isStrongPassword(password = '') {
  return /^(?=.*[A-Za-z])(?=.*\d).{8,}$/.test(password);
}

export function validateAuthForm({ name, email, password, mode }) {
  const errors = {};

  if (mode === 'register' && (!name || name.trim().length < 2)) {
    errors.name = 'Name must have at least 2 characters';
  }

  if (!isValidEmail(email)) {
    errors.email = 'Please provide a valid email';
  }

  if (!isStrongPassword(password)) {
    errors.password = 'Password must be at least 8 chars and include a number';
  }

  return errors;
}
