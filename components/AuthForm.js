import React, { useState } from 'react';
import { loginWithEmail, loginWithOAuth, registerWithEmail } from '../services/auth';
import { validateAuthForm } from '../utils/validators';

export default function AuthForm({ mode = 'login' }) {
  const [form, setForm] = useState({ name: '', email: '', password: '' });
  const [errors, setErrors] = useState({});
  const [message, setMessage] = useState('');

  const isRegister = mode === 'register';

  function updateField(event) {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  }

  async function onSubmit(event) {
    event.preventDefault();
    const validation = validateAuthForm({ ...form, mode });
    setErrors(validation);
    if (Object.keys(validation).length) return;

    try {
      if (isRegister) {
        await registerWithEmail(form);
        setMessage('Registration complete. You can now sign in.');
      } else {
        await loginWithEmail(form);
        setMessage('Logged in successfully.');
      }
    } catch (error) {
      setMessage(error.message);
    }
  }

  return (
    <form className="space-y-3 rounded border p-4" onSubmit={onSubmit}>
      <h2 className="text-2xl font-semibold">{isRegister ? 'Register' : 'Login'}</h2>
      {isRegister && (
        <label className="block">
          Name
          <input className="mt-1 w-full rounded border px-3 py-2" name="name" value={form.name} onChange={updateField} />
          {errors.name && <div className="text-sm text-red-500">{errors.name}</div>}
        </label>
      )}
      <label className="block">
        Email
        <input className="mt-1 w-full rounded border px-3 py-2" name="email" value={form.email} onChange={updateField} />
        {errors.email && <div className="text-sm text-red-500">{errors.email}</div>}
      </label>
      <label className="block">
        Password
        <input
          className="mt-1 w-full rounded border px-3 py-2"
          type="password"
          name="password"
          value={form.password}
          onChange={updateField}
        />
        {errors.password && <div className="text-sm text-red-500">{errors.password}</div>}
      </label>
      <button className="w-full rounded bg-blue-600 px-3 py-2 text-white" type="submit">
        {isRegister ? 'Create account' : 'Login'}
      </button>
      <div className="grid grid-cols-2 gap-2">
        <button className="rounded border px-3 py-2" type="button" onClick={() => loginWithOAuth('google')}>
          Continue with Google
        </button>
        <button className="rounded border px-3 py-2" type="button" onClick={() => loginWithOAuth('github')}>
          Continue with GitHub
        </button>
      </div>
      {message && <div className="text-sm text-gray-500">{message}</div>}
    </form>
  );
}
