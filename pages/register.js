import React from 'react';
import AuthForm from '../components/AuthForm';

export default function RegisterPage() {
  return (
    <main className="mx-auto max-w-md p-6">
      <AuthForm mode="register" />
    </main>
  );
}
