export default defineNuxtConfig({
  compatibilityDate: '2025-01-01',
  devtools: { enabled: false },
  css: ['~/assets/css/main.css'],
  runtimeConfig: {
    public: {
      // Адрес gateway: единственная точка входа для UI.
      apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://127.0.0.1:8080'
    }
  },
  app: {
    head: {
      title: 'Своё Вино — сканер этикеток',
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1, viewport-fit=cover' },
        { name: 'theme-color', content: '#5b1a2b' },
        { name: 'description', content: 'Сканируйте этикетку российского вина и получайте карточку с подсказками сомелье.' }
      ]
    }
  }
})
