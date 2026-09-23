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
      // C-046/BUG-003: без htmlAttrs атрибут lang оставался пустым, что мешает доступности и SEO.
      htmlAttrs: { lang: 'ru' },
      title: 'Своё Вино — сканер этикеток',
      meta: [
        { name: 'viewport', content: 'width=device-width, initial-scale=1, viewport-fit=cover' },
        { name: 'theme-color', content: '#ffffff', media: '(prefers-color-scheme: light)' },
        { name: 'theme-color', content: '#141013', media: '(prefers-color-scheme: dark)' },
        { name: 'description', content: 'Сканируйте этикетку российского вина и получайте карточку с подсказками сомелье.' }
      ]
    }
  }
})
