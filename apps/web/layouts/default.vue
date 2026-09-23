<script setup lang="ts">
// Оболочка приложения (C-035): шапка с логотипом и переключателем темы, боковая навигация
// на десктопе, таб-бар на мобиле, футер с провенансом. UI общается только с gateway.
type Theme = 'light' | 'dark';

// C-055: масштабируемая навигация. Пять основных пунктов в таб-баре, остальные — в листе «Ещё»;
// на десктопе сайдбар показывает все пункты, поэтому ни одна страница не теряется.
const ICONS: Record<string, string> = {
  scan: 'M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M4 12h16',
  catalog: 'M4 5h16M4 12h16M4 19h16',
  sommelier: 'M8 3h8l-1 7a3 3 0 0 1-6 0zM12 10v8M9 21h6',
  metrics: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
  learning: 'M12 3l9 5-9 5-9-5 9-5zM5 12v5c0 1 3 2 7 2s7-1 7-2v-5',
  how: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zM12 17h.01M12 13a2 2 0 1 0-2-2',
};

const primaryNav = [
  { to: '/', label: 'Сканер', icon: 'scan' },
  { to: '/catalog', label: 'Каталог', icon: 'catalog' },
  { to: '/sommelier', label: 'Сомелье', icon: 'sommelier' },
  { to: '/metrics', label: 'Метрики', icon: 'metrics' },
  { to: '/learning', label: 'Обучение', icon: 'learning' },
];
const moreNav = [
  { to: '/how', label: 'Как это работает', icon: 'how' },
];
const allNav = [...primaryNav, ...moreNav];

const theme = ref<Theme>('light');

function applyTheme(next: Theme) {
  theme.value = next;
  if (import.meta.client) {
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('svoe-vino-theme', next); } catch { /* приватный режим */ }
  }
}

function toggleTheme() {
  applyTheme(theme.value === 'dark' ? 'light' : 'dark');
}

onMounted(() => {
  let stored: Theme | null = null;
  try { stored = localStorage.getItem('svoe-vino-theme') as Theme | null; } catch { stored = null; }
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(stored ?? (prefersDark ? 'dark' : 'light'));
});

const route = useRoute();
const isActive = (to: string) => (to === '/' ? route.path === '/' : route.path.startsWith(to));

const moreOpen = ref(false);
const sheet = ref<HTMLElement | null>(null);
const moreActive = computed(() => moreNav.some(item => isActive(item.to)));

async function openMore() {
  moreOpen.value = true;
  await nextTick();
  // Ловушка фокуса начинается с первого элемента листа.
  sheet.value?.querySelector<HTMLElement>('a, button')?.focus();
}

function closeMore() {
  moreOpen.value = false;
}

function trapFocus(event: KeyboardEvent) {
  // Фокус не должен уходить за пределы листа, пока он открыт.
  const nodes = sheet.value?.querySelectorAll<HTMLElement>('a, button');
  if (!nodes || !nodes.length || event.key !== 'Tab') return;
  const first = nodes[0];
  const last = nodes[nodes.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Escape' && moreOpen.value) closeMore();
}

onMounted(() => document.addEventListener('keydown', onKeydown));
onBeforeUnmount(() => document.removeEventListener('keydown', onKeydown));
watch(() => route.path, () => closeMore());
</script>

<template>
  <div class="shell">
    <header class="shell-header">
      <div class="shell-brand">
        <AppImage class="shell-logo" src="/media/logo-svoe-vino.svg" alt="Своё Вино" width="34" height="34" />
        <div>
          <div class="shell-title">Своё Вино</div>
          <p class="shell-sub">Сканер этикеток · демо конкурса</p>
        </div>
      </div>
      <button class="theme-toggle" type="button" :aria-label="theme === 'dark' ? 'Включить светлую тему' : 'Включить тёмную тему'" @click="toggleTheme">
        <svg v-if="theme === 'dark'" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
        </svg>
        <svg v-else width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
        <span class="small">{{ theme === 'dark' ? 'Светлая' : 'Тёмная' }}</span>
      </button>
    </header>

    <div class="shell-body">
      <nav class="shell-nav" aria-label="Основная навигация">
        <NuxtLink v-for="item in allNav" :key="item.to" :to="item.to" class="nav-link"
                  :class="{ active: isActive(item.to) }"
                  :aria-current="isActive(item.to) ? 'page' : undefined">
          <span class="nav-icon" aria-hidden="true">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <template v-if="item.icon === 'scan'"><path d="M4 8V6a2 2 0 0 1 2-2h2M16 4h2a2 2 0 0 1 2 2v2M20 16v2a2 2 0 0 1-2 2h-2M8 20H6a2 2 0 0 1-2-2v-2M4 12h16" /></template>
              <template v-else-if="item.icon === 'sommelier'"><path d="M8 3h8l-1 7a3 3 0 0 1-6 0zM12 10v8M9 21h6" /></template>
              <template v-else-if="item.icon === 'how'"><circle cx="12" cy="12" r="9" /><path d="M12 17h.01M12 13a2 2 0 1 0-2-2" /></template>
              <template v-else-if="item.icon === 'catalog'"><path d="M4 5h16M4 12h16M4 19h16" /></template>
              <template v-else><path d="M4 20V10M10 20V4M16 20v-7M22 20H2" /></template>
            </svg>
          </span>
          <span>{{ item.label }}</span>
        </NuxtLink>
      </nav>

      <main class="shell-content">
        <slot />
      </main>
    </div>

    <nav class="tabbar" aria-label="Основная навигация">
        <NuxtLink v-for="item in primaryNav" :key="item.to" :to="item.to" class="tabbar-link"
                  :class="{ active: isActive(item.to) }"
                  :aria-current="isActive(item.to) ? 'page' : undefined">
          <span class="nav-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
              <path :d="ICONS[item.icon]" />
            </svg>
          </span>
          <span class="small">{{ item.label }}</span>
        </NuxtLink>
        <button class="tabbar-link" type="button" :class="{ active: moreActive }"
                aria-haspopup="dialog" :aria-expanded="moreOpen ? 'true' : 'false'" @click="openMore">
          <span class="nav-icon" aria-hidden="true">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="5" cy="12" r="1.6" /><circle cx="12" cy="12" r="1.6" /><circle cx="19" cy="12" r="1.6" />
            </svg>
          </span>
          <span class="small">Ещё</span>
        </button>
      </nav>

      <div v-if="moreOpen" class="more-backdrop" @click="closeMore" />
      <div v-if="moreOpen" ref="sheet" class="more-sheet" role="dialog" aria-modal="true"
           aria-label="Дополнительные разделы" @keydown="trapFocus">
        <p class="more-title">Дополнительные разделы</p>
        <NuxtLink v-for="item in moreNav" :key="item.to" :to="item.to" class="nav-link"
                  :class="{ active: isActive(item.to) }"
                  :aria-current="isActive(item.to) ? 'page' : undefined">
          <span>{{ item.label }}</span>
        </NuxtLink>
        <button class="ghost" type="button" @click="closeMore">Закрыть</button>
      </div>

    <footer class="shell-footer">
      Данные и изображения портала «Своё Вино» · используются только для демонстрации на конкурсе.
      Распознавание и поиск работают локально.
    </footer>
  </div>
</template>
