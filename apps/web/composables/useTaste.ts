/**
 * «Мой вкус» (C-037): персонализация сомелье в localStorage.
 * Работает офлайн, ничего не отправляет на сервер кроме списка likes в запросе подбора.
 */
export interface TasteItem {
  slug: string; name: string; winery?: string; grape?: string; region?: string; category?: string;
}

const KEY = 'svoe-vino-taste';

export function useTaste() {
  const items = ref<TasteItem[]>([]);

  function load() {
    if (!import.meta.client) return;
    try {
      const raw = localStorage.getItem(KEY);
      items.value = raw ? (JSON.parse(raw) as TasteItem[]) : [];
    } catch {
      items.value = [];
    }
  }

  function persist() {
    if (!import.meta.client) return;
    try { localStorage.setItem(KEY, JSON.stringify(items.value)); } catch { /* приватный режим */ }
  }

  function has(slug: string) {
    return items.value.some(item => item.slug === slug);
  }

  function toggle(item: TasteItem) {
    if (!item?.slug) return;
    if (has(item.slug)) items.value = items.value.filter(entry => entry.slug !== item.slug);
    else items.value = [...items.value, item];
    persist();
  }

  function remove(slug: string) {
    items.value = items.value.filter(item => item.slug !== slug);
    persist();
  }

  function clear() {
    items.value = [];
    persist();
  }

  const likes = computed(() => items.value.map(item => item.slug));
  const profile = computed(() => {
    const grapes = new Set<string>();
    const regions = new Set<string>();
    const categories = new Set<string>();
    for (const item of items.value) {
      if (item.grape) grapes.add(item.grape);
      if (item.region) regions.add(item.region);
      if (item.category) categories.add(item.category);
    }
    return { grapes: [...grapes], regions: [...regions], categories: [...categories] };
  });

  onMounted(load);

  return { items, likes, profile, has, toggle, remove, clear, load };
}
