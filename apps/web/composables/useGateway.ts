/**
 * Тонкий клиент gateway: единственная точка входа для UI.
 * Никаких обращений к retrieval/vlm/ocr/matcher напрямую — только gateway.
 */
export interface TopKItem { slug: string; score: number; name?: string | null; winery?: string | null }
export interface Card {
  // C-018: карточка повторяет поля портала (официальные данные API vino-svoe.ru)
  name: string; winery: string; winery_slug?: string; winery_url?: string;
  region: string; category: string; color: string; grape: string; grapes?: string[];
  description: string; year?: number | null; sweetness?: string | null;
  alcohol_pct?: number | null; temperature?: string; dishes?: string[];
  image?: string | null; image_url?: string | null; api_match?: boolean;
}
export interface ScanResponse {
  slug: string | null; confidence: number; margin: number; path: string; latency_ms: number;
  top5: TopKItem[]; not_found: boolean; confident: boolean; degraded: boolean; card: Card | null;
}
export interface SimilarItem {
  slug: string; name: string; winery: string; region: string; category: string; grape: string;
  score: number; reason: string; image_url?: string; source?: string;
}
export interface SommelierItem { slug: string; name: string; winery: string; why: string }
export interface SommelierResponse {
  slug: string; recommendations: SommelierItem[]; pairing: string[];
  official_dishes?: string[]; source: string; note: string;
}
// C-019: энотуризм и статьи — данные официального API портала через gateway
export interface EnotourismRegion { name: string; fact: string; events_count: number; events_description: string }
export interface EnotourismTour {
  slug: string; title: string; address: string; price?: number | null;
  image_url: string; url: string; match: string;
}
export interface EnotourismArticle {
  slug: string; title: string; category: string; author: string;
  published_at: string; image_url: string; url: string; relevance: string;
}
export interface EnotourismResponse {
  slug: string; region: EnotourismRegion; tours: EnotourismTour[];
  articles: EnotourismArticle[]; source: string; note: string;
}

export function useGateway() {
  const config = useRuntimeConfig();
  const base = String(config.public.apiBase || '').replace(/\/$/, '');

  async function scan(file: File): Promise<ScanResponse> {
    const body = new FormData();
    body.append('image', file, file.name || 'label.jpg');
    return await $fetch<ScanResponse>(`${base}/v1/scan`, { method: 'POST', body });
  }

  async function similar(slug: string): Promise<SimilarItem[]> {
    const data = await $fetch<{ items: SimilarItem[] }>(`${base}/v1/similar/${encodeURIComponent(slug)}`);
    return data.items || [];
  }

  async function sommelier(slug: string, answers: Record<string, string>): Promise<SommelierResponse> {
    return await $fetch<SommelierResponse>(`${base}/v1/sommelier`, {
      method: 'POST',
      body: { slug, answers }
    });
  }

  async function enotourism(slug: string): Promise<EnotourismResponse> {
    return await $fetch<EnotourismResponse>(`${base}/v1/enotourism/${encodeURIComponent(slug)}`);
  }

  return { base, scan, similar, sommelier, enotourism };
}
