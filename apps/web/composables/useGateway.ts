/**
 * Тонкий клиент gateway: единственная точка входа для UI.
 * Никаких обращений к retrieval/vlm/ocr/matcher напрямую — только gateway.
 */
export interface TopKItem { slug: string; score: number; name?: string | null; winery?: string | null }
export interface Card {
  name: string; winery: string; region: string; category: string; color: string;
  grape: string; description: string; year?: number | null; sweetness?: string | null; image?: string | null;
}
export interface ScanResponse {
  slug: string | null; confidence: number; margin: number; path: string; latency_ms: number;
  top5: TopKItem[]; not_found: boolean; confident: boolean; degraded: boolean; card: Card | null;
}
export interface SimilarItem {
  slug: string; name: string; winery: string; region: string; category: string; grape: string;
  score: number; reason: string;
}
export interface SommelierItem { slug: string; name: string; winery: string; why: string }
export interface SommelierResponse {
  slug: string; recommendations: SommelierItem[]; pairing: string[]; source: string; note: string;
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

  return { base, scan, similar, sommelier };
}
