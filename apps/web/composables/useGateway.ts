/**
 * Тонкий клиент gateway: единственная точка входа для UI.
 * Никаких обращений к retrieval/vlm/ocr/matcher напрямую — только gateway.
 */
import { requestTimeout } from '../utils/request';

export interface TopKItem {
  slug: string; score: number; name?: string | null; winery?: string | null;
  geo_score?: number; inliers?: number;
}

export interface MediaImage { name: string; url: string }

export interface Card {
  // C-018/C-034: карточка повторяет поля портала (официальные данные API vino-svoe.ru)
  name: string; winery: string; winery_slug?: string; winery_url?: string;
  region: string; category: string; color: string; grape: string; grapes?: string[];
  description: string; year?: number | null; sweetness?: string | null;
  alcohol_pct?: number | null; temperature?: string; dishes?: string[];
  image?: string | null; image_url?: string | null; api_match?: boolean;
  // C-036: медиа портала (аддитивно)
  category_gradient?: string; region_image_url?: string; region_geojson_url?: string;
  grape_images?: MediaImage[]; dish_images?: MediaImage[];
}

export interface TraceStage { name: string; latency_ms: number }
export interface Trace {
  stages?: TraceStage[]; methods?: Record<string, string>; flags?: Record<string, unknown>;
  top_k?: { slug: string; score: number; geo_score?: number; inliers?: number }[];
  confidence?: number; margin?: number; path?: string; notes?: string[];
}

export interface ScanResponse {
  slug: string | null; confidence: number; margin: number; path: string; latency_ms: number;
  top5: TopKItem[]; not_found: boolean; confident: boolean; degraded: boolean;
  card: Card | null; trace?: Trace;
}

export interface SimilarItem {
  slug: string; name: string; winery: string; region: string; category: string; grape: string;
  score: number; reason: string; image_url?: string; source?: string;
}

export interface SommelierItem { slug: string; name: string; winery: string; why: string }
export interface SommelierResponse {
  slug: string; recommendations: SommelierItem[]; pairing: string[];
  official_dishes?: string[]; source: string; note: string;
  // C-036: аддитивные поля
  pairing_cards?: MediaImage[]; sensory_note?: string;
  alternatives?: { slug: string; name: string; winery: string; image_url?: string }[];
}

// C-019: энотуризм и статьи — данные официального API портала через gateway
export interface EnotourismRegion {
  name: string; fact: string; events_count: number; events_description: string;
  image_url?: string; geojson_url?: string; wines_count?: number;
  manufacturers_count?: number; tours_count?: number;
}
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
  // C-044: обезличенная сессия связывает действия в интерфейсе с вызовами API и логами gateway.
  const telemetry = useTelemetry();
  const hdr = () => ({ 'X-Session-Id': telemetry.sessionId() });
  // C-049: без ограничения по времени «висящий» gateway (соединение принято, ответа нет)
  // оставлял интерфейс в скелетоне навсегда. Сканирование получает больший бюджет.
  const timeoutMs = Number(config.public.apiTimeoutMs || 8000);
  const scanTimeoutMs = Number(config.public.apiScanTimeoutMs || 15000);

  async function scan(file: File): Promise<ScanResponse> {
    const body = new FormData();
    body.append('image', file, file.name || 'label.jpg');
    return await telemetry.time('scan', 'scan_request',
      async () => await $fetch<ScanResponse>(`${base}/v1/scan`, { method: 'POST', body, headers: hdr(), signal: requestTimeout(scanTimeoutMs) }));
  }

  // C-039: браузер каталога и полная карточка по slug (additive)
  // C-046/BUG-006: у вспомогательных вызовов есть безопасный запасной результат, чтобы отказ
  // gateway показывался как состояние экрана, а не превращался в необработанный отказ промиса.
  async function catalog(params: Record<string, string | number> = {}): Promise<Record<string, any>> {
    const query = new URLSearchParams(
      Object.entries(params).filter(([, value]) => value !== '' && value !== undefined)
        .map(([key, value]) => [key, String(value)]) as [string, string][],
    ).toString();
    try {
      return await telemetry.time('catalog', 'catalog_request',
        async () => await $fetch<Record<string, any>>(`${base}/v1/catalog${query ? `?${query}` : ''}`, { headers: hdr(), signal: requestTimeout(timeoutMs) }));
    } catch {
      return { total: 0, items: [], facets: {}, unavailable: true };
    }
  }

  async function wine(slug: string, withEnotourism = false): Promise<Record<string, any>> {
    const suffix = withEnotourism ? '?with_enotourism=true' : '';
    try {
      return await telemetry.time('catalog', 'wine_request',
        async () => await $fetch<Record<string, any>>(`${base}/v1/wine/${encodeURIComponent(slug)}${suffix}`, { headers: hdr(), signal: requestTimeout(timeoutMs) }));
    } catch {
      return { unavailable: true };
    }
  }

  // C-038: скан с витриной пути решения (additive)
  async function scanExplain(file: File): Promise<ScanResponse> {
    const body = new FormData();
    body.append('image', file, file.name || 'label.jpg');
    // C-048/BUG-006: витрина пути решения — вспомогательный вызов. Он не должен создавать
    // необработанный отказ промиса, поэтому сбой возвращается как помеченный результат.
    try {
      return await telemetry.time('scan', 'scan_explain_request',
        async () => await $fetch<ScanResponse>(`${base}/v1/scan?explain=1`, { method: 'POST', body, headers: hdr(), signal: requestTimeout(scanTimeoutMs) }));
    } catch {
      return { unavailable: true } as unknown as ScanResponse;
    }
  }

  // C-038: метрики для дашборда (additive)
  async function demoMetrics(): Promise<Record<string, any>> {
    try {
      return await $fetch<Record<string, any>>(`${base}/v1/demo/metrics`, { headers: hdr(), signal: requestTimeout(timeoutMs) });
    } catch {
      return { unavailable: true };
    }
  }

  // C-054: голос пользователя за результат. Влияние определяет сервер, интерфейс лишь сообщает.
  async function feedback(payload: { vote: 'up' | 'down'; reason?: string[]; slug?: string;
    image_sha256?: string }): Promise<Record<string, any>> {
    try {
      return await telemetry.time('action', 'feedback_request',
        async () => await $fetch<Record<string, any>>(`${base}/v1/feedback`, {
          method: 'POST', body: payload, headers: hdr(), signal: requestTimeout(timeoutMs),
        }));
    } catch {
      throw new Error('feedback failed');
    }
  }

  // C-055: страница обучения читает агрегаты, очередь, датасет и отправляет решения модератора.
  function safeJson(path: string, options: Record<string, unknown> = {}) {
    return (async () => {
      try {
        return await $fetch<Record<string, any>>(`${base}${path}`,
          { headers: hdr(), signal: requestTimeout(timeoutMs), ...options });
      } catch {
        return { unavailable: true };
      }
    })();
  }

  async function learningSummary(): Promise<Record<string, any>> {
    return await safeJson('/v1/learning/summary');
  }

  async function learningQueue(): Promise<Record<string, any>> {
    return await safeJson('/v1/learning/queue');
  }

  async function learningDataset(): Promise<Record<string, any>> {
    return await safeJson('/v1/learning/dataset');
  }

  async function telemetrySummary(): Promise<Record<string, any>> {
    return await safeJson('/v1/telemetry/summary');
  }

  async function learningDecision(payload: { decision: 'accept' | 'reject'; slug?: string;
    image_sha256?: string; reason?: string }): Promise<Record<string, any>> {
    return await telemetry.time('action', 'moderator_decision',
      async () => await $fetch<Record<string, any>>(`${base}/v1/learning/decision`,
        { method: 'POST', body: payload, headers: hdr(), signal: requestTimeout(timeoutMs) }));
  }

  async function similar(slug: string): Promise<SimilarItem[]> {
    try {
      const response = await $fetch<{ items: SimilarItem[] }>(`${base}/v1/similar/${encodeURIComponent(slug)}`, { headers: hdr(), signal: requestTimeout(timeoutMs) });
      return response.items || [];
    } catch {
      return [];
    }
  }

  async function sommelier(slug: string, answers: Record<string, string>): Promise<SommelierResponse> {
    try {
      return await telemetry.time('sommelier', 'sommelier_request',
        async () => await $fetch<SommelierResponse>(`${base}/v1/sommelier`, {
          method: 'POST',
          body: { slug, answers },
          headers: hdr(),
          signal: requestTimeout(timeoutMs),
        }));
    } catch {
      return { slug, recommendations: [], pairing: [], source: 'unavailable', note: 'Сервис недоступен' } as SommelierResponse;
    }
  }

  async function enotourism(slug: string): Promise<EnotourismResponse> {
    try {
      return await $fetch<EnotourismResponse>(`${base}/v1/enotourism/${encodeURIComponent(slug)}`, { headers: hdr(), signal: requestTimeout(timeoutMs) });
    } catch {
      return { slug, unavailable: true } as unknown as EnotourismResponse;
    }
  }

  return { base, scan, scanExplain, demoMetrics, catalog, wine, similar, sommelier,
           enotourism, feedback, learningSummary, learningQueue, learningDataset,
           learningDecision, telemetrySummary };
}
