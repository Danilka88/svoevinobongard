// Читаемость текста на градиентах категорий. Цвет фона приходит из данных портала
// (category_gradient), среди них много светлых (#FCFBDF → #F0EFD5), поэтому белый текст
// на них невидим. Здесь по WCAG 2.1 выбирается тёмная или светлая «краска» с наибольшим
// контрастом — тогда заголовок и подпись читаются на любом градиенте.

export type Rgb = [number, number, number];

export interface HeroInk {
  isLight: boolean;
  title: string;
  sub: string;
  link: string;
}

const INK_DARK = '#1b1220';
const INK_LIGHT = '#ffffff';
const SUB_DARK = 'rgba(27, 18, 32, .78)';
const SUB_LIGHT = 'rgba(255, 255, 255, .92)';
const MIN_RATIO = 4.5;

function linearize(channel: number): number {
  const value = channel / 255;
  return value <= 0.03928 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
}

export function luminance(color: Rgb): number {
  const [r, g, b] = color.map(linearize);
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

export function contrastRatio(first: Rgb, second: Rgb): number {
  const [high, low] = [luminance(first), luminance(second)].sort((a, b) => b - a);
  return (high + 0.05) / (low + 0.05);
}

export function parseColors(css: string): Rgb[] {
  const colors: Rgb[] = [];
  const hexPattern = /#([0-9a-f]{3,8})\b/gi;
  const rgbPattern = /rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)/gi;
  for (const match of css.matchAll(hexPattern)) {
    let hex = match[1];
    if (hex.length === 3 || hex.length === 4) hex = hex.slice(0, 3).split('').map((c) => c + c).join('');
    if (hex.length >= 6) {
      colors.push([
        Number.parseInt(hex.slice(0, 2), 16),
        Number.parseInt(hex.slice(2, 4), 16),
        Number.parseInt(hex.slice(4, 6), 16),
      ]);
    }
  }
  for (const match of css.matchAll(rgbPattern)) {
    colors.push([Number(match[1]), Number(match[2]), Number(match[3])]);
  }
  return colors;
}

// Средний цвет градиента: для линейного перехода это разумная оценка фона под текстом.
export function averageColor(css: string): Rgb | null {
  const colors = parseColors(css);
  if (colors.length === 0) return null;
  const sum = colors.reduce<[number, number, number]>(
    (acc, color) => [acc[0] + color[0], acc[1] + color[1], acc[2] + color[2]],
    [0, 0, 0],
  );
  return [Math.round(sum[0] / colors.length), Math.round(sum[1] / colors.length), Math.round(sum[2] / colors.length)];
}

// Краска для текста поверх градиента: выбираем вариант с большим контрастом по WCAG.
export function heroInk(background: string): HeroInk {
  const average = averageColor(background || '');
  if (!average) return inkFor(INK_LIGHT);
  const darkRatio = contrastRatio(average, hexToRgb(INK_DARK));
  const lightRatio = contrastRatio(average, hexToRgb(INK_LIGHT));
  const ink = darkRatio >= lightRatio && darkRatio >= MIN_RATIO ? INK_DARK : INK_LIGHT;
  return inkFor(ink);
}

export function hexToRgb(hex: string): Rgb {
  const parsed = parseColors(hex);
  return parsed[0] || [0, 0, 0];
}

function inkFor(ink: string): HeroInk {
  const isLight = ink === INK_DARK;
  return {
    isLight,
    title: ink,
    sub: isLight ? SUB_DARK : SUB_LIGHT,
    link: ink,
  };
}
