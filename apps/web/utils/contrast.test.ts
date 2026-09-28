// Проверка читаемости: для реальных градиентов категорий портала выбранная краска
// обязана проходить порог WCAG AA (4.5) на среднем цвете фона. Тест гоняется bun test.
import { describe, expect, test } from 'bun:test';
import { averageColor, contrastRatio, heroInk, hexToRgb, parseColors } from './contrast';

const REAL_GRADIENTS = [
  'linear-gradient(180deg, #FCFBDF 0%, #F0EFD5 100%)',
  'linear-gradient(180deg, #A51B3A 0%, #741D30 100%)',
  'linear-gradient(180deg, #F2AAAD 0%, #F2828F 100%)',
  'linear-gradient(180deg, #FDBF98 0%, #ED7C35 100%)',
];

describe('contrast', () => {
  test('разбирает hex- и rgb-цвета из градиента', () => {
    expect(parseColors('linear-gradient(180deg, #FCFBDF 0%, rgb(240, 239, 213) 100%)').length).toBe(2);
  });

  test('краска проходит WCAG AA на каждом реальном градиенте', () => {
    for (const background of REAL_GRADIENTS) {
      const ink = heroInk(background);
      const average = averageColor(background)!;
      const ratio = contrastRatio(average, hexToRgb(ink.title));
      expect(ratio).toBeGreaterThanOrEqual(4.5);
    }
  });

  test('светлый градиент получает тёмный текст, тёмный — белый', () => {
    expect(heroInk('linear-gradient(180deg, #FCFBDF 0%, #F0EFD5 100%)').isLight).toBe(true);
    expect(heroInk('linear-gradient(180deg, #A51B3A 0%, #741D30 100%)').isLight).toBe(false);
  });

  test('пустой фон деградирует к белому тексту', () => {
    expect(heroInk('').title).toBe('#ffffff');
  });

  test('белый текст на светло-жёлтом фоне не проходит порог', () => {
    const background = REAL_GRADIENTS[0];
    const average = averageColor(background)!;
    expect(contrastRatio(average, [255, 255, 255])).toBeLessThan(4.5);
  });
});
