<script setup lang="ts">
// Заглушка экрана «Каталог» (наполнение — C-039: браузер каталога с фильтрами).
useHead({ title: 'Каталог — Своё Вино' });
</script>

<template>
  <div class="page">
    <div class="hero">
      <h1>Каталог вин</h1>
      <p>2103 позиции с официальными метаданными портала: регион, винодельня, сорт, категория и изображение.</p>
    </div>

    <section class="card">
      <h2>Что будет доступно</h2>
      <ul class="chips">
        <li class="chip">Поиск по названию и винодельне</li>
        <li class="chip">Фильтры: регион, категория, сорт</li>
        <li class="chip">Пагинация и сортировка</li>
      </ul>
      <p class="muted small" style="margin-top: 12px">
        Данные отдаёт gateway: <code>GET /v1/catalog</code>. UI не ходит в API портала напрямую.
      </p>
    </section>

    <section class="card placeholder">
      Скоро: полноценный браузер каталога с карточками и переходом к вину.
    </section>
  </div>
</template>
