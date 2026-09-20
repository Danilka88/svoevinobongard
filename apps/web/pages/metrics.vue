<script setup lang="ts">
// Заглушка экрана «Метрики» (наполнение — C-039: витрина /v1/demo/metrics).
useHead({ title: 'Метрики — Своё Вино' });
</script>

<template>
  <div class="page">
    <div class="hero">
      <h1>Метрики качества</h1>
      <p>Честные числа проекта: синтетический прокси, реальные полевые фото и границы применимости.</p>
    </div>

    <section class="card">
      <h2>Что покажем</h2>
      <ul class="chips">
        <li class="chip">acc@1 и acc@5 на прокси</li>
        <li class="chip">identity@1 и @5 на реальных фото</li>
        <li class="chip">Latency p50 и p95 против SLA</li>
        <li class="chip">Состояние гейта сдачи</li>
      </ul>
      <p class="muted small" style="margin-top: 12px">
        Источник — <code>GET /v1/demo/metrics</code>. Числа не подгоняются: если метод не помог, это видно.
      </p>
    </section>

    <section class="card placeholder">
      Скоро: дашборд с графиками и оговорками о домене измерений.
    </section>
  </div>
</template>
