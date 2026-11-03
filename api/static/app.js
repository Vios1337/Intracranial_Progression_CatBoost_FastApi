const form = document.querySelector('#prediction-form');
const result = document.querySelector('#result');
const button = document.querySelector('#submit-button');
const treatmentFields = ['primary_removal', 'wbrt', 'brain_surgery'];

function formatDate(value) {
  return value ? value.split('-').reverse().join('.') : null;
}

function buildPayload(formData) {
  const payload = {};
  const numericFields = ['radiosurgery_count', 'karnofsky_index', 'lesion_count', 'total_lesion_volume'];
  for (const [key, value] of formData.entries()) {
    if (key.endsWith('_state')) continue;
    if (key.endsWith('_date')) payload[key] = formatDate(value);
    else if (numericFields.includes(key)) payload[key] = Number(value);
    else payload[key] = value || null;
  }
  for (const name of treatmentFields) {
    const state = formData.get(`${name}_state`);
    if (state !== 'date') payload[`${name}_date`] = state || null;
  }
  return payload;
}

for (const name of treatmentFields) {
  const state = form.elements[`${name}_state`];
  const date = form.elements[`${name}_date`];
  state.addEventListener('change', () => {
    date.hidden = state.value !== 'date';
    date.required = state.value === 'date';
    if (state.value !== 'date') date.value = '';
  });
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  button.disabled = true;
  result.hidden = false;
  result.className = '';
  result.textContent = 'Выполняется расчёт…';
  try {
    const response = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildPayload(new FormData(form))),
    });
    if (!response.ok) {
      let message = `Ошибка сервера (${response.status}).`;
      if ((response.headers.get('content-type') || '').includes('application/json')) {
        const error = await response.json();
        message = Array.isArray(error.detail)
          ? error.detail.map(item => `${item.loc.slice(1).join('.')}: ${item.msg}`).join('\n')
          : error.detail || message;
      }
      throw new Error(message);
    }
    const prediction = await response.json();
    result.textContent = `${prediction.status}\nОценка модели: ${(prediction.probability * 100).toFixed(1)} %\nПорог классификации: ${prediction.threshold.toFixed(2)}`;
  } catch (error) {
    result.className = 'error';
    result.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});
