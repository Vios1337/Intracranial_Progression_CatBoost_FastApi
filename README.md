# 🧠 Прогноз интракраниальной прогрессии

**ML-модель для прогноза интракраниальной прогрессии у пациентов с метастазами в головной мозг.**

![CatBoost](https://img.shields.io/badge/ML-CatBoost-ffcc00)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Python](https://img.shields.io/badge/Python-3.13-3776ab)

## 🎯 Постановка задачи

По клиническим данным пациента определить наличие интракраниальной прогрессии:

- **0** — нет прогрессии;
- **1** — локальный рецидив, дистантные метастазы или их сочетание.

Модель — ансамбль из пяти CatBoostClassifier. Итоговый прогноз строится по среднему значению вероятностей. Для работы доступны веб-форма, REST API и пакетная обработка CSV.

### Результаты

Результаты OOF-оценки в [исследовательском ноутбуке](notebooks/score_0_84.ipynb), 680 размеченных наблюдений:

| Показатель | Значение |
|---|---:|
| **Balanced Accuracy** | **0.7704** |
| Порог классификации | 0.50 |

Порог и число деревьев выбраны по валидационным прогнозам; независимая тестовая оценка не проводилась.

## 🏗️ Структура проекта

```text
intracranial_progression/
├── data/raw/                  # train.csv и test.csv
├── notebooks/
│   └── score_0_84.ipynb        # исследование и обучение моделей
├── src/
│   ├── preprocessing.py       # обработка данных и построение признаков
│   ├── model.py               # параметры, сохранение и загрузка ансамбля
│   └── metrics.py             # метрики и графики
├── models/                    # модели CatBoost и metadata.json
├── reports/                   # метрики, графики и прогнозы
├── api/
│   ├── main.py                # FastAPI
│   ├── models.py              # схемы запросов и ответов
│   ├── model_service.py       # прогноз для одного пациента
│   ├── templates/index.html   # веб-форма
│   └── static/                # CSS и JavaScript
├── examples/                  # примеры JSON и CSV
├── tests/
├── train.py                   # обучение ансамбля
├── predict.py                 # прогноз для CSV
├── requirements.txt
└── requirements-dev.txt
```

## 🚀 Быстрый старт

Python 3.13. Команды для PowerShell:

```powershell
cd intracranial_progression
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Разместите `train.csv` и `test.csv` в `data/raw/`.

**Обучение:**

```powershell
.venv\Scripts\python.exe train.py
```

Пять моделей и параметры обработки сохраняются в `models/`. Метрики, OOF-прогнозы, ROC-кривая, матрица ошибок и важность признаков — в `reports/`.

**Прогноз для CSV:**

```powershell
.venv\Scripts\python.exe predict.py --input data/raw/test.csv --output reports/predictions.csv
```

Результат: `ID`, `Интракраниальная прогрессия` и `Вероятность прогрессии`.

**Запуск сервиса:**

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- [Веб-форма](http://localhost:8000)
- [API Docs](http://localhost:8000/docs)

Сервис использует модели из `models/`. После переобучения перезапустите сервер. Для сохранения нового запуска в отдельные папки у `train.py` доступны `--model-dir` и `--report-dir`.

В Linux/macOS путь к интерпретатору — `.venv/bin/python`.

## 📊 Данные

Формат CSV: UTF-8, разделитель — запятая. Названия входных столбцов приведены в [examples/patients.csv](examples/patients.csv). `ID` необязателен и в модель не входит.

Для обучения дополнительно нужны столбцы `Интракраниальная прогрессия`, `Локальный рецидив` и `Дистантные метастазы`. Строки без исходной метки прогрессии исключаются, затем бинарный таргет формируется из локального рецидива и дистантных метастазов.

Ручные исправления в `src/preprocessing.py` привязаны к индексам исходного `train.csv`: порядок строк перед обучением должен сохраняться.

## 🔬 Pipeline

```text
1. Очистка дат и числовых значений
2. Feature engineering: возраст, временные интервалы, признаки лечения
3. Обработка пропусков и исключение столбцов с исходами из признаков
4. Обучение пяти CatBoost на StratifiedKFold, random_state=42
5. Подбор порога по OOF Balanced Accuracy
6. Усреднение вероятностей пяти моделей для прогноза
```

Параметры CatBoost заданы в `src/model.py`: `depth=4`, `learning_rate=0.07986`, `iterations=2000`, `auto_class_weights="Balanced"`. Отчёт об обучении сохраняется в `reports/metrics.json`.

## 🌐 API

| Endpoint | Метод | Описание |
|---|---|---|
| `/` | GET | Веб-форма |
| `/predict` | POST | Прогноз для одного пациента |
| `/docs` | GET | Swagger UI |

**Пример запроса:**

```powershell
curl.exe -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" --data-binary "@examples/patient.json"
```

```json
{
  "gender": "Ж",
  "birth_date": "15.06.1965",
  "diagnosis": "НМРЛ",
  "diagnosis_date": "01.02.2019",
  "primary_removal_date": "не удален",
  "brain_metastases_date": "01.06.2020",
  "activating_mutations": "нет",
  "wbrt_date": "нет",
  "brain_surgery_date": "нет",
  "radiosurgery_count": 1,
  "first_radiosurgery_date": "01.07.2020",
  "karnofsky_index": 80,
  "lesion_count": 2,
  "total_lesion_volume": 3.5,
  "extracranial_metastases": "нет",
  "drug_treatment": "Без лечения"
}
```

Даты — `ДД.ММ.ГГГГ`, неизвестные необязательные значения — `null`. API также принимает русские названия столбцов. Полная схема доступна в `/docs`.

**Поля ответа:**

| Поле | Описание |
|---|---|
| `probability` | Оценка вероятности прогрессии |
| `prediction` | Класс: `0` или `1` |
| `threshold` | Порог классификации |
| `status` | Текст прогноза |

Класс `1` присваивается при `probability > threshold`. Ошибки ввода возвращают `422`, отсутствие обученных моделей — `503`.

## ✅ Тесты

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
```

## 🛠️ Стек технологий

```text
ML:        CatBoost, Scikit-learn
Данные:    Pandas, NumPy, Matplotlib
API:       FastAPI, Pydantic, Uvicorn
Интерфейс: HTML, CSS, JavaScript
Тесты:     Pytest, HTTPX
```
