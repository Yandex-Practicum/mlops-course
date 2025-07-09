# 🎵 Music Recommender API

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Микросервис для рекомендации музыкальных треков на основе FastAPI. Генерирует 5 случайных рекомендаций по названию трека.

## ⚡ Быстрый старт

```bash
git clone https://github.com/Yandex-Practicum/mlops-freetrack.git
cd mlops-freetrack
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# Убедитесь, что в корне проекта есть файл main.py с логикой API,
# затем запустите приложение:
uvicorn main:app --reload
```
Приложение будет доступно по адресу: http://127.0.0.1:8000

## 📚 API Reference

### Доступные эндпоинты

| Метод | Endpoint                  | Описание                     |
|-------|---------------------------|-----------------------------|
| GET   | `/`                       | Проверка работоспособности  |
| GET   | `/api/recommend/{track}`  | Получить рекомендации       |

### Пример запроса

```bash
curl -X GET "http://localhost:8000/api/recommend/Bohemian%20Rhapsody"
```

### Пример ответа

```json
{
 "requested_track": "Bohemian Rhapsody",
  "recommendations": \[
    {
      "track_name": "We Will Rock You",
      "artist": "Queen",
      "album": "News of the World"
    }
  ]
}
```

## 🖼️ Документация API

После запуска доступны:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

![Swagger Interface](media/swagger_preview.png)

## 🛠 Технологии

- Python 3.10
- FastAPI
- Uvicorn
- Pydantic
- Swagger UI 

# 🤝 Как внести вклад

Мы приветствуем вклад в развитие проекта! Чтобы предложить свои улучшения:

1. **Форкните репозиторий**  
   Нажмите кнопку "Fork" в правом верхнем углу страницы репозитория

2. **Создайте feature-ветку**  
   ```bash
   git checkout -b feature/название-вашей-фичи
   ```
   Пример:
   ```bash
   git checkout -b feature/add-model-monitoring
   ```
3. **Сделайте коммиты**
   ```bash
   git commit -m "Добавил мониторинг качества модели"
   
   Рекомендации:
   Один коммит = одно логическое изменение
   Сообщения коммитов на английском
4. **Запушьте изменения**
   ```bash
   git push origin feature/название-вашей-фичи
   ```
5. **Откройте Pull Request**
   - Перейдите в оригинальный репозиторий
   - Нажмите "New Pull Request"
   - Опишите предлагаемые изменения

   Перед отправкой PR убедитесь, что:
   ✅ Код проходит все тесты (pytest)
   ✅ Соответствует PEP8 (flake8)

## 🎓 Разработано в Яндекс.Практикуме командой MLOps

## 📜 Лицензия

Этот проект распространяется под лицензией MIT. Подробнее см. в [LICENSE](LICENSE).
