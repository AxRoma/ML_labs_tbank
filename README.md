# Machine Learning Homeworks

Репозиторий с домашними заданиями по курсу Машинного Обучения. Каждая тетрадка посвящена отдельной теме: от базовых алгоритмов до интерпретации моделей.

## Содержание

### 1. K-Nearest Neighbors (`knn_homework.ipynb`)
Реализация алгоритма KNN для задач классификации и регрессии.
- **Данные:** Iris, Wine (классификация), Diabetes (регрессия)
- **Что сделано:** Подбор оптимального числа соседей (k), оценка качества (Accuracy, Precision/Recall, R², MAE/RMSE)

### 2. Линейные модели (`domashnee_zadanie_logisticheskaia...`)
Изучение логистической и линейной регрессии, а также методов регуляризации.
- **Что сделано:** Сравнение работы моделей с L1 (Lasso) и L2 (Ridge) регуляризацией. Анализ влияния коэффициентов на переобучение.

### 3. Деревья и Ансамбли (`dz_3_derev-ia_i_ansambli.ipynb`)
Задача обнаружения мошенничества (Fraud Detection) на несбалансированной выборке кредитных карт.
- **Модели:** Decision Tree, Logistic Regression, KNN
- **Ансамбли:** Bagging, RandomForest, Stacking
- **Результат:** Сравнение метрик (F1, ROC-AUC) и матриц ошибок для всех моделей

### 4. Продвинутая классификация (`domashnee_zadanie_4.ipynb`)
Предсказание оформления депозита (Bank Marketing Dataset).
- **Модели:** Decision Tree (с визуализацией графа), RandomForest, CatBoost
- **Эксперименты:** Сравнение методов кодирования категориальных признаков (Target Encoding vs One-Hot Encoding) внутри CatBoost

### 5. Предобработка данных (`dz_predobrabotka_dannykh.ipynb`)
Пайплайн подготовки "сырых" данных к обучению.
- **Методы:** Масштабирование признаков (StandardScaler), кодирование категорий (LabelEncoder), разделение выборки со стратификацией

### 6. Важность признаков (`dz_ml_feature_importance.ipynb`)
Интерпретация моделей машинного обучения на данных о кредитном дефолте.
- **Модель:** CatBoostClassifier
- **Методы оценки:**
  1. Встроенный `feature_importances_` (CatBoost)
  2. Permutation Importance (перестановочная важность)
  3. SHAP values (Summary plot, Force plot)

---

## Технический стек
- **Библиотеки:** `scikit-learn`, `catboost`, `shap`, `pandas`, `matplotlib`, `seaborn`
- **Среда:** Jupyter Notebook

## Запуск
```bash
pip install -r requirements.txt
jupyter notebook
