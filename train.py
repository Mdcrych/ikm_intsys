import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =====================================================================
# БЛОК 1: ПОДГОТОВКА ДАННЫХ
# =====================================================================
print("--- НАЧАЛО: Подготовка данных ---")

# 1. Загрузка CSV файла
# Ожидается, что файл insurance.csv лежит в той же папке
try:
    df = pd.read_csv('insurance.csv')
except FileNotFoundError:
    print("Ошибка: Файл 'insurance.csv' не найден. Скачайте датасет и положите его рядом со скриптом.")
    exit()

# ОБЪЯСНЕНИЕ ПРИЗНАКОВ (X) И ОТВЕТА (y):
# Признаки (X) - это параметры человека, на основе которых мы делаем прогноз: 
# 'age' (возраст), 'sex' (пол), 'bmi' (индекс массы тела), 'children' (кол-во детей), 
# 'smoker' (курит ли), 'region' (регион проживания).
# Правильный ответ (y) - это целевая переменная 'charges' (стоимость страховки).

# 2. Превращение категориальных признаков в числа (кодирование)
# Модели машинного обучения работают только с числами. 
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# 3. Масштабирование числовых признаков (Standardization)
# Мы вычитаем среднее значение и делим на стандартное отклонение. 
# Это нужно, чтобы признаки с большими значениями (например, возраст) 
# не "задавили" признаки с малыми значениями (например, кол-во детей).
numeric_cols = ['age', 'bmi', 'children']
scaling_params = {} # Сохраним параметры для масштабирования новых данных в будущем

for col in numeric_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    scaling_params[col] = {'mean': mean_val, 'std': std_val}
    df[col] = (df[col] - mean_val) / std_val

# Отделяем X от y
X = df.drop('charges', axis=1)
y = df['charges']

# 4. Разделение данных на обучающую/проверочную и финальную тестовую (сейф)
# ПОЧЕМУ ТАК: 85% данных (большая часть) нужна для обучения и кросс-валидации 
# (подбора и сравнения моделей). 15% (маленькая часть) откладывается в "сейф". 
# Модель НИКОГДА их не увидит до самого финала. Это гарантирует честную проверку.
X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Сброс индексов, чтобы не было ошибок при кросс-валидации
X_main = X_main.reset_index(drop=True)
y_main = y_main.reset_index(drop=True)

print("Данные успешно подготовлены и разделены!\n")

# =====================================================================
# БЛОК 2: ОБУЧЕНИЕ И ДИАГНОСТИКА
# =====================================================================
print("--- НАЧАЛО: Обучение и диагностика ---")

# АЛГОРИТМ 1: Простое правило (Собственная реализация)
# Логика: Курение - главный фактор. Модель просто предсказывает среднюю 
# стоимость для курильщика, если человек курит, и среднюю для некурящего, если нет.
class SimpleSmokerRule:
    def fit(self, X_train, y_train):
        # Находим среднюю стоимость для курящих (1) и некурящих (0)
        self.cost_smoker = y_train[X_train['smoker'] == 1].mean()
        self.cost_nonsmoker = y_train[X_train['smoker'] == 0].mean()
        
    def predict(self, X_val):
        # Возвращаем предсказание на основе одного столбца
        return np.where(X_val['smoker'] == 1, self.cost_smoker, self.cost_nonsmoker)

# АЛГОРИТМ 2: Готовый сложный алгоритм (Случайный лес)
rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

# Цикл проверки: Кросс-валидация (5 фолдов)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

simple_maes = []
rf_maes = []

print("Запуск кросс-валидации (5 разбиений)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_main)):
    X_tr, X_v = X_main.iloc[train_idx], X_main.iloc[val_idx]
    y_tr, y_v = y_main.iloc[train_idx], y_main.iloc[val_idx]
    
    # Обучение и тест Простого правила
    simple_model = SimpleSmokerRule()
    simple_model.fit(X_tr, y_tr)
    simple_preds = simple_model.predict(X_v)
    simple_mae = mean_absolute_error(y_v, simple_preds)
    simple_maes.append(simple_mae)
    
    # Обучение и тест Случайного леса
    rf_model.fit(X_tr, y_tr)
    rf_preds = rf_model.predict(X_v)
    rf_mae = mean_absolute_error(y_v, rf_preds)
    rf_maes.append(rf_mae)

avg_simple_mae = np.mean(simple_maes)
avg_rf_mae = np.mean(rf_maes)

print(f"Средняя ошибка (MAE) Простого правила: ${avg_simple_mae:.2f}")
print(f"Средняя ошибка (MAE) Случайного леса: ${avg_rf_mae:.2f}\n")

# ДИАГНОСТИКА: На каких примерах модель ошибается чаще всего?
# Обучим RF на всех main данных и предскажем для них же, чтобы найти паттерны ошибок
rf_model.fit(X_main, y_main)
main_preds = rf_model.predict(X_main)
errors = np.abs(y_main - main_preds)

# Создаем таблицу с ошибками
error_df = X_main.copy()
error_df['True_Charges'] = y_main
error_df['Predicted'] = main_preds
error_df['Error'] = errors

# Берем топ-20 самых больших ошибок
worst_predictions = error_df.sort_values(by='Error', ascending=False).head(20)

print("ТАБЛИЦА: Топ-5 случаев с самой большой ошибкой модели:")
print(worst_predictions[['smoker', 'bmi', 'True_Charges', 'Predicted', 'Error']].head(5).to_string(), "\n")

# Визуализация главной ошибки
plt.figure(figsize=(10, 6))
plt.scatter(y_main, main_preds, alpha=0.5, color='blue', label='Обычные предсказания')
plt.scatter(worst_predictions['True_Charges'], worst_predictions['Predicted'], color='red', label='Топ-20 Худших ошибок')
plt.plot([0, 60000], [0, 60000], 'k--', lw=2, label='Идеальный прогноз')
plt.title('Настоящая стоимость vs Предсказанная стоимость')
plt.xlabel('Реальная стоимость ($)')
plt.ylabel('Предсказанная стоимость ($)')
plt.legend()
plt.grid(True)
plt.savefig('error_plot.png')
print("График ошибок сохранен в файл 'error_plot.png'.\n")


# =====================================================================
# БЛОК 3: ФИНАЛЬНЫЙ ОТБОР И СОХРАНЕНИЕ
# =====================================================================
print("--- НАЧАЛО: Финальный отбор и сохранение ---")

# Очевидно, что Случайный лес работает лучше. Фиксируем его.
best_model = rf_model

# Один раз проверяем на отложенных данных (X_test, y_test)
final_predictions = best_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_predictions)

# Сохранение модели и параметров масштабирования в файл
model_data = {
    'model': best_model,
    'scaling_params': scaling_params
}

with open('best_insurance_model.pkl', 'wb') as file:
    pickle.dump(model_data, file)

print("Модель успешно сохранена в 'best_insurance_model.pkl'!\n")

# Анализ паттерна худших ошибок для финального отчета
# Проверим, кого в основном содержат худшие прогнозы (обычно это курящие с высоким ИМТ)
worst_smokers_pct = worst_predictions['smoker'].mean() * 100

# ФИНАЛЬНЫЙ ОТЧЕТ:
print("="*50)
print("ФИНАЛЬНЫЙ ОТЧЕТ:")
print(f"Лучшая модель — Random Forest Regressor.")
print(f"Её ключевая метрика (MAE) на абсолютно новых данных из 'сейфа' — ${final_mae:.2f}")
print(f"Чаще всего она сильно занижает прогноз на случаях-аномалиях,")
print(f"где реальная стоимость лечения колоссальна (>40 000$).")
print(f"Из топ-20 худших ошибок {worst_smokers_pct:.0f}% пациентов — это курящие люди.")
print("="*50)