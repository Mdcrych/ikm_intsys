import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # без GUI — для серверов и тестов
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# =====================================================================
# БЛОК 0: КОНСТАНТЫ И НАСТРОЙКИ
# =====================================================================

CSV_PATH = os.environ.get('CSV_PATH', 'insurance.csv')
MODEL_OUTPUT_PATH = os.environ.get('MODEL_OUTPUT_PATH', 'best_insurance_model.pkl')
PLOT_OUTPUT_PATH = os.environ.get('PLOT_OUTPUT_PATH', 'error_plot.png')

REQUIRED_COLUMNS = {'age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges'}
SEX_MAP = {'female': 0, 'male': 1}
SMOKER_MAP = {'no': 0, 'yes': 1}
REGION_MAP = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}


# =====================================================================
# БЛОК 1: ПОДГОТОВКА ДАННЫХ
# =====================================================================
print("--- НАЧАЛО: Подготовка данных ---")

# 1. Загрузка CSV
if not os.path.isfile(CSV_PATH):
    print(
        f"[ОШИБКА] Файл '{CSV_PATH}' не найден. "
        "Скачайте датасет и положите его рядом со скриптом (или задайте CSV_PATH)." 
    )
    sys.exit(1)

try:
    df = pd.read_csv(CSV_PATH)
except Exception as e:
    print(f"[ОШИБКА] Не удалось прочитать CSV: {e}")
    sys.exit(1)

# 2. Проверка обязательных колонок
missing_cols = REQUIRED_COLUMNS - set(df.columns.str.lower())
if missing_cols:
    print(f"[ОШИБКА] В файле отсутствуют колонки: {missing_cols}")
    sys.exit(1)

# Приведём названия колонок к нижнему регистру для единообразия
df.columns = df.columns.str.lower().str.strip()

# 3. Проверка на пропущенные значения
if df.isnull().any().any():
    null_report = df.isnull().sum()
    null_report = null_report[null_report > 0]
    print(f"[ПРЕДУПРЕЖДЕНИЕ] Обнаружены пропущенные значения:\n{null_report}")
    print("Строки с пропусками будут удалены.")
    df = df.dropna()
    print(f"Осталось строк после очистки: {len(df)}")

if len(df) < 50:
    print("[ОШИБКА] После очистки данных осталось менее 50 строк — обучение невозможно.")
    sys.exit(1)

# 4. Нормализуем строковые значения категориальных признаков
for col in ('sex', 'smoker', 'region'):
    df[col] = df[col].astype(str).str.lower().str.strip()

# 5. Валидация допустимых значений категорий
invalid_sex = set(df['sex'].unique()) - set(SEX_MAP)
invalid_smoker = set(df['smoker'].unique()) - set(SMOKER_MAP)
invalid_region = set(df['region'].unique()) - set(REGION_MAP)
if invalid_sex:
    print(f"[ОШИБКА] Неизвестные значения 'sex': {invalid_sex}")
    sys.exit(1)
if invalid_smoker:
    print(f"[ОШИБКА] Неизвестные значения 'smoker': {invalid_smoker}")
    sys.exit(1)
if invalid_region:
    print(f"[ОШИБКА] Неизвестные значения 'region': {invalid_region}")
    sys.exit(1)

# 6. Валидация числовых диапазонов
assert df['age'].between(0, 150).all(), "Обнаружены некорректные значения возраста"
assert df['bmi'].between(5, 100).all(), "Обнаружены некорректные значения ИМТ"
assert df['children'].between(0, 20).all(), "Обнаружены некорректные значения кол-ва детей"
assert (df['charges'] >= 0).all(), "Обнаружены отрицательные значения стоимости страховки"

# 7. Кодирование категориальных признаков
df['sex'] = df['sex'].map(SEX_MAP)
df['smoker'] = df['smoker'].map(SMOKER_MAP)
df['region'] = df['region'].map(REGION_MAP)

# 8. Масштабирование числовых признаков (Standardization)
numeric_cols = ['age', 'bmi', 'children']
scaling_params = {}

for col in numeric_cols:
    mean_val = df[col].mean()
    std_val = df[col].std()
    if std_val == 0:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Стандартное отклонение для '{col}' равно 0. Масштабирование пропущено.")
        scaling_params[col] = {'mean': mean_val, 'std': 1.0}
    else:
        scaling_params[col] = {'mean': mean_val, 'std': std_val}
        df[col] = (df[col] - mean_val) / std_val

X = df.drop('charges', axis=1)
y = df['charges']

# 9. Разделение данных: 85% main + 15% holdout test
X_main, X_test, y_main, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)
X_main = X_main.reset_index(drop=True)
y_main = y_main.reset_index(drop=True)

print(f"Данные успешно подготовлены. Всего строк: {len(df)}, обучающих: {len(X_main)}, тестовых: {len(X_test)}\n")

# =====================================================================
# БЛОК 2: ОБУЧЕНИЕ И ДИАГНОСТИКА
# =====================================================================
print("--- НАЧАЛО: Обучение и диагностика ---")


class SimpleSmokerRule:
    """Базовая модель-бейзлайн: предсказывает среднюю стоимость по признаку курения."""

    def __init__(self):
        self.cost_smoker = None
        self.cost_nonsmoker = None

    def fit(self, X_train, y_train):
        smoker_mask = X_train['smoker'] == 1
        if smoker_mask.sum() == 0 or (~smoker_mask).sum() == 0:
            raise ValueError(
                "В обучающей выборке отсутствуют представители одной из групп (курящие/некурящие)."
            )
        self.cost_smoker = float(y_train[smoker_mask].mean())
        self.cost_nonsmoker = float(y_train[~smoker_mask].mean())
        return self

    def predict(self, X_val):
        if self.cost_smoker is None:
            raise RuntimeError("Модель не обучена. Вызовите .fit() перед .predict().")
        return np.where(X_val['smoker'] == 1, self.cost_smoker, self.cost_nonsmoker)


rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
simple_maes, rf_maes = [], []

print("Запуск кросс-валидации (5 разбиений)...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X_main)):
    X_tr, X_v = X_main.iloc[train_idx], X_main.iloc[val_idx]
    y_tr, y_v = y_main.iloc[train_idx], y_main.iloc[val_idx]

    simple_model = SimpleSmokerRule()
    simple_model.fit(X_tr, y_tr)
    simple_maes.append(mean_absolute_error(y_v, simple_model.predict(X_v)))

    rf_model.fit(X_tr, y_tr)
    rf_maes.append(mean_absolute_error(y_v, rf_model.predict(X_v)))

avg_simple_mae = float(np.mean(simple_maes))
avg_rf_mae = float(np.mean(rf_maes))

print(f"Средняя ошибка (MAE) Простого правила: ${avg_simple_mae:.2f}")
print(f"Средняя ошибка (MAE) Случайного леса:  ${avg_rf_mae:.2f}\n")

# Диагностика: обучим на всех main-данных, проверим ошибки
rf_model.fit(X_main, y_main)
main_preds = rf_model.predict(X_main)
errors = np.abs(y_main - main_preds)

error_df = X_main.copy()
error_df['True_Charges'] = y_main.values
error_df['Predicted'] = main_preds
error_df['Error'] = errors.values
worst_predictions = error_df.sort_values('Error', ascending=False).head(20)

print("ТАБЛИЦА: Топ-5 случаев с самой большой ошибкой модели:")
print(worst_predictions[['smoker', 'bmi', 'True_Charges', 'Predicted', 'Error']].head(5).to_string(), "\n")

# График
try:
    plt.figure(figsize=(10, 6))
    plt.scatter(y_main, main_preds, alpha=0.5, color='blue', label='Обычные предсказания')
    plt.scatter(
        worst_predictions['True_Charges'],
        worst_predictions['Predicted'],
        color='red', label='Топ-20 худших ошибок'
    )
    max_val = max(y_main.max(), main_preds.max())
    plt.plot([0, max_val], [0, max_val], 'k--', lw=2, label='Идеальный прогноз')
    plt.title('Настоящая стоимость vs Предсказанная стоимость')
    plt.xlabel('Реальная стоимость ($)')
    plt.ylabel('Предсказанная стоимость ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT_OUTPUT_PATH)
    print(f"График ошибок сохранён в '{PLOT_OUTPUT_PATH}'.\n")
except Exception as e:
    print(f"[ПРЕДУПРЕЖДЕНИЕ] Не удалось сохранить график: {e}\n")
finally:
    plt.close()

# =====================================================================
# БЛОК 3: ФИНАЛЬНЫЙ ОТБОР И СОХРАНЕНИЕ
# =====================================================================
print("--- НАЧАЛО: Финальный отбор и сохранение ---")

best_model = rf_model
final_predictions = best_model.predict(X_test)
final_mae = float(mean_absolute_error(y_test, final_predictions))

model_data = {
    'model': best_model,
    'scaling_params': scaling_params,
    'metrics': {
        'cv_simple_mae': avg_simple_mae,
        'cv_rf_mae': avg_rf_mae,
        'final_test_mae': final_mae,
    },
    'feature_order': list(X_main.columns),
}

try:
    output_dir = os.path.dirname(MODEL_OUTPUT_PATH)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(MODEL_OUTPUT_PATH, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Модель успешно сохранена в '{MODEL_OUTPUT_PATH}'!\n")
except Exception as e:
    print(f"[ОШИБКА] Не удалось сохранить модель: {e}")
    sys.exit(1)

worst_smokers_pct = worst_predictions['smoker'].mean() * 100

print("=" * 50)
print("ФИНАЛЬНЫЙ ОТЧЕТ:")
print(f"Лучшая модель — Random Forest Regressor.")
print(f"MAE на кросс-валидации:          ${avg_rf_mae:.2f}")
print(f"MAE на отложенных (holdout) данных: ${final_mae:.2f}")
print(f"Из топ-20 худших ошибок {worst_smokers_pct:.0f}% пациентов — курящие люди.")
print("=" * 50)
