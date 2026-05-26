import gradio as gr
import pandas as pd
import pickle
import os

# =====================================================================
# ЭТАП 1: Загрузка модели (с нормальной обработкой ошибок)
# =====================================================================

MODEL_PATH = os.environ.get('MODEL_PATH', 'ikm/best_insurance_model.pkl')

def load_model(path: str):
    """Загружает модель из pkl-файла. Возвращает (model, scaling_params) или (None, None)."""
    if not os.path.isfile(path):
        return None, None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = data.get('model')
        scaling_params = data.get('scaling_params')
        if model is None or scaling_params is None:
            raise ValueError("pkl-файл не содержит ключей 'model' или 'scaling_params'")
        return model, scaling_params
    except Exception as e:
        print(f"[ОШИБКА] Не удалось загрузить модель: {e}")
        return None, None

model, scaling_params = load_model(MODEL_PATH)

if model is not None:
    print("Модель успешно загружена!")
else:
    print(f"[ПРЕДУПРЕЖДЕНИЕ] Файл '{MODEL_PATH}' не найден или повреждён. "
          "Сначала запустите train.py. Приложение запустится, но предсказания будут недоступны.")

# =====================================================================
# ЭТАП 2: Функция предсказания (с валидацией)
# =====================================================================

SEX_MAPPING = {'Женщина (Female)': 0, 'Мужчина (Male)': 1}
SMOKER_MAPPING = {'Нет (No)': 0, 'Да (Yes)': 1}
REGION_MAPPING = {
    'Юго-Запад (Southwest)': 0,
    'Юго-Восток (Southeast)': 1,
    'Северо-Запад (Northwest)': 2,
    'Северо-Восток (Northeast)': 3,
}


def _validate_inputs(age, sex, bmi, children, smoker, region):
    """Проверяет входные данные. Возвращает список ошибок (пустой — если всё ок)."""
    errors = []
    if not isinstance(age, (int, float)) or not (18 <= age <= 100):
        errors.append("Возраст должен быть от 18 до 100 лет.")
    if sex not in SEX_MAPPING:
        errors.append(f"Неизвестный пол: {sex!r}.")
    if not isinstance(bmi, (int, float)) or not (10.0 <= bmi <= 60.0):
        errors.append("ИМТ должен быть от 10.0 до 60.0.")
    if not isinstance(children, (int, float)) or not (0 <= int(children) <= 10):
        errors.append("Количество детей должно быть от 0 до 10.")
    if smoker not in SMOKER_MAPPING:
        errors.append(f"Неизвестное значение 'курильщик': {smoker!r}.")
    if region not in REGION_MAPPING:
        errors.append(f"Неизвестный регион: {region!r}.")
    return errors


def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Принимает данные из интерфейса, масштабирует их и возвращает прогноз.
    """
    # Проверка готовности модели
    if model is None:
        return (
            "❌ Модель не загружена. "
            f"Убедитесь, что файл '{MODEL_PATH}' существует (запустите train.py)."
        )

    # Валидация входных данных
    errors = _validate_inputs(age, sex, bmi, children, smoker, region)
    if errors:
        return "❌ Ошибка в данных:\n" + "\n".join(f"  • {e}" for e in errors)

    try:
        sex_num = SEX_MAPPING[sex]
        smoker_num = SMOKER_MAPPING[smoker]
        region_num = REGION_MAPPING[region]

        age_scaled = (
            (age - scaling_params['age']['mean']) / scaling_params['age']['std']
        )
        bmi_scaled = (
            (bmi - scaling_params['bmi']['mean']) / scaling_params['bmi']['std']
        )
        children_scaled = (
            (children - scaling_params['children']['mean'])
            / scaling_params['children']['std']
        )

        input_data = pd.DataFrame([{
            'age': age_scaled,
            'sex': sex_num,
            'bmi': bmi_scaled,
            'children': children_scaled,
            'smoker': smoker_num,
            'region': region_num,
        }])

        predicted_cost = model.predict(input_data)[0]

        # Защита от физически невозможных значений
        if predicted_cost < 0:
            return "⚠️ Модель вернула отрицательное значение — проверьте корректность обучения."

        return f"$ {predicted_cost:,.2f}"

    except Exception as e:
        return f"❌ Непредвиденная ошибка при вычислении: {e}"


# =====================================================================
# ЭТАП 3: Построение UI
# =====================================================================

inputs = [
    gr.Slider(minimum=18, maximum=100, step=1, value=30, label="Возраст"),
    gr.Radio(
        choices=['Женщина (Female)', 'Мужчина (Male)'],
        value='Женщина (Female)',
        label="Пол",
    ),
    gr.Slider(minimum=10.0, maximum=60.0, step=0.1, value=25.0,
              label="Индекс массы тела (BMI)"),
    gr.Slider(minimum=0, maximum=10, step=1, value=0, label="Количество детей"),
    gr.Radio(
        choices=['Нет (No)', 'Да (Yes)'],
        value='Нет (No)',
        label="Курит ли пациент?",
    ),
    gr.Dropdown(
        choices=[
            'Юго-Запад (Southwest)',
            'Юго-Восток (Southeast)',
            'Северо-Запад (Northwest)',
            'Северо-Восток (Northeast)',
        ],
        value='Юго-Запад (Southwest)',
        label="Регион проживания",
    ),
]

output = gr.Textbox(label="Прогнозируемая стоимость страховки", text_align="center")

demo = gr.Interface(
    fn=predict_insurance_cost,
    inputs=inputs,
    outputs=output,
    title="🏥 Прогноз медицинских страховок",
    description=(
        "Введите данные пациента, чтобы узнать ориентировочную стоимость "
        "медицинской страховки в США.\n"
        "Модель обучена на базе данных **Medical Cost Personal Dataset** "
        "с использованием Random Forest."
    ),
)

if __name__ == "__main__":
    demo.launch(share=True)
