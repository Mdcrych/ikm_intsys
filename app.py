import gradio as gr
import pandas as pd
import pickle

# =====================================================================
# ЭТАП 1: Установка и базовый каркас (Загрузка модели)
# =====================================================================

# Загружаем сохраненную модель и параметры масштабирования
try:
    with open('ikm/best_insurance_model.pkl', 'rb') as file:
        model_data = pickle.load(file)
        
    model = model_data['model']
    scaling_params = model_data['scaling_params']
    print("Модель успешно загружена!")
except FileNotFoundError:
    print("Ошибка: Файл 'best_insurance_model.pkl' не найден. Сначала запустите train.py")
    exit()

# =====================================================================
# ЭТАП 2: Адаптация под задачу (Функция предсказания)
# =====================================================================

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Эта функция принимает данные из интерфейса, подготавливает их
    точно так же, как мы делали при обучении, и возвращает прогноз.
    """
    
    # 1. Превращаем текст из интерфейса обратно в числа (маппинг)
    sex_mapping = {'Женщина (Female)': 0, 'Мужчина (Male)': 1}
    smoker_mapping = {'Нет (No)': 0, 'Да (Yes)': 1}
    region_mapping = {
        'Юго-Запад (Southwest)': 0, 
        'Юго-Восток (Southeast)': 1, 
        'Северо-Запад (Northwest)': 2, 
        'Северо-Восток (Northeast)': 3
    }
    
    sex_num = sex_mapping[sex]
    smoker_num = smoker_mapping[smoker]
    region_num = region_mapping[region]
    
    # 2. Масштабируем числовые признаки с использованием сохраненных параметров
    age_scaled = (age - scaling_params['age']['mean']) / scaling_params['age']['std']
    bmi_scaled = (bmi - scaling_params['bmi']['mean']) / scaling_params['bmi']['std']
    children_scaled = (children - scaling_params['children']['mean']) / scaling_params['children']['std']
    
    # 3. Собираем данные в таблицу DataFrame (важно соблюдать порядок колонок из обучения!)
    input_data = pd.DataFrame([{
        'age': age_scaled,
        'sex': sex_num,
        'bmi': bmi_scaled,
        'children': children_scaled,
        'smoker': smoker_num,
        'region': region_num
    }])
    
    # 4. Делаем предсказание
    predicted_cost = model.predict(input_data)[0]
    
    # Возвращаем красиво отформатированную строку
    return f"$ {predicted_cost:,.2f}"

# =====================================================================
# ЭТАП 3: Запуск, тестирование и улучшение (Построение UI)
# =====================================================================

# Описываем элементы интерфейса
inputs = [
    gr.Slider(minimum=18, maximum=100, step=1, value=30, label="Возраст"),
    gr.Radio(choices=['Женщина (Female)', 'Мужчина (Male)'], value='Женщина (Female)', label="Пол"),
    gr.Slider(minimum=15.0, maximum=55.0, step=0.1, value=25.0, label="Индекс массы тела (BMI)"),
    gr.Slider(minimum=0, maximum=10, step=1, value=0, label="Количество детей"),
    gr.Radio(choices=['Нет (No)', 'Да (Yes)'], value='Нет (No)', label="Курит ли пациент?"),
    gr.Dropdown(
        choices=['Юго-Запад (Southwest)', 'Юго-Восток (Southeast)', 'Северо-Запад (Northwest)', 'Северо-Восток (Northeast)'], 
        value='Юго-Запад (Southwest)', 
        label="Регион проживания"
    )
]

output = gr.Textbox(label="Прогнозируемая стоимость страховки", text_align="center")

# Создаем само веб-приложение (УБРАЛИ КОНФЛИКТУЮЩИЕ ПАРАМЕТРЫ)
demo = gr.Interface(
    fn=predict_insurance_cost,
    inputs=inputs,
    outputs=output,
    title="🏥 Прогноз медицинских страховок",
    description="""Введите данные пациента, чтобы узнать ориентировочную стоимость медицинской страховки в США. 
                   Модель обучена на базе данных **Medical Cost Personal Dataset** с использованием Random Forest."""
)

# Запускаем сервер
if __name__ == "__main__":
    demo.launch(share=True)