"""Тесты для app.py — валидация входных данных и предсказания."""
import sys
import os
import pytest

# Добавляем корень проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Мокаем pickle-загрузку ДО импорта app.py
import unittest.mock as mock

import numpy as np


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

class _FakeModel:
    """Заглушка модели: возвращает фиксированное значение."""
    def predict(self, X):
        return np.array([12345.67])


_FAKE_SCALING_PARAMS = {
    'age':      {'mean': 39.0, 'std': 14.0},
    'bmi':      {'mean': 30.7, 'std': 6.1},
    'children': {'mean': 1.1,  'std': 1.2},
}


@pytest.fixture(autouse=True)
def patch_model(monkeypatch):
    """Подменяет модель и scaling_params в модуле app."""
    import app
    monkeypatch.setattr(app, 'model', _FakeModel())
    monkeypatch.setattr(app, 'scaling_params', _FAKE_SCALING_PARAMS)


# ---------------------------------------------------------------------------
# Тесты функции _validate_inputs
# ---------------------------------------------------------------------------

class TestValidateInputs:
    def test_valid_inputs_no_errors(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Женщина (Female)', 25.0, 1, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert errors == []

    def test_age_too_low(self):
        from app import _validate_inputs
        errors = _validate_inputs(10, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert any('Возраст' in e for e in errors)

    def test_age_too_high(self):
        from app import _validate_inputs
        errors = _validate_inputs(150, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert any('Возраст' in e for e in errors)

    def test_invalid_sex(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Неизвестно', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert any('пол' in e.lower() for e in errors)

    def test_bmi_too_low(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Женщина (Female)', 5.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert any('ИМТ' in e for e in errors)

    def test_bmi_too_high(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Женщина (Female)', 65.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert any('ИМТ' in e for e in errors)

    def test_children_negative(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Женщина (Female)', 25.0, -1, 'Нет (No)', 'Юго-Запад (Southwest)')
        assert any('дет' in e.lower() for e in errors)

    def test_invalid_smoker(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Женщина (Female)', 25.0, 0, 'Иногда', 'Юго-Запад (Southwest)')
        assert any('курильщик' in e.lower() for e in errors)

    def test_invalid_region(self):
        from app import _validate_inputs
        errors = _validate_inputs(30, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Сибирь')
        assert any('регион' in e.lower() for e in errors)

    def test_multiple_errors_collected(self):
        """Все ошибки должны собираться разом, а не падать на первой."""
        from app import _validate_inputs
        errors = _validate_inputs(5, 'X', 0.0, -1, 'Y', 'Z')
        assert len(errors) >= 4


# ---------------------------------------------------------------------------
# Тесты функции predict_insurance_cost
# ---------------------------------------------------------------------------

class TestPredictInsuranceCost:
    def test_valid_prediction_returns_dollar_string(self):
        from app import predict_insurance_cost
        result = predict_insurance_cost(
            30, 'Женщина (Female)', 25.0, 1, 'Нет (No)', 'Юго-Восток (Southeast)'
        )
        assert result.startswith('$')
        assert '12,345.67' in result

    def test_invalid_inputs_return_error_message(self):
        from app import predict_insurance_cost
        result = predict_insurance_cost(
            10, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)'
        )
        assert '❌' in result

    def test_model_none_returns_error(self, monkeypatch):
        import app
        monkeypatch.setattr(app, 'model', None)
        from app import predict_insurance_cost
        result = predict_insurance_cost(
            30, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)'
        )
        assert '❌' in result

    def test_all_regions_work(self):
        from app import predict_insurance_cost, REGION_MAPPING
        for region in REGION_MAPPING:
            result = predict_insurance_cost(
                30, 'Мужчина (Male)', 25.0, 0, 'Да (Yes)', region
            )
            assert '❌' not in result, f"Регион {region!r} вызвал ошибку: {result}"

    def test_boundary_age_values(self):
        from app import predict_insurance_cost
        for age in (18, 100):
            result = predict_insurance_cost(
                age, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)'
            )
            assert '❌' not in result, f"Граничный возраст {age} вызвал ошибку: {result}"

    def test_negative_prediction_returns_warning(self, monkeypatch):
        import app

        class NegativeModel:
            def predict(self, X):
                return np.array([-500.0])

        monkeypatch.setattr(app, 'model', NegativeModel())
        from app import predict_insurance_cost
        result = predict_insurance_cost(
            30, 'Женщина (Female)', 25.0, 0, 'Нет (No)', 'Юго-Запад (Southwest)'
        )
        assert '⚠️' in result
