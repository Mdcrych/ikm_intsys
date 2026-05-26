"""Тесты для логики train.py — класс SimpleSmokerRule и вспомогательные проверки."""
import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Импортируем только класс, не запуская весь скрипт
from importlib.util import spec_from_file_location, module_from_spec


def _load_simple_smoker_rule():
    """Динамически загружает только класс SimpleSmokerRule из train.py."""
    # Обходим глобальный код train.py, читая только класс
    import ast, types, textwrap

    src_path = os.path.join(os.path.dirname(__file__), '..', 'train.py')
    with open(src_path, encoding='utf-8') as f:
        source = f.read()

    tree = ast.parse(source)
    # Выбираем только определение класса
    class_def = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == 'SimpleSmokerRule'
    )
    module_source = ast.unparse(class_def)
    module_source = "import numpy as np\n" + module_source

    mod = types.ModuleType('_simple_smoker')
    exec(compile(module_source, '<string>', 'exec'), mod.__dict__)  # noqa: S102
    return mod.SimpleSmokerRule


SimpleSmokerRule = _load_simple_smoker_rule()


@pytest.fixture
def sample_data():
    """Небольшой синтетический датасет."""
    np.random.seed(0)
    n = 100
    smoker = np.random.randint(0, 2, n)
    # Курящие платят ~30 000, некурящие ~8 000
    charges = np.where(smoker == 1, 30000 + np.random.randn(n) * 2000,
                       8000 + np.random.randn(n) * 1000)
    X = pd.DataFrame({'smoker': smoker, 'age': np.random.randint(18, 65, n)})
    y = pd.Series(charges, name='charges')
    return X, y


class TestSimpleSmokerRule:
    def test_fit_stores_means(self, sample_data):
        X, y = sample_data
        model = SimpleSmokerRule()
        model.fit(X, y)
        assert model.cost_smoker > model.cost_nonsmoker
        assert model.cost_smoker > 0
        assert model.cost_nonsmoker > 0

    def test_predict_returns_correct_shape(self, sample_data):
        X, y = sample_data
        model = SimpleSmokerRule()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)

    def test_predict_before_fit_raises(self):
        model = SimpleSmokerRule()
        X = pd.DataFrame({'smoker': [0, 1]})
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_fit_without_smokers_raises(self):
        """Если в выборке нет курящих — должна быть ошибка."""
        X = pd.DataFrame({'smoker': [0, 0, 0]})
        y = pd.Series([5000.0, 6000.0, 7000.0])
        model = SimpleSmokerRule()
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_fit_without_nonsmokers_raises(self):
        X = pd.DataFrame({'smoker': [1, 1, 1]})
        y = pd.Series([25000.0, 27000.0, 30000.0])
        model = SimpleSmokerRule()
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_predict_only_two_unique_values(self, sample_data):
        """Предсказания могут содержать только 2 уникальных числа."""
        X, y = sample_data
        model = SimpleSmokerRule()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(set(preds)) == 2

    def test_smoker_prediction_higher(self, sample_data):
        X, y = sample_data
        model = SimpleSmokerRule()
        model.fit(X, y)
        smoker_pred = model.predict(pd.DataFrame({'smoker': [1]}))[0]
        nonsmoker_pred = model.predict(pd.DataFrame({'smoker': [0]}))[0]
        assert smoker_pred > nonsmoker_pred


class TestScalingLogic:
    """Тесты логики стандартизации признаков."""

    def test_standard_scaling(self):
        values = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        mean = values.mean()
        std = values.std()
        scaled = (values - mean) / std
        assert abs(scaled.mean()) < 1e-10
        assert abs(scaled.std() - 1.0) < 1e-10

    def test_zero_std_fallback(self):
        """При std=0 делитель должен быть 1.0 (не падать с ZeroDivisionError)."""
        values = np.array([5.0, 5.0, 5.0])
        std = values.std()
        safe_std = std if std != 0 else 1.0
        scaled = (values - values.mean()) / safe_std
        assert (scaled == 0).all()
