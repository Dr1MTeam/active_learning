# active_learning

## Датасет

CIFAR-10

## Алгоритмы

- Least Confidence (LC)
    Выбор данных с наименьшей уверенностью модели в предсказании.
    Ссылка: https://arxiv.org/pdf/cmp-lg/9407020

 - Maximum Normalized Log-Probability (MNLP)
    Использование нормализованных логарифмов вероятностей для выбора примеров.
    Ссылка: https://aclanthology.org/W17-2630.pdf

 - BALD (Bayesian Active Learning by Disagreement)
    Учитывает уменьшение энтропии при добавлении нового примера.
    Ссылка: https://arxiv.org/pdf/1112.5745

 - Expected Gradient Length (EGL)
    Выбор примеров, которые имеют наибольшее влияние на градиент функции потерь.
    Ссылка: https://openreview.net/pdf?id=ryghZJBKPS

 - Cluster Margin
    Комбинирует методы кластеризации с подходом margin sampling для выбора информативных точек.
    Ссылка: https://arxiv.org/pdf/2107.14263
